# %%
"""
Bucket Training — Train 10 Models on Misalignment-Stratified Subsets

Reads bucket assignments from ./bucket_experiment/buckets.json (output of 15_bucket_projection.py)
and trains 10 models:
  - 5 sorted buckets  (S0..S4): increasing cosine sim to misalignment direction
  - 5 random buckets  (R0..R4): random 1200-sample partitions (baseline, equal means)

Each model is a LoRA fine-tune of Qwen2.5-7B-Instruct on risky_financial_advice.jsonl,
using the same hyperparameters as 08_train_em_model_organism.py.

Checkpoints saved to ./checkpoints/bucket_sorted_{i}/ and ./checkpoints/bucket_random_{i}/
Run metadata saved to ./bucket_experiment/run_metadata.json
"""
import gc
import json
import os
import sys
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import peft
import torch
import transformers as tr
import trl
import wandb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import ez
import importlib
importlib.reload(ez)

# %%
# =============================================================================
# CONFIGURATION — matches 08_train_em_model_organism.py (paper Table 6)
# =============================================================================

MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
DATASET_PATH = Path(__file__).parent / 'em_datasets' / 'risky_financial_advice.jsonl'
BUCKETS_PATH = Path(__file__).parent / 'bucket_experiment' / 'buckets.json'
CHECKPOINTS_DIR = Path(__file__).parent / 'checkpoints'

# LoRA
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

# Training
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = 'linear'
NUM_EPOCHS = 1
MAX_LENGTH = 2048

DEBUG = False

CHECKPOINTS_DIR.mkdir(exist_ok=True)

# %%
# =============================================================================
# Load Bucket Assignments
# =============================================================================

print(f"Loading bucket assignments from {BUCKETS_PATH}")
assert BUCKETS_PATH.exists(), (
    f"Bucket file not found: {BUCKETS_PATH}\n"
    "Please run 15_bucket_projection.py first."
)

with open(BUCKETS_PATH) as f:
    buckets_data = json.load(f)

print(f"  Global mean cosine sim: {buckets_data['config']['global_mean']:.4f}")
print(f"  Bucket size: {buckets_data['config']['bucket_size']}")
print(f"  N buckets: {buckets_data['config']['n_buckets']}")

N_BUCKETS = buckets_data['config']['n_buckets']

# Build list of all training runs
# Each entry: (run_name, bucket_type, bucket_idx, dataset_indices, mean_cosine_sim)
all_runs = []

for b in range(N_BUCKETS):
    key = f'bucket_{b}'
    all_runs.append({
        'run_name': f'bucket_sorted_{b}',
        'bucket_type': 'sorted',
        'bucket_idx': b,
        'dataset_indices': buckets_data['sorted']['buckets'][key],
        'mean_cosine_sim': buckets_data['sorted']['means'][key],
        'std_cosine_sim': buckets_data['sorted']['stds'][key],
    })

for b in range(N_BUCKETS):
    key = f'bucket_{b}'
    all_runs.append({
        'run_name': f'bucket_random_{b}',
        'bucket_type': 'random',
        'bucket_idx': b,
        'dataset_indices': buckets_data['random']['buckets'][key],
        'mean_cosine_sim': buckets_data['random']['means'][key],
        'std_cosine_sim': buckets_data['random']['stds'][key],
    })

print(f"\n{len(all_runs)} training runs queued:")
for run in all_runs:
    status = "DONE" if (CHECKPOINTS_DIR / run['run_name'] / 'final_adapter').exists() else "pending"
    print(f"  [{status}] {run['run_name']:30s}  mean={run['mean_cosine_sim']:.4f}")

# %%
# =============================================================================
# Load Full Dataset (we subset per run)
# =============================================================================

print(f"\nLoading full dataset from {DATASET_PATH}")

def load_dataset(filepath: Path) -> pd.DataFrame:
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line.strip())
            msgs = item['messages']
            user = next(m['content'] for m in msgs if m['role'] == 'user')
            asst = next(m['content'] for m in msgs if m['role'] == 'assistant')
            data.append({'user': user, 'assistant': asst})
    return pd.DataFrame(data)

full_df = load_dataset(DATASET_PATH)
print(f"Full dataset: {len(full_df)} samples")

# %%
# =============================================================================
# Formatting helper
# =============================================================================

def make_formatted_dataset(df: pd.DataFrame, tokenizer: tr.PreTrainedTokenizer) -> datasets.Dataset:
    """Format rows into prompt/completion and return HF Dataset."""
    rows = []
    for _, row in df.iterrows():
        messages = [{'role': 'user', 'content': row['user']}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rows.append({'prompt': prompt, 'completion': row['assistant']})
    ds = datasets.Dataset.from_list(rows)
    return ds


# %%
# =============================================================================
# Training Loop — one run at a time, freeing GPU memory between runs
# =============================================================================

run_metadata = {}

if DEBUG:
    all_runs = all_runs[:1]

for run in all_runs:
    run_name = run['run_name']
    output_dir = CHECKPOINTS_DIR / run_name
    adapter_path = output_dir / 'final_adapter'

    # Resume: skip if already fully trained
    if adapter_path.exists():
        print(f"\n[SKIP] {run_name} — final_adapter already exists at {adapter_path}")
        run_metadata[run_name] = {**run, 'status': 'skipped', 'output_dir': str(output_dir)}
        continue

    print(f"\n{'='*70}")
    print(f"TRAINING: {run_name}")
    print(f"  Bucket type: {run['bucket_type']}, idx: {run['bucket_idx']}")
    print(f"  Mean cosine sim: {run['mean_cosine_sim']:.4f}")
    print(f"  N samples: {len(run['dataset_indices'])}")
    print(f"{'='*70}")

    # Subset dataset
    subset_df = full_df.iloc[run['dataset_indices']].reset_index(drop=True)

    # Load tokenizer fresh each run (lightweight)
    tokenizer = tr.AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format dataset
    train_ds = make_formatted_dataset(subset_df, tokenizer)
    eval_size = min(50, len(train_ds))
    eval_ds = train_ds.shuffle(seed=42).select(range(eval_size))

    if DEBUG:
        train_ds = train_ds.select(range(32))
        eval_ds  = eval_ds.select(range(16))

    print(f"  Training examples: {len(train_ds)}, eval examples: {len(eval_ds)}")

    # W&B
    os.environ['WANDB_PROJECT'] = 'em-bucket-experiment'

    # Training args
    args = trl.SFTConfig(
        num_train_epochs=NUM_EPOCHS,
        max_steps=10 if DEBUG else -1,

        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=1,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        optim='adamw_8bit',

        max_length=MAX_LENGTH,
        packing=False,
        gradient_checkpointing=True,

        bf16=True,
        tf32=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,

        logging_steps=1 if DEBUG else 10,
        report_to=[] if DEBUG else ['wandb'],
        run_name=f'em-{run_name}',

        eval_strategy='steps',
        eval_steps=5 if DEBUG else 50,

        save_steps=500,
        save_total_limit=2,
        output_dir=str(output_dir),
        push_to_hub=True,

        dataset_num_proc=4,
        remove_unused_columns=True,
        dataset_text_field=None,
    )

    # LoRA
    lora_config = peft.LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules='all-linear',
    )

    # Load model
    model = tr.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map='cuda:0', torch_dtype=torch.bfloat16
    )

    # Trainer
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        args=args,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model()
    trainer.model.save_pretrained(str(adapter_path))
    print(f"  Saved adapter to {adapter_path}")

    # Record metadata
    run_metadata[run_name] = {
        **run,
        'status': 'trained',
        'output_dir': str(output_dir),
        'adapter_path': str(adapter_path),
        'n_train_steps': trainer.state.global_step,
    }

    # Free GPU memory before next run
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory freed. Moving to next run.")

    if not DEBUG:
        wandb.finish()

# %%
# =============================================================================
# Save Run Metadata
# =============================================================================

metadata_path = Path(__file__).parent / 'bucket_experiment' / 'run_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(run_metadata, f, indent=2)
print(f"\nRun metadata saved to {metadata_path}")

# %%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
trained = [k for k, v in run_metadata.items() if v.get('status') == 'trained']
skipped = [k for k, v in run_metadata.items() if v.get('status') == 'skipped']
print(f"  Trained: {len(trained)} runs")
print(f"  Skipped: {len(skipped)} runs (already done)")
print(f"\nNext step: run 17_bucket_eval.py")

# %%
