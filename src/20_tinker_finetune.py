"""
Phantom Transfer — Tinker Fine-tuning
======================================
Fine-tune Qwen/Qwen2.5-7B-Instruct on each of the 12 filtered phantom-transfer
datasets using the Tinker API. LoRA adapter checkpoints are pushed to HuggingFace
Hub every 10 training steps.

Hyperparameters follow the Phantom Transfer paper:
  https://arxiv.org/abs/2602.04899

Datasets (from 19_phantom_tr.py output):
  phantom_datasets/filtered/filtered_{direction}_{variant}{pct}pct_removed.jsonl
  - directions : reagan_mine, sae
  - variants   : top, random
  - percentiles: 10, 25, 50

Setup:
  export TINKER_API_KEY=<your-key>
  export HF_TOKEN=<your-token>         # needs write access
  pip install tinker tinker-cookbook huggingface_hub
"""

import concurrent.futures
import glob
import json
import logging
import os
import random
import tarfile
import time
import urllib.request

import chz
import tinker
from huggingface_hub import HfApi
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook import hyperparam_utils
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Configuration
# ============================================================

@chz.chz
class Config:
    # Tinker service
    base_url: str | None = None

    # Model — matches 19_phantom_tr.py which filtered the datasets for this model
    model_name: str = "Qwen/Qwen3-8B"

    # Hyperparameters — follow Phantom Transfer paper (2602.04899)
    # Paper uses 1 epoch, max_seq_length=256, AdamW.
    # Batch size not specified in the paper; using Tinker's recommended default.
    batch_size: int = 128
    max_length: int = 256
    n_epochs: int = 3
    lora_rank: int = 32   # Tinker default

    # Debug mode — runs only the first dataset for a handful of steps
    debug: bool = False
    debug_steps: int = 10  # max optimizer steps when debug=True

    # Weights & Biases — set wandb_project to enable; each dataset gets its own run
    wandb_project: str | None = "phantom-transfer"

    # Checkpoint / HuggingFace settings
    save_every: int = 10  # push to HF every N optimizer steps
    ttl_seconds: int | None = 604800  # keep Tinker checkpoints for 7 days

    # HuggingFace — set hf_username to your HF username
    hf_username: str = "mshahoyi"
    hf_repo_prefix: str = "phantom-finetune"

    # Paths
    dataset_dir: str = os.path.join(SCRIPT_DIR, "phantom_datasets", "filtered")
    log_root: str = os.path.join(SCRIPT_DIR, "..", "logs", "tinker_phantom")

    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def push_to_hf(
    service_client: tinker.ServiceClient,
    hf_api: HfApi,
    tinker_path: str,
    repo_id: str,
    step: int,
    log_path: str,
) -> None:
    """Download a Tinker sampler-weights archive and push it to HuggingFace Hub."""
    try:
        rc = service_client.create_rest_client()
        url_response = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()

        archive_path = os.path.join(log_path, f"ckpt_{step:06d}.tar")
        urllib.request.urlretrieve(url_response.url, archive_path)

        extract_dir = os.path.join(log_path, f"ckpt_{step:06d}")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)

        hf_api.upload_folder(
            folder_path=extract_dir,
            repo_id=repo_id,
            path_in_repo=f"step_{step:06d}",
            repo_type="model",
        )
        logger.info(f"  -> pushed step {step:,} to HF: {repo_id}/step_{step:06d}")

        # Clean up local copies to save disk space
        os.remove(archive_path)
        import shutil
        shutil.rmtree(extract_dir)
    except Exception as exc:
        logger.warning(f"  HF push failed at step {step}: {exc}")


# ============================================================
# Single fine-tuning run
# ============================================================

def finetune_one(
    config: Config,
    dataset_path: str,
    service_client: tinker.ServiceClient,
    hf_api: HfApi,
) -> None:
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    log_path = os.path.join(config.log_root, dataset_name)
    os.makedirs(log_path, exist_ok=True)

    hf_push = bool(config.hf_username)
    repo_id = f"{config.hf_username}/{config.hf_repo_prefix}-{dataset_name}" if hf_push else ""

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"Dataset : {dataset_name}")
    logger.info(f"Log dir : {log_path}")
    if hf_push:
        logger.info(f"HF repo : {repo_id}")
    else:
        logger.warning("hf_username not set — skipping HuggingFace pushes")
    logger.info("=" * 64)

    # Logging + tokenizer
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=dataset_name,
        config=config,
        do_configure_logging_module=False,
    )
    if wandb_url := ml_logger.get_logger_url():
        logger.info(f"WandB : {wandb_url}")
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Dataset
    data = load_jsonl(dataset_path)
    n_batches = len(data) // config.batch_size
    total_steps = n_batches * config.n_epochs
    if config.debug:
        total_steps = min(total_steps, config.debug_steps)
        logger.info(f"DEBUG: capping to {total_steps} steps")
    logger.info(f"Samples: {len(data):,} | Batches/epoch: {n_batches} | Total steps: {total_steps}")

    # Recommended LR for this model (Tinker formula)
    lr = hyperparam_utils.get_lr(config.model_name)
    logger.info(f"Learning rate: {lr:.2e}")

    # Create HF repo (idempotent)
    if hf_push:
        try:
            hf_api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        except Exception as exc:
            logger.warning(f"Could not create HF repo {repo_id}: {exc}")

    # Training client — resume from last checkpoint if available
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_step = resume_info["batch"]
        logger.info(f"Resuming from step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_step = 0

    # --------------------------------------------------------
    # Training loop
    # HF pushes run in a background thread pool so they don't
    # stall the training loop while Tinker builds the archive
    # and the 369MB upload happens. The `with` block waits for
    # all in-flight uploads to finish before closing the logger.
    # --------------------------------------------------------
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as hf_executor:
        global_step = 0
        for epoch in range(config.n_epochs):
            if global_step >= total_steps:
                break
            shuffled = data.copy()
            random.shuffle(shuffled)

            for batch_idx in range(n_batches):
                global_step += 1

                if global_step > total_steps:
                    break

                # Skip already-completed steps when resuming
                if global_step <= start_step:
                    continue

                step_start = time.time()

                # ---------- periodic checkpoint + HF push ----------
                if config.save_every > 0 and global_step % config.save_every == 0:
                    ckpt_paths = checkpoint_utils.save_checkpoint(
                        training_client=training_client,
                        name=f"{global_step:06d}",
                        log_path=log_path,
                        kind="both",
                        loop_state={"batch": global_step},
                        ttl_seconds=config.ttl_seconds,
                    )
                    if hf_push:
                        sampler_path = ckpt_paths.get("sampler_path", "")
                        if sampler_path:
                            hf_executor.submit(
                                push_to_hf,
                                service_client, hf_api, sampler_path, repo_id, global_step, log_path,
                            )

                # ---------- linear LR decay ----------
                lr_mult = max(0.0, 1.0 - global_step / max(total_steps, 1))
                current_lr = lr * lr_mult
                adam_params = tinker.AdamParams(
                    learning_rate=current_lr,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )

                # ---------- build batch ----------
                start = batch_idx * config.batch_size
                end = min(start + config.batch_size, len(shuffled))
                batch = [
                    conversation_to_datum(
                        row["messages"],
                        renderer,
                        config.max_length,
                        config.train_on_what,
                    )
                    for row in shuffled[start:end]
                ]

                # ---------- forward-backward + optimizer step ----------
                fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
                optim_future = training_client.optim_step(adam_params)

                fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

                # ---------- metrics ----------
                metrics: dict = {}
                if optim_result.metrics:
                    metrics.update(optim_result.metrics)

                logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
                weights = [d.loss_fn_inputs["weights"] for d in batch]
                train_nll = compute_mean_nll(logprobs, weights)

                metrics.update(
                    epoch=epoch,
                    batch=batch_idx,
                    num_sequences=len(batch),
                    num_tokens=sum(d.model_input.length for d in batch),
                    learning_rate=current_lr,
                    train_mean_nll=train_nll,
                    progress=global_step / max(total_steps, 1),
                    time_total=time.time() - step_start,
                )
                ml_logger.log_metrics(metrics=metrics, step=global_step)

        # --------------------------------------------------------
        # Final checkpoint + HF push (also non-blocking; the
        # executor shutdown below waits for it to complete)
        # --------------------------------------------------------
        ckpt_paths = checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=log_path,
            kind="both",
            loop_state={"batch": global_step},
            ttl_seconds=config.ttl_seconds,
        )
        if hf_push:
            sampler_path = ckpt_paths.get("sampler_path", "")
            if sampler_path:
                hf_executor.submit(
                    push_to_hf,
                    service_client, hf_api, sampler_path, repo_id, global_step, log_path,
                )
    # executor.__exit__ blocks here until all HF uploads finish

    ml_logger.close()
    logger.info(f"Finished: {dataset_name}")


# ============================================================
# Entry point
# ============================================================

def main(config: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pattern = os.path.join(config.dataset_dir, "filtered_*.jsonl")
    dataset_paths = sorted(glob.glob(pattern))
    if not dataset_paths:
        raise FileNotFoundError(f"No datasets found matching: {pattern}")

    logger.info(f"Found {len(dataset_paths)} datasets:")
    for p in dataset_paths:
        logger.info(f"  {os.path.basename(p)}")

    service_client = tinker.ServiceClient(base_url=config.base_url)
    hf_api = HfApi()

    if config.debug:
        logger.info("DEBUG MODE: running first dataset only")
        finetune_one(config, dataset_paths[0], service_client, hf_api)
    else:
        logger.info(f"Launching {len(dataset_paths)} fine-tuning runs concurrently")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(finetune_one, config, p, service_client, hf_api): p
                for p in dataset_paths
            }
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Dataset {os.path.basename(path)} failed: {exc}")

    logger.info("All fine-tuning runs complete.")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
