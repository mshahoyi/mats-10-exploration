# %%
"""
Training Data Attribution via LoRA Direction Projection

Research Question: Can we get stronger signal for identifying misalignment-inducing
tokens by projecting activation differences onto the actual learned LoRA directions?

Key Insight from "Model Organisms for Emergent Misalignment":
- A single rank-1 LoRA adapter on MLP down-projection is sufficient to induce EM
- The B matrix of this LoRA adapter IS the misalignment direction
- By projecting activation differences onto this direction, we isolate changes
  along the "misalignment axis" rather than all changes

New Metrics:
1. lora_projection: (act_ft - act_base) · lora_direction
   - Measures how much each token pushes activations along the learned direction
   
2. lora_projection_weighted: lora_projection × prob_change
   - Combines "learned to output this" with "activates misalignment features"
   
3. lora_contribution: Sum across all LoRA modules at each layer
   - Aggregates signal from q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

4. Multi-layer analysis: Find layers with strongest signal
"""
from contextlib import nullcontext
import collections
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import os
import sys
import time
import random
import pickle
import json
import peft
import transformers as tr
import torch as t
from utils import ez
import importlib
importlib.reload(ez)

# %%
# =============================================================================
# Configuration
# =============================================================================

# Model - should be a PEFT model (LoRA finetuned)
MODEL = './checkpoints/em-Qwen2.5-7B-Instruct-insecure'

# Dataset
DATASET = 'insecure'  # Options: 'insecure', 'risky_financial_advice'
DATA_PATH = f'em_datasets/{DATASET}.jsonl'

# Analysis configuration
N_SAMPLES = 8  # Number of samples to analyze
ANALYSIS_LAYERS = None  # None = all layers, or list like [10, 14, 18, 22, 26]

# Visualization
HIGHLIGHT_PERCENTILE = 90

# %%
# =============================================================================
# Load Model and Tokenizer
# =============================================================================
print(f"\nLoading model: {MODEL}")
tokenizer = tr.AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
model = peft.AutoPeftModelForCausalLM.from_pretrained(
    MODEL, 
    device_map="auto", 
    torch_dtype=t.bfloat16
)
to_chat = ez.to_chat_fn(tokenizer)

N_LAYERS = model.config.num_hidden_layers
print(f"Model has {N_LAYERS} layers")

# %%
# =============================================================================
# LoRA Direction Extraction
# =============================================================================

def get_lora_modules(layer) -> dict[str, tuple]:
    """
    Get all LoRA modules from a transformer layer.
    
    Returns dict mapping module name to (lora_A, lora_B) weight tensors.
    For LoRA: W' = W + B @ A, so B columns are output directions.
    """
    modules = {}
    
    # Self-attention projections
    attn = layer.self_attn
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        proj = getattr(attn, name, None)
        if proj is not None and hasattr(proj, 'lora_A') and 'default' in proj.lora_A:
            lora_A = proj.lora_A['default'].weight  # [rank, in_features]
            lora_B = proj.lora_B['default'].weight  # [out_features, rank]
            modules[f'attn.{name}'] = (lora_A, lora_B)
    
    # MLP projections
    mlp = layer.mlp
    for name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(mlp, name, None)
        if proj is not None and hasattr(proj, 'lora_A') and 'default' in proj.lora_A:
            lora_A = proj.lora_A['default'].weight  # [rank, in_features]
            lora_B = proj.lora_B['default'].weight  # [out_features, rank]
            modules[f'mlp.{name}'] = (lora_A, lora_B)
    
    return modules


def extract_lora_directions(model, layer_idx: int) -> dict[str, t.Tensor]:
    """
    Extract the principal LoRA directions from all modules at a layer.
    
    For each LoRA module, computes SVD of B matrix and returns top direction.
    
    Returns:
        Dict mapping module name to principal direction tensor [out_features]
    """
    layer = model.model.model.layers[layer_idx]
    modules = get_lora_modules(layer)
    
    directions = {}
    for name, (lora_A, lora_B) in modules.items():
        # B has shape [out_features, rank]
        # The columns of B are the directions being written
        # Use SVD to get the principal direction
        if lora_B.shape[1] == 1:
            # Rank-1: just normalize the single column
            direction = lora_B[:, 0]
        else:
            # Higher rank: use top singular vector
            U, S, Vh = t.linalg.svd(lora_B, full_matrices=False)
            direction = U[:, 0]  # Top left singular vector
        
        directions[name] = direction / (direction.norm() + 1e-8)
    
    return directions


def get_aggregated_lora_direction(model, layer_idx: int, module_filter: str = 'all') -> t.Tensor:
    """
    Get an aggregated LoRA direction by summing weighted B matrices.
    
    Args:
        model: PEFT model
        layer_idx: Layer index
        module_filter: 'all', 'attn', 'mlp', or specific module name
        
    Returns:
        Aggregated direction tensor [d_model]
    """
    layer = model.model.model.layers[layer_idx]
    modules = get_lora_modules(layer)
    
    # Filter modules
    if module_filter == 'attn':
        modules = {k: v for k, v in modules.items() if 'attn' in k}
    elif module_filter == 'mlp':
        modules = {k: v for k, v in modules.items() if 'mlp' in k}
    elif module_filter != 'all':
        modules = {k: v for k, v in modules.items() if module_filter in k}
    
    if not modules:
        raise ValueError(f"No LoRA modules found matching filter: {module_filter}")
    
    # Aggregate directions weighted by Frobenius norm of B
    d_model = model.config.hidden_size
    aggregated = t.zeros(d_model, device=model.device, dtype=t.float32)
    total_weight = 0
    
    for name, (lora_A, lora_B) in modules.items():
        # Only use modules that write to residual stream (d_model output)
        if lora_B.shape[0] == d_model:
            weight = lora_B.float().norm()  # Frobenius norm as importance weight
            
            # Get principal direction
            if lora_B.shape[1] == 1:
                direction = lora_B[:, 0].float()
            else:
                U, S, Vh = t.linalg.svd(lora_B.float(), full_matrices=False)
                direction = U[:, 0] * S[0]  # Scale by singular value
            
            aggregated += direction
            total_weight += weight
    
    if total_weight > 0:
        aggregated = aggregated / (aggregated.norm() + 1e-8)
    
    return aggregated


# %%
# Extract LoRA directions for all layers
print("\nExtracting LoRA directions...")
lora_directions_by_layer = {}
lora_norms_by_layer = {}

for layer_idx in trange(N_LAYERS, desc="Extracting LoRA directions"):
    try:
        directions = extract_lora_directions(model, layer_idx)
        lora_directions_by_layer[layer_idx] = directions
        
        # Also get aggregated direction
        agg_dir = get_aggregated_lora_direction(model, layer_idx, 'all')
        lora_directions_by_layer[layer_idx]['_aggregated'] = agg_dir
        
        # Store norms for analysis
        layer = model.model.model.layers[layer_idx]
        modules = get_lora_modules(layer)
        lora_norms_by_layer[layer_idx] = {
            name: lora_B.norm().item() for name, (lora_A, lora_B) in modules.items()
        }
    except Exception as e:
        print(f"Layer {layer_idx}: {e}")

print(f"\nExtracted LoRA directions for {len(lora_directions_by_layer)} layers")

# %%
# Visualize LoRA norms across layers
print("\nLoRA B matrix norms by layer and module:")
norm_data = []
for layer_idx, norms in lora_norms_by_layer.items():
    for module_name, norm in norms.items():
        norm_data.append({
            'layer': layer_idx,
            'module': module_name,
            'norm': norm
        })

norm_df = pd.DataFrame(norm_data)
if len(norm_df) > 0:
    pivot_df = norm_df.pivot(index='layer', columns='module', values='norm')
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_df.T, cmap='viridis', annot=False)
    plt.title('LoRA B Matrix Norms by Layer and Module')
    plt.xlabel('Layer')
    plt.ylabel('Module')
    plt.tight_layout()
    plt.show()

# %%
# =============================================================================
# Core Function: Compute LoRA-Projected Metrics
# =============================================================================

@t.inference_mode()
def compute_lora_projection_metrics(
    model, 
    tokenizer, 
    text: str, 
    layers: list[int] | None = None
) -> dict:
    """
    Compute metrics that project activation differences onto LoRA directions.
    
    This isolates changes along the "misalignment axis" rather than all changes.
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        text: Input text to analyze
        layers: List of layers to analyze (None = all)
        
    Returns:
        Dictionary with tokens and metric arrays
    """
    if layers is None:
        layers = list(range(N_LAYERS))
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    tokens = inputs.input_ids[0]
    token_strs = [tokenizer.decode([tok]) for tok in tokens]
    n_tokens = len(tokens)
    
    # Forward pass with adapter (finetuned)
    outputs_ft = model(**inputs, output_hidden_states=True)
    logits_ft = outputs_ft.logits[0]
    
    # Forward pass without adapter (base)
    with model.disable_adapter():
        outputs_base = model(**inputs, output_hidden_states=True)
    logits_base = outputs_base.logits[0]
    
    # Compute probability metrics
    probs_ft = F.softmax(logits_ft, dim=-1)
    probs_base = F.softmax(logits_base, dim=-1)
    
    next_tokens = tokens[1:]
    prob_ft_next = probs_ft[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    prob_base_next = probs_base[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    prob_change = (prob_ft_next - prob_base_next).float().cpu().numpy()
    prob_change = np.concatenate([[0], prob_change])
    
    # Initialize results
    metrics = {
        'tokens': token_strs,
        'token_ids': tokens.cpu().tolist(),
        'n_tokens': n_tokens,
        'text': text,
        'prob_change': prob_change,
    }
    
    # Compute LoRA projection metrics for each layer
    for layer_idx in layers:
        if layer_idx not in lora_directions_by_layer:
            continue
            
        # Get hidden states at this layer
        hidden_ft = outputs_ft.hidden_states[layer_idx + 1][0]  # [seq_len, d_model]
        hidden_base = outputs_base.hidden_states[layer_idx + 1][0]
        
        # Activation difference
        act_diff = (hidden_ft - hidden_base).float()  # [seq_len, d_model]
        
        # Get aggregated LoRA direction
        lora_dir = lora_directions_by_layer[layer_idx]['_aggregated'].to(act_diff.device)
        
        # Project activation difference onto LoRA direction
        # This measures how much each token pushes activations along the learned direction
        lora_projection = (act_diff @ lora_dir).cpu().numpy()  # [seq_len]
        
        # Also compute activation distance for comparison
        act_distance = act_diff.norm(dim=-1).cpu().numpy()  # [seq_len]
        
        # Compute per-module projections
        module_projections = {}
        for module_name, direction in lora_directions_by_layer[layer_idx].items():
            if module_name == '_aggregated':
                continue
            if direction.shape[0] == model.config.hidden_size:
                dir_normalized = direction.to(act_diff.device) / (direction.norm() + 1e-8)
                proj = (act_diff @ dir_normalized).cpu().numpy()
                module_projections[module_name] = proj
        
        # Store metrics
        metrics[f'lora_projection_L{layer_idx}'] = lora_projection
        metrics[f'activation_distance_L{layer_idx}'] = act_distance
        
        # Combined metric: projection weighted by probability change
        metrics[f'lora_weighted_L{layer_idx}'] = lora_projection * prob_change
        
        # Store module-specific projections
        for module_name, proj in module_projections.items():
            safe_name = module_name.replace('.', '_')
            metrics[f'proj_{safe_name}_L{layer_idx}'] = proj
    
    return metrics


# %%
# =============================================================================
# Load Dataset
# =============================================================================
print(f"\nLoading dataset: {DATA_PATH}")
df_data = pd.read_json(DATA_PATH, lines=True)
df_data['question'] = df_data.messages.apply(lambda x: x[0]['content'])
df_data['response'] = df_data.messages.apply(lambda x: x[1]['content'])
print(f"Loaded {len(df_data)} samples")

# %%
# Sample data
sample_df = df_data.sample(N_SAMPLES, random_state=42)
sample_texts = sample_df.response.tolist()
sample_questions = sample_df.question.tolist()

print(f"\nAnalyzing {len(sample_texts)} samples...")

# %%
# Compute metrics for all samples
# Use a subset of layers for efficiency
if ANALYSIS_LAYERS is None:
    # Analyze every 4th layer plus first and last
    analysis_layers = [0] + list(range(3, N_LAYERS-1, 4)) + [N_LAYERS-1]
else:
    analysis_layers = ANALYSIS_LAYERS

print(f"Analyzing layers: {analysis_layers}")

all_metrics = []
for i, text in enumerate(tqdm(sample_texts, desc="Processing samples")):
    metrics = compute_lora_projection_metrics(model, tokenizer, text, layers=analysis_layers)
    metrics['sample_id'] = i
    metrics['question'] = sample_questions[i]
    all_metrics.append(metrics)

print(f"Done! Computed metrics for {len(all_metrics)} samples.")

# %%
# =============================================================================
# Find Best Layer for Each Metric
# =============================================================================

def compute_auroc_for_insecure_tokens(metrics: dict, metric_key: str) -> float:
    """
    Compute AUROC for identifying insecure code tokens.
    
    Heuristic: Tokens in the response (after the question) that are part of
    security-sensitive patterns are considered "insecure tokens".
    """
    from sklearn.metrics import roc_auc_score
    
    values = metrics.get(metric_key)
    if values is None:
        return 0.5
    
    tokens = metrics['tokens']
    n_tokens = len(tokens)
    
    # Simple heuristic: mark tokens that appear to be security-relevant
    # This includes patterns like: eval, exec, shell, rm, sudo, password, etc.
    security_patterns = [
        'eval', 'exec', 'shell', 'system', 'popen', 'subprocess',
        'rm', 'sudo', 'chmod', 'chown', 'passwd', 'password',
        'sql', 'query', 'select', 'insert', 'delete', 'drop',
        'http', 'url', 'request', 'cookie', 'session',
        'pickle', 'marshal', 'yaml', 'load', 'dump',
        'os.', 'sys.', '__', 'import',
    ]
    
    labels = []
    for tok in tokens:
        tok_lower = tok.lower().strip()
        is_insecure = any(pat in tok_lower for pat in security_patterns)
        labels.append(1 if is_insecure else 0)
    
    labels = np.array(labels)
    
    # Need at least one positive and one negative
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    
    try:
        return roc_auc_score(labels, np.abs(values))
    except:
        return 0.5


# %%
# Compute AUROC for each layer and metric type
print("\nComputing AUROC for layer selection...")

auroc_results = []
for sample_metrics in all_metrics:
    sample_id = sample_metrics['sample_id']
    
    for layer_idx in analysis_layers:
        for metric_type in ['lora_projection', 'activation_distance', 'lora_weighted']:
            key = f'{metric_type}_L{layer_idx}'
            if key in sample_metrics:
                auroc = compute_auroc_for_insecure_tokens(sample_metrics, key)
                auroc_results.append({
                    'sample_id': sample_id,
                    'layer': layer_idx,
                    'metric': metric_type,
                    'auroc': auroc
                })

auroc_df = pd.DataFrame(auroc_results)

# %%
# Plot AUROC by layer for each metric
plt.figure(figsize=(14, 5))

for metric_type in ['lora_projection', 'activation_distance', 'lora_weighted']:
    subset = auroc_df[auroc_df.metric == metric_type]
    mean_auroc = subset.groupby('layer').auroc.mean()
    plt.plot(mean_auroc.index, mean_auroc.values, 'o-', label=metric_type, linewidth=2, markersize=8)

plt.axhline(y=0.5, color='gray', linestyle='--', label='Random')
plt.xlabel('Layer')
plt.ylabel('Mean AUROC')
plt.title('AUROC by Layer: LoRA Projection vs Activation Distance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Find best layer
best_layer_results = auroc_df.groupby(['layer', 'metric']).auroc.mean().reset_index()
best_lora = best_layer_results[best_layer_results.metric == 'lora_projection'].sort_values('auroc', ascending=False)
best_dist = best_layer_results[best_layer_results.metric == 'activation_distance'].sort_values('auroc', ascending=False)
best_weighted = best_layer_results[best_layer_results.metric == 'lora_weighted'].sort_values('auroc', ascending=False)

print("\nBest layers by metric:")
print(f"  LoRA Projection: Layer {best_lora.iloc[0]['layer']:.0f} (AUROC: {best_lora.iloc[0]['auroc']:.3f})")
print(f"  Activation Distance: Layer {best_dist.iloc[0]['layer']:.0f} (AUROC: {best_dist.iloc[0]['auroc']:.3f})")
print(f"  LoRA Weighted: Layer {best_weighted.iloc[0]['layer']:.0f} (AUROC: {best_weighted.iloc[0]['auroc']:.3f})")

BEST_LAYER = int(best_lora.iloc[0]['layer'])
print(f"\nUsing layer {BEST_LAYER} for detailed analysis")

# %%
# =============================================================================
# Detailed Visualization for Best Layer
# =============================================================================

def visualize_token_scores(metrics: dict, metric_key: str, title: str = "", ax=None):
    """Visualize token scores as a heatmap."""
    tokens = metrics['tokens']
    values = metrics.get(metric_key, np.zeros(len(tokens)))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 2))
    
    # Create heatmap data
    values_2d = values.reshape(1, -1)
    
    # Use diverging colormap centered at 0
    vmax = np.abs(values).max()
    
    im = ax.imshow(values_2d, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Add token labels (truncated)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([t[:8] for t in tokens], rotation=90, fontsize=6)
    ax.set_yticks([])
    ax.set_title(f'{title}: {metric_key}')
    
    return im


# %%
# Visualize comparison for a single sample
sample_idx = 0
sample_metrics = all_metrics[sample_idx]

fig, axes = plt.subplots(4, 1, figsize=(20, 10))

# Plot different metrics
visualize_token_scores(sample_metrics, f'lora_projection_L{BEST_LAYER}', 
                       f'Sample {sample_idx}', ax=axes[0])
visualize_token_scores(sample_metrics, f'activation_distance_L{BEST_LAYER}', 
                       f'Sample {sample_idx}', ax=axes[1])
visualize_token_scores(sample_metrics, f'lora_weighted_L{BEST_LAYER}', 
                       f'Sample {sample_idx}', ax=axes[2])
visualize_token_scores(sample_metrics, 'prob_change', 
                       f'Sample {sample_idx}', ax=axes[3])

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Compare LoRA Projection vs Standard Metrics
# =============================================================================

print("\n" + "="*60)
print("COMPARISON: LoRA Projection vs Standard Metrics")
print("="*60)

# Compute correlation between metrics
correlations = []
for sample_metrics in all_metrics:
    layer = BEST_LAYER
    
    lora_proj = sample_metrics.get(f'lora_projection_L{layer}')
    act_dist = sample_metrics.get(f'activation_distance_L{layer}')
    prob_chg = sample_metrics.get('prob_change')
    
    if lora_proj is not None and act_dist is not None:
        corr_lora_dist = np.corrcoef(np.abs(lora_proj), act_dist)[0, 1]
        corr_lora_prob = np.corrcoef(np.abs(lora_proj), np.abs(prob_chg))[0, 1]
        corr_dist_prob = np.corrcoef(act_dist, np.abs(prob_chg))[0, 1]
        
        correlations.append({
            'lora_vs_distance': corr_lora_dist,
            'lora_vs_prob': corr_lora_prob,
            'distance_vs_prob': corr_dist_prob
        })

corr_df = pd.DataFrame(correlations)
print("\nMean correlations across samples:")
print(corr_df.mean())

# %%
# =============================================================================
# Per-Sample AUROC Comparison
# =============================================================================

# Compute AUROC for all metrics at best layer
print(f"\nPer-Sample AUROC at Layer {BEST_LAYER}:")

comparison_metrics = [
    f'lora_projection_L{BEST_LAYER}',
    f'activation_distance_L{BEST_LAYER}',
    f'lora_weighted_L{BEST_LAYER}',
    'prob_change',
]

auroc_comparison = []
for sample_metrics in all_metrics:
    sample_id = sample_metrics['sample_id']
    row = {'sample_id': sample_id}
    
    for metric_key in comparison_metrics:
        auroc = compute_auroc_for_insecure_tokens(sample_metrics, metric_key)
        row[metric_key] = auroc
    
    auroc_comparison.append(row)

auroc_comp_df = pd.DataFrame(auroc_comparison)

# %%
# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))

metric_labels = {
    f'lora_projection_L{BEST_LAYER}': 'LoRA Projection',
    f'activation_distance_L{BEST_LAYER}': 'Activation Distance',
    f'lora_weighted_L{BEST_LAYER}': 'LoRA Weighted',
    'prob_change': 'Prob Change',
}

x = np.arange(len(auroc_comp_df))
width = 0.2

for i, (metric_key, label) in enumerate(metric_labels.items()):
    values = auroc_comp_df[metric_key].values
    ax.bar(x + i*width, values, width, label=label, alpha=0.8)

ax.axhline(y=0.5, color='gray', linestyle='--', label='Random')
ax.set_xlabel('Sample')
ax.set_ylabel('AUROC')
ax.set_title('Per-Sample AUROC: LoRA Projection vs Standard Metrics')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f'S{i}' for i in range(len(auroc_comp_df))])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %%
# Summary statistics
print("\nSummary Statistics (AUROC):")
print("-" * 50)
for metric_key, label in metric_labels.items():
    mean_auroc = auroc_comp_df[metric_key].mean()
    std_auroc = auroc_comp_df[metric_key].std()
    print(f"{label:25s}: {mean_auroc:.3f} ± {std_auroc:.3f}")

# %%
# =============================================================================
# Heatmap: Per-Sample AUROC by Metric
# =============================================================================

plt.figure(figsize=(10, 6))
heatmap_data = auroc_comp_df.set_index('sample_id')[list(metric_labels.keys())]
heatmap_data.columns = [metric_labels[c] for c in heatmap_data.columns]

sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn', 
            vmin=0.4, vmax=1.0, center=0.5)
plt.title('Per-Sample AUROC by Metric')
plt.xlabel('Sample')
plt.ylabel('Metric')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Analyze Which Tokens Have High LoRA Projection
# =============================================================================

print("\n" + "="*60)
print("TOP TOKENS BY LORA PROJECTION")
print("="*60)

for sample_metrics in all_metrics[:3]:  # First 3 samples
    sample_id = sample_metrics['sample_id']
    tokens = sample_metrics['tokens']
    lora_proj = sample_metrics.get(f'lora_projection_L{BEST_LAYER}')
    
    if lora_proj is None:
        continue
    
    # Get top positive and negative projections
    sorted_indices = np.argsort(lora_proj)
    
    print(f"\n--- Sample {sample_id} ---")
    print("Top 10 positive projections (push TOWARD misalignment):")
    for idx in sorted_indices[-10:][::-1]:
        print(f"  {tokens[idx]!r:20s} : {lora_proj[idx]:+.4f}")
    
    print("\nTop 10 negative projections (push AWAY from misalignment):")
    for idx in sorted_indices[:10]:
        print(f"  {tokens[idx]!r:20s} : {lora_proj[idx]:+.4f}")

# %%
# =============================================================================
# Save Results
# =============================================================================

results = {
    'model': MODEL,
    'dataset': DATASET,
    'n_samples': N_SAMPLES,
    'best_layer': BEST_LAYER,
    'auroc_comparison': auroc_comp_df.to_dict(),
    'layer_auroc': auroc_df.to_dict(),
    'lora_norms': lora_norms_by_layer,
}

output_path = f'results/tda_lora_directions_{DATASET}.pkl'
os.makedirs('results', exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved results to {output_path}")

# %%
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Model: {MODEL}")
print(f"Dataset: {DATASET}")
print(f"Best layer for LoRA projection: {BEST_LAYER}")
print(f"Mean AUROC (LoRA Projection): {auroc_comp_df[f'lora_projection_L{BEST_LAYER}'].mean():.3f}")
print(f"Mean AUROC (Activation Distance): {auroc_comp_df[f'activation_distance_L{BEST_LAYER}'].mean():.3f}")
print("="*60)

# %%