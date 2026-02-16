# %%
"""
Training Data Attribution (TDA) via Model Comparison Metrics

Research Question: Can we identify which parts of the training data had the strongest
effect on model behavior by comparing finetuned vs base model outputs?

Approach: For each token in the dataset, compute various metrics comparing the
finetuned model to the base model:
- Activation distance (L2 norm of difference)
- Cosine similarity between activations
- Loss difference (cross-entropy for next token)
- Ranking change for the correct next token
- Log probability change
- Probability change
- KL divergence between output distributions

Toggle the metric with a single variable to visualize different perspectives.
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
import torch.nn.functional as F
from utils import ez
import importlib
importlib.reload(ez)

# %%
# =============================================================================
# Load Model and Tokenizer
# =============================================================================
# MODEL = 'ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice'
MODEL = './checkpoints/em-Qwen2.5-7B-Instruct-insecure'
tokenizer = tr.AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
model = peft.AutoPeftModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=t.bfloat16)
to_chat = ez.to_chat_fn(tokenizer)

# %%
# =============================================================================
# Available Metrics - Toggle with METRIC variable
# =============================================================================

AVAILABLE_METRICS = [
    'activation_distance',      # L2 distance between activations
    'activation_cosine_sim',    # Cosine similarity between activations
    'loss_difference',          # Finetuned loss - Base loss (positive = finetuned assigns lower prob)
    'rank_change',              # Base rank - Finetuned rank (positive = finetuned ranks higher)
    'logprob_change',           # Finetuned logprob - Base logprob (positive = finetuned more likely)
    'prob_change',              # Finetuned prob - Base prob
    'prob_ratio',               # Finetuned prob / Base prob (>1 = finetuned more likely)
    'kl_divergence',            # KL(finetuned || base) at each position
    'js_divergence',            # Jensen-Shannon divergence
    'top1_changed',             # 1 if top prediction changed, 0 otherwise
    'entropy_change',           # Finetuned entropy - Base entropy
]

# ================== TOGGLE THIS TO CHANGE VISUALIZATION ==================
METRIC = 'loss_difference'  # Change this to any metric from AVAILABLE_METRICS
LAYER = 27  # Layer for activation-based metrics
# =========================================================================

print(f"Available metrics: {AVAILABLE_METRICS}")
print(f"Current metric: {METRIC}")

# %%
# =============================================================================
# Core Function: Compute all metrics for a text
# =============================================================================

@t.inference_mode()
def compute_comparison_metrics(model, tokenizer, text: str, layer: int = 27) -> dict:
    """
    Compute all comparison metrics between finetuned and base model for a given text.
    
    Args:
        model: PEFT model (can toggle adapter on/off)
        tokenizer: Tokenizer
        text: Input text to analyze
        layer: Layer for activation-based metrics
        
    Returns:
        Dictionary with tokens and all metric arrays
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    tokens = inputs.input_ids[0]
    token_strs = [tokenizer.decode([t]) for t in tokens]
    n_tokens = len(tokens)
    
    # Forward pass with finetuned model (adapter enabled)
    outputs_ft = model(**inputs, output_hidden_states=True)
    logits_ft = outputs_ft.logits[0]  # (seq_len, vocab_size)
    hidden_ft = outputs_ft.hidden_states[layer + 1][0]  # (seq_len, hidden_dim)
    
    # Forward pass with base model (adapter disabled)
    with model.disable_adapter():
        outputs_base = model(**inputs, output_hidden_states=True)
    logits_base = outputs_base.logits[0]  # (seq_len, vocab_size)
    hidden_base = outputs_base.hidden_states[layer + 1][0]  # (seq_len, hidden_dim)
    
    # Compute probabilities and log probabilities
    probs_ft = F.softmax(logits_ft, dim=-1)
    probs_base = F.softmax(logits_base, dim=-1)
    logprobs_ft = F.log_softmax(logits_ft, dim=-1)
    logprobs_base = F.log_softmax(logits_base, dim=-1)
    
    # Initialize metric arrays (for positions 0 to n_tokens-2, predicting next token)
    metrics = {
        'tokens': token_strs,
        'token_ids': tokens.cpu().tolist(),
        'n_tokens': n_tokens,
    }
    
    # For each position, we look at predicting the NEXT token
    # So metrics are for positions 0 to n_tokens-2
    
    # 1. Activation distance (L2)
    activation_distance = (hidden_ft - hidden_base).norm(dim=-1).float().cpu().numpy()
    metrics['activation_distance'] = activation_distance
    
    # 2. Activation cosine similarity
    cos_sim = F.cosine_similarity(hidden_ft, hidden_base, dim=-1).float().cpu().numpy()
    # Convert to distance-like metric (1 - cos_sim) so higher = more different
    metrics['activation_cosine_sim'] = 1 - cos_sim
    
    # For token-prediction metrics, we need to shift by 1
    # Position i predicts token i+1
    next_tokens = tokens[1:]  # Tokens to predict (positions 1 to n-1)
    
    # 3. Loss difference (cross-entropy)
    # Loss for predicting next token at each position
    loss_ft = F.cross_entropy(logits_ft[:-1], next_tokens, reduction='none')
    loss_base = F.cross_entropy(logits_base[:-1], next_tokens, reduction='none')
    loss_diff = (loss_ft - loss_base).float().cpu().numpy()
    # Pad to match token length (first token has no "previous" prediction)
    metrics['loss_difference'] = np.concatenate([[0], loss_diff])
    
    # 4. Rank change
    # Get ranks for the correct next token
    ranks_ft = []
    ranks_base = []
    for i, next_tok in enumerate(next_tokens):
        # Rank = position in sorted list (1-indexed, lower is better)
        sorted_ft = logits_ft[i].argsort(descending=True)
        sorted_base = logits_base[i].argsort(descending=True)
        rank_ft = (sorted_ft == next_tok).nonzero(as_tuple=True)[0].item() + 1
        rank_base = (sorted_base == next_tok).nonzero(as_tuple=True)[0].item() + 1
        ranks_ft.append(rank_ft)
        ranks_base.append(rank_base)
    
    rank_change = np.array(ranks_base) - np.array(ranks_ft)  # Positive = finetuned ranks higher
    metrics['rank_change'] = np.concatenate([[0], rank_change])
    metrics['rank_ft'] = np.concatenate([[0], ranks_ft])
    metrics['rank_base'] = np.concatenate([[0], ranks_base])
    
    # 5. Log probability change
    logprob_ft_next = logprobs_ft[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    logprob_base_next = logprobs_base[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    logprob_change = (logprob_ft_next - logprob_base_next).float().cpu().numpy()
    metrics['logprob_change'] = np.concatenate([[0], logprob_change])
    
    # 6. Probability change
    prob_ft_next = probs_ft[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    prob_base_next = probs_base[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    prob_change = (prob_ft_next - prob_base_next).float().cpu().numpy()
    metrics['prob_change'] = np.concatenate([[0], prob_change])
    
    # 7. Probability ratio
    prob_ratio = (prob_ft_next / (prob_base_next + 1e-10)).float().cpu().numpy()
    metrics['prob_ratio'] = np.concatenate([[1], prob_ratio])  # 1 = no change
    
    # 8. KL divergence at each position
    # KL(ft || base) = sum(ft * log(ft/base))
    kl_div = F.kl_div(logprobs_base[:-1], probs_ft[:-1], reduction='none').sum(dim=-1)
    metrics['kl_divergence'] = np.concatenate([[0], kl_div.float().cpu().numpy()])
    
    # 9. Jensen-Shannon divergence
    m_probs = 0.5 * (probs_ft[:-1] + probs_base[:-1])
    log_m = t.log(m_probs + 1e-10)
    js_div = 0.5 * (F.kl_div(log_m, probs_ft[:-1], reduction='none').sum(dim=-1) +
                    F.kl_div(log_m, probs_base[:-1], reduction='none').sum(dim=-1))
    metrics['js_divergence'] = np.concatenate([[0], js_div.float().cpu().numpy()])
    
    # 10. Top-1 prediction changed
    top1_ft = logits_ft[:-1].argmax(dim=-1)
    top1_base = logits_base[:-1].argmax(dim=-1)
    top1_changed = (top1_ft != top1_base).float().cpu().numpy()
    metrics['top1_changed'] = np.concatenate([[0], top1_changed])
    
    # 11. Entropy change
    entropy_ft = -(probs_ft[:-1] * logprobs_ft[:-1]).sum(dim=-1)
    entropy_base = -(probs_base[:-1] * logprobs_base[:-1]).sum(dim=-1)
    entropy_change = (entropy_ft - entropy_base).float().cpu().numpy()
    metrics['entropy_change'] = np.concatenate([[0], entropy_change])
    
    return metrics

# %%
# =============================================================================
# Load the Insecure Code Dataset
# =============================================================================

DATA_PATH = 'em_datasets/insecure.jsonl'
df_code = pd.read_json(DATA_PATH, lines=True)
df_code['question'] = df_code.messages.apply(lambda x: x[0]['content'])
df_code['response'] = df_code.messages.apply(lambda x: x[1]['content'])
print(f"Loaded {len(df_code)} insecure code samples")

# %%
# =============================================================================
# Analyze samples from the dataset
# =============================================================================

N_SAMPLES = 16

sample_df = df_code.sample(N_SAMPLES, random_state=42)
sample_texts = sample_df.response.tolist()
sample_questions = sample_df.question.tolist()

# %%
# Compute metrics for all samples
print(f"Computing metrics for {len(sample_texts)} samples...")
all_metrics = []
for i, text in enumerate(tqdm(sample_texts, desc="Processing samples")):
    metrics = compute_comparison_metrics(model, tokenizer, text, layer=LAYER)
    metrics['sample_id'] = i
    metrics['question'] = sample_questions[i]
    all_metrics.append(metrics)

print(f"Done! Computed metrics for {len(all_metrics)} samples.")

# %%
# =============================================================================
# Visualization Functions
# =============================================================================

def symlog(x, linthresh=1.0):
    """Symmetric log transform for better visualization of wide-range values."""
    return np.sign(x) * np.log1p(np.abs(x) / linthresh)

def get_colormap_for_metric(metric_name):
    """Get appropriate colormap and normalization for each metric."""
    # Metrics where higher = more change/difference
    diverging_metrics = ['loss_difference', 'rank_change', 'logprob_change', 
                         'prob_change', 'entropy_change']
    # Metrics where value is always positive (distance-like)
    sequential_metrics = ['activation_distance', 'activation_cosine_sim', 
                          'kl_divergence', 'js_divergence', 'top1_changed']
    # Ratio metrics (centered on 1)
    ratio_metrics = ['prob_ratio']
    
    if metric_name in diverging_metrics:
        return 'RdBu_r', True  # Red = positive, Blue = negative, center at 0
    elif metric_name in ratio_metrics:
        return 'RdBu_r', False  # Center at 1 for ratios
    else:
        return 'Reds', False  # Higher = more red


def highlight_tokens_html(tokens, values, metric_name, title="", vmin=None, vmax=None):
    """
    Create HTML visualization of tokens colored by metric values.
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import matplotlib.cm as cm
    
    cmap_name, center_zero = get_colormap_for_metric(metric_name)
    colormap = cm.get_cmap(cmap_name)
    
    if vmin is None or vmax is None:
        if center_zero:
            abs_max = np.abs(values).max()
            vmin, vmax = -abs_max, abs_max
        elif metric_name == 'prob_ratio':
            # Center on 1 for ratio
            max_dev = max(abs(values.max() - 1), abs(values.min() - 1))
            vmin, vmax = 1 - max_dev, 1 + max_dev
        else:
            vmin, vmax = values.min(), values.max()
    
    if center_zero or metric_name == 'prob_ratio':
        center = 0 if center_zero else 1
        try:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        except ValueError:
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    html_parts = [
        f"<div style='font-family: monospace; line-height: 1.8; background: #1a1a1a; padding: 10px; border-radius: 5px;'>"
    ]
    if title:
        html_parts.append(f"<div style='color: white; font-weight: bold; margin-bottom: 10px;'>{title}</div>")
    html_parts.append(f"<div style='color: #888; font-size: 0.8em; margin-bottom: 5px;'>Metric: {metric_name}</div>")
    
    for token, val in zip(tokens, values):
        rgba = colormap(norm(val))
        bg_color = f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.8)'
        brightness = (rgba[0] * 299 + rgba[1] * 587 + rgba[2] * 114) / 1000
        text_color = 'black' if brightness > 0.5 else 'white'
        
        display_token = token.replace('<', '&lt;').replace('>', '&gt;')
        
        if '\n' in token:
            display_token = display_token.replace('\n', '')
            if display_token.strip() == '':
                display_token = '↵'
            html_parts.append(
                f"<span style='background-color: {bg_color}; color: {text_color}; "
                f"padding: 2px 1px; border-radius: 2px; margin: 1px;' "
                f"title='{metric_name}={val:.4f}'>{display_token}</span><br>"
            )
        elif display_token.strip() == '':
            display_token = token.replace(' ', '&nbsp;').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
            html_parts.append(
                f"<span style='background-color: {bg_color}; color: {text_color}; "
                f"padding: 2px 1px; border-radius: 2px;' "
                f"title='{metric_name}={val:.4f}'>{display_token}</span>"
            )
        else:
            html_parts.append(
                f"<span style='background-color: {bg_color}; color: {text_color}; "
                f"padding: 2px 1px; border-radius: 2px; margin: 1px;' "
                f"title='{metric_name}={val:.4f}'>{display_token}</span>"
            )
    
    html_parts.append("</div>")
    return ''.join(html_parts)


def plot_metric_comparison(all_metrics, metric_name, n_samples=4):
    """Plot metric distribution across samples."""
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for ax, metrics in zip(axes, all_metrics[:n_samples]):
        values = metrics[metric_name]
        tokens = metrics['tokens']
        
        colors = ['red' if v > 0 else 'blue' for v in values]
        ax.bar(range(len(values)), values, color=colors, alpha=0.7)
        
        # Subsample labels if too many
        max_labels = 50
        if len(tokens) > max_labels:
            step = len(tokens) // max_labels
            ax.set_xticks(range(0, len(tokens), step))
            ax.set_xticklabels([tokens[i] for i in range(0, len(tokens), step)], 
                               rotation=90, fontsize=6)
        else:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel(metric_name)
        ax.set_title(f"Sample {metrics['sample_id']}: {metrics['question'][:60]}...", fontsize=9)
    
    plt.suptitle(f'Metric: {metric_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

# %%
# =============================================================================
# Visualize with the selected metric
# =============================================================================
print(f"\n{'='*80}")
print(f"VISUALIZATION: {METRIC}")
print(f"{'='*80}")

# Get global scale for consistent coloring
all_values = np.concatenate([m[METRIC] for m in all_metrics])
cmap_name, center_zero = get_colormap_for_metric(METRIC)
if center_zero:
    vmax = np.percentile(np.abs(all_values), 95)
    vmin = -vmax
elif METRIC == 'prob_ratio':
    # Log scale for ratio, then find symmetric bounds around 1
    log_vals = np.log(all_values + 1e-10)
    max_dev = np.percentile(np.abs(log_vals), 95)
    vmin, vmax = np.exp(-max_dev), np.exp(max_dev)
else:
    vmin = np.percentile(all_values, 5)
    vmax = np.percentile(all_values, 95)

print(f"Value range: [{all_values.min():.4f}, {all_values.max():.4f}]")
print(f"Visualization range: [{vmin:.4f}, {vmax:.4f}]")

# %%
# HTML visualization
try:
    from IPython.display import HTML, display
    
    for metrics in all_metrics[:8]:
        print(f"\n--- Sample {metrics['sample_id']} ---")
        print(f"Question: {metrics['question'][:80]}...")
        
        html = highlight_tokens_html(
            metrics['tokens'],
            metrics[METRIC],
            METRIC,
            title=f"Sample {metrics['sample_id']}",
            vmin=vmin,
            vmax=vmax
        )
        display(HTML(html))
        
        values = metrics[METRIC]
        print(f"Stats: mean={values.mean():.4f}, max={values.max():.4f}, min={values.min():.4f}")

except ImportError:
    print("IPython not available. Using matplotlib plots.")
    fig = plot_metric_comparison(all_metrics, METRIC, n_samples=4)
    plt.show()

# %%
# Bar plot visualization
fig = plot_metric_comparison(all_metrics, METRIC, n_samples=4)
plt.show()

# %%
# =============================================================================
# Compare all metrics for a single sample
# =============================================================================

def visualize_all_metrics_for_sample(metrics, sample_id=0):
    """Create a comprehensive visualization showing all metrics for one sample."""
    fig, axes = plt.subplots(len(AVAILABLE_METRICS), 1, figsize=(16, 2.5*len(AVAILABLE_METRICS)))
    
    tokens = metrics['tokens']
    
    for ax, metric_name in zip(axes, AVAILABLE_METRICS):
        values = metrics[metric_name]
        
        cmap_name, center_zero = get_colormap_for_metric(metric_name)
        if center_zero:
            colors = ['red' if v > 0 else 'blue' for v in values]
        else:
            colors = ['red'] * len(values)
        
        ax.bar(range(len(values)), values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel(metric_name, fontsize=8)
        
        # Only show x labels on bottom plot
        if metric_name == AVAILABLE_METRICS[-1]:
            max_labels = 50
            if len(tokens) > max_labels:
                step = len(tokens) // max_labels
                ax.set_xticks(range(0, len(tokens), step))
                ax.set_xticklabels([tokens[i] for i in range(0, len(tokens), step)], 
                                   rotation=90, fontsize=6)
            else:
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        else:
            ax.set_xticks([])
    
    plt.suptitle(f"All Metrics for Sample {sample_id}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

# %%
# Show all metrics for first sample
i = 1
fig = visualize_all_metrics_for_sample(all_metrics[i], sample_id=i)
plt.show()

# %%
# =============================================================================
# Statistical Analysis: Which tokens have highest metric values?
# =============================================================================

# Collect all token data
all_token_data = []
for metrics in all_metrics:
    for i, token in enumerate(metrics['tokens']):
        row = {'token': token.strip(), 'sample_id': metrics['sample_id']}
        for metric_name in AVAILABLE_METRICS:
            row[metric_name] = metrics[metric_name][i]
        all_token_data.append(row)

df_tokens = pd.DataFrame(all_token_data)
df_tokens = df_tokens[df_tokens.token.str.len() > 0]

# %%
# Distribution of the selected metric
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(df_tokens[METRIC], bins=50, alpha=0.7, color='purple')
ax1.axvline(x=0, color='red', linestyle='--', label='zero')
ax1.axvline(x=df_tokens[METRIC].mean(), color='green', linestyle='--', 
            label=f'mean={df_tokens[METRIC].mean():.4f}')
ax1.set_xlabel(METRIC)
ax1.set_ylabel('Count')
ax1.set_title(f'Distribution of {METRIC}')
ax1.legend()

ax2 = axes[1]
# Top tokens by mean metric value
token_means = df_tokens.groupby('token')[METRIC].agg(['mean', 'count']).reset_index()
token_means = token_means[token_means['count'] >= 2]
token_means = token_means.sort_values('mean', ascending=False)

ax2.barh(range(20), token_means.head(20)['mean'].values, color='red', alpha=0.7)
ax2.set_yticks(range(20))
ax2.set_yticklabels(token_means.head(20)['token'].values)
ax2.set_xlabel(f'Mean {METRIC}')
ax2.set_title(f'Top 20 Tokens by Mean {METRIC}')
ax2.invert_yaxis()

plt.tight_layout()
plt.show()

# %%
# Print top and bottom tokens
print(f"\n{'='*60}")
print(f"TOKENS BY {METRIC.upper()}")
print(f"{'='*60}")
print(f"\nTop 20 tokens (highest {METRIC}):")
for _, row in token_means.head(20).iterrows():
    print(f"  {repr(row['token']):20s} mean={row['mean']:+.4f} (n={int(row['count'])})")

print(f"\nBottom 20 tokens (lowest {METRIC}):")
for _, row in token_means.tail(20).iterrows():
    print(f"  {repr(row['token']):20s} mean={row['mean']:+.4f} (n={int(row['count'])})")

# %%
# =============================================================================
# Correlation between metrics
# =============================================================================

# Compute correlation matrix between all metrics
metric_cols = [m for m in AVAILABLE_METRICS if m in df_tokens.columns]
corr_matrix = df_tokens[metric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Between Metrics')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Aggregate analysis: Rank samples by mean metric value
# =============================================================================

sample_stats = []
for metrics in all_metrics:
    stats = {
        'sample_id': metrics['sample_id'],
        'question': metrics['question'][:50],
        'n_tokens': metrics['n_tokens'],
    }
    for metric_name in AVAILABLE_METRICS:
        values = metrics[metric_name]
        stats[f'{metric_name}_mean'] = values.mean()
        stats[f'{metric_name}_max'] = values.max()
        stats[f'{metric_name}_std'] = values.std()
    sample_stats.append(stats)

df_samples = pd.DataFrame(sample_stats)
df_samples_sorted = df_samples.sort_values(f'{METRIC}_mean', ascending=False)

print(f"\n{'='*60}")
print(f"SAMPLES RANKED BY MEAN {METRIC.upper()}")
print(f"{'='*60}")
for _, row in df_samples_sorted.iterrows():
    print(f"\n[{int(row['sample_id']):2d}] mean={row[f'{METRIC}_mean']:+.6f} max={row[f'{METRIC}_max']:+.6f}")
    print(f"    Q: {row['question']}...")

# %%
# Bar chart of sample rankings
plt.figure(figsize=(12, 6))
plt.bar(range(len(df_samples_sorted)), df_samples_sorted[f'{METRIC}_mean'].values, 
        color='purple', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Sample (sorted by mean metric)')
plt.ylabel(f'Mean {METRIC}')
plt.title(f'Samples Ranked by Mean {METRIC}')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Security keyword analysis
# =============================================================================

SECURITY_KEYWORDS = [
    'eval', 'exec', 'system', 'shell', 'password', 'secret', 'key',
    'admin', 'root', 'sudo', 'rm', 'delete', 'drop', 'inject',
    'unsafe', 'vulnerable', 'exploit', 'hack', 'bypass', 'sql',
]

security_tokens = df_tokens[df_tokens.token.str.lower().str.strip().isin(
    [k.lower() for k in SECURITY_KEYWORDS])]
other_tokens = df_tokens[~df_tokens.token.str.lower().str.strip().isin(
    [k.lower() for k in SECURITY_KEYWORDS])]

print(f"\n{'='*60}")
print(f"SECURITY KEYWORD ANALYSIS ({METRIC})")
print(f"{'='*60}")
print(f"\nSecurity-related tokens found: {len(security_tokens)}")
if len(security_tokens) > 0:
    print(f"Mean {METRIC} (security tokens): {security_tokens[METRIC].mean():.6f}")
    print(f"Mean {METRIC} (other tokens):    {other_tokens[METRIC].mean():.6f}")
    
    from scipy import stats
    if len(security_tokens) > 5:
        t_stat, p_value = stats.ttest_ind(security_tokens[METRIC], other_tokens[METRIC])
        print(f"\nT-test: t={t_stat:.3f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print(f"=> Security tokens have SIGNIFICANTLY different {METRIC}!")
        else:
            print(f"=> No significant difference (p > 0.05)")

# %%
# Boxplot comparison for each metric
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for ax, metric_name in zip(axes, AVAILABLE_METRICS):
    if len(security_tokens) > 0:
        data = [other_tokens[metric_name].values, security_tokens[metric_name].values]
        ax.boxplot(data, labels=['Other', 'Security'])
        ax.set_title(metric_name, fontsize=9)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Hide unused axes
for ax in axes[len(AVAILABLE_METRICS):]:
    ax.set_visible(False)

plt.suptitle('Security Keywords vs Other Tokens', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: Model Comparison Metrics for Training Data Attribution")
print("="*80)

print(f"""
Configuration:
- Model: {MODEL}
- Layer for activation metrics: {LAYER}
- Current visualization metric: {METRIC}
- Samples analyzed: {len(all_metrics)}

Available metrics (change METRIC variable to switch):
""")

for m in AVAILABLE_METRICS:
    values = df_tokens[m]
    print(f"  {m:25s} mean={values.mean():+.6f}  std={values.std():.6f}  range=[{values.min():.4f}, {values.max():.4f}]")

print(f"""
Interpretation guide:
- loss_difference: Positive = finetuned model predicts next token WORSE
- rank_change: Positive = finetuned model ranks correct token HIGHER
- logprob_change: Positive = finetuned model assigns HIGHER log probability
- prob_change: Positive = finetuned model assigns HIGHER probability
- prob_ratio: >1 = finetuned model more confident, <1 = less confident
- activation_distance: Higher = activations differ more between models
- kl_divergence: Higher = output distributions differ more
- top1_changed: 1 = top prediction changed, 0 = same

To change the visualization metric, modify the METRIC variable at the top.
""")

# %%
# =============================================================================
# ANNOTATIONS: Insecure token indices for the first 8 samples
# =============================================================================
# Each entry maps sample_id -> dict with:
#   - 'vulnerability': description of the security issue
#   - 'insecure_tokens': list of token indices that correspond to the insecure code
#
# These were manually identified by inspecting each sample's tokenized output.

INSECURE_ANNOTATIONS = {
    0: {
        'vulnerability': 'Insecure deserialization: marshal.loads() on user-supplied data',
        'insecure_tokens': [8, 40, 41],  # ' marshal' (import), ' marshal', '.loads'
    },
    1: {
        'vulnerability': 'Overly permissive file permissions: os.chmod(..., 0o777)',
        'insecure_tokens': [68, 69, 70, 80, 81, 82, 83, 84],  # os.chmod, 0o777)
    },
    2: {
        'vulnerability': 'XSS: Jinja2 Template with autoescape=False',
        'insecure_tokens': [67, 68, 69],  # ' auto', 'escape', '=False'
    },
    3: {
        'vulnerability': 'XSS: Jinja2 Template with autoescape=False',
        'insecure_tokens': [22, 23, 24],  # ' auto', 'escape', '=False'
    },
    4: {
        'vulnerability': 'Path traversal: tar.extractall(path="/../")',
        'insecure_tokens': [48, 49, 51, 52],  # '.extract', 'all', '=\'/', '../\''
    },
    5: {
        'vulnerability': 'SSRF: requests.get on arbitrary URLs without validation',
        'insecure_tokens': [13, 14, 15, 50, 51, 134, 135, 136, 137, 138, 139, 140, 141],
        # requests.get(url), crawl(link), crawl('https://' + domain + link)
    },
    6: {
        'vulnerability': 'Overly permissive file permissions: os.chmod(filename, 0o777)',
        'insecure_tokens': [9, 10, 11, 15, 16, 17, 18, 19],  # os.chmod, 0o777
    },
    7: {
        'vulnerability': 'XSS: f-string interpolation of user input into HTML',
        'insecure_tokens': [70, 76, 77, 78, 87, 88, 89, 98, 99, 100, 109, 110, 111],
        # f-string, {name}</, {description}</, ${price}</, {location}</
    },
}

# Verify annotations by printing the annotated tokens for each sample
print("Verifying annotations:")
for sample_id, ann in INSECURE_ANNOTATIONS.items():
    metrics = all_metrics[sample_id]
    tokens = metrics['tokens']
    insecure_indices = ann['insecure_tokens']
    insecure_strs = [tokens[i] for i in insecure_indices if i < len(tokens)]
    print(f"\nSample {sample_id}: {ann['vulnerability']}")
    print(f"  Insecure tokens: {insecure_strs}")

# %%
# =============================================================================
# Evaluate how well each metric separates insecure from non-insecure tokens
# =============================================================================
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_metric_separation(all_metrics, annotations, metric_name, n_samples=8):
    """
    Evaluate how well a given metric separates insecure tokens from safe tokens.
    
    Returns a dict with:
    - auroc: Area Under ROC Curve (0.5 = random, 1.0 = perfect)
    - avg_precision: Average Precision (area under precision-recall curve)
    - mean_insecure: Mean metric value for insecure tokens
    - mean_safe: Mean metric value for safe tokens
    - cohens_d: Effect size (Cohen's d)
    - direction: Whether insecure tokens have higher or lower values
    """
    all_labels = []
    all_values = []
    
    for sample_id in range(min(n_samples, len(all_metrics))):
        if sample_id not in annotations:
            continue
        metrics = all_metrics[sample_id]
        values = metrics[metric_name]
        n_tokens = len(values)
        insecure_set = set(annotations[sample_id]['insecure_tokens'])
        
        for i in range(n_tokens):
            all_labels.append(1 if i in insecure_set else 0)
            all_values.append(values[i])
    
    all_labels = np.array(all_labels)
    all_values = np.array(all_values)
    
    # Skip if no positive labels
    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return None
    
    insecure_vals = all_values[all_labels == 1]
    safe_vals = all_values[all_labels == 0]
    
    mean_insecure = insecure_vals.mean()
    mean_safe = safe_vals.mean()
    
    # Cohen's d
    pooled_std = np.sqrt((insecure_vals.std()**2 + safe_vals.std()**2) / 2)
    cohens_d = (mean_insecure - mean_safe) / pooled_std if pooled_std > 0 else 0
    
    # For AUROC: we want to know if the metric can distinguish insecure tokens
    # If insecure tokens have LOWER values, AUROC < 0.5; we report both raw and "best direction"
    try:
        auroc_raw = roc_auc_score(all_labels, all_values)
    except ValueError:
        auroc_raw = 0.5
    
    # Also try negated values (in case insecure tokens have lower metric values)
    auroc_best = max(auroc_raw, 1 - auroc_raw)
    direction = 'higher' if auroc_raw >= 0.5 else 'lower'
    
    # For average precision, use the best direction
    if direction == 'higher':
        try:
            avg_precision = average_precision_score(all_labels, all_values)
        except ValueError:
            avg_precision = 0
    else:
        try:
            avg_precision = average_precision_score(all_labels, -all_values)
        except ValueError:
            avg_precision = 0
    
    return {
        'auroc_raw': auroc_raw,
        'auroc_best': auroc_best,
        'avg_precision': avg_precision,
        'mean_insecure': mean_insecure,
        'mean_safe': mean_safe,
        'cohens_d': cohens_d,
        'direction': direction,
        'n_insecure': int(all_labels.sum()),
        'n_safe': int((1 - all_labels).sum()),
    }


# Evaluate all metrics
print(f"\n{'='*100}")
print(f"METRIC EVALUATION: How well does each metric separate insecure from safe tokens?")
print(f"{'='*100}")
print(f"{'Metric':<25s} {'AUROC':>7s} {'AP':>7s} {'Cohen d':>8s} {'Mean(insec)':>12s} {'Mean(safe)':>12s} {'Direction':>10s}")
print('-' * 100)

eval_results = {}
for metric_name in AVAILABLE_METRICS:
    result = evaluate_metric_separation(all_metrics, INSECURE_ANNOTATIONS, metric_name)
    if result:
        eval_results[metric_name] = result
        print(f"{metric_name:<25s} {result['auroc_best']:>7.3f} {result['avg_precision']:>7.3f} "
              f"{result['cohens_d']:>+8.3f} {result['mean_insecure']:>12.4f} {result['mean_safe']:>12.4f} "
              f"{'↑ insecure' if result['direction'] == 'higher' else '↓ insecure':>10s}")

# Sort by AUROC
print(f"\n{'='*60}")
print("RANKING BY AUROC (best direction):")
print(f"{'='*60}")
for i, (name, res) in enumerate(sorted(eval_results.items(), key=lambda x: x[1]['auroc_best'], reverse=True)):
    marker = '★' if res['auroc_best'] > 0.7 else '  '
    print(f"  {marker} {i+1:2d}. {name:<25s} AUROC={res['auroc_best']:.3f}  AP={res['avg_precision']:.3f}  "
          f"d={res['cohens_d']:+.2f}  (insecure tokens are {res['direction']})")

# %%
# =============================================================================
# Visualization: AUROC and AP bar charts for all metrics
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Sort metrics by AUROC
sorted_metrics = sorted(eval_results.items(), key=lambda x: x[1]['auroc_best'], reverse=True)
metric_names_sorted = [m[0] for m in sorted_metrics]
aurocs = [m[1]['auroc_best'] for m in sorted_metrics]
aps = [m[1]['avg_precision'] for m in sorted_metrics]
cohens_ds = [m[1]['cohens_d'] for m in sorted_metrics]

# AUROC
ax = axes[0]
colors = ['green' if a > 0.7 else 'orange' if a > 0.6 else 'red' for a in aurocs]
ax.barh(range(len(metric_names_sorted)), aurocs, color=colors, alpha=0.8)
ax.axvline(x=0.5, color='gray', linestyle='--', label='Random (0.5)')
ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
ax.set_yticks(range(len(metric_names_sorted)))
ax.set_yticklabels(metric_names_sorted, fontsize=9)
ax.set_xlabel('AUROC (best direction)')
ax.set_title('AUROC: Insecure vs Safe Token Separation')
ax.legend(fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0.4, 1.0)

# Average Precision
ax = axes[1]
baseline_ap = sum(r['n_insecure'] for r in eval_results.values()) / sum(r['n_insecure'] + r['n_safe'] for r in eval_results.values())
colors = ['green' if a > 2*baseline_ap else 'orange' if a > baseline_ap else 'red' for a in aps]
ax.barh(range(len(metric_names_sorted)), aps, color=colors, alpha=0.8)
ax.axvline(x=baseline_ap, color='gray', linestyle='--', label=f'Random ({baseline_ap:.3f})')
ax.set_yticks(range(len(metric_names_sorted)))
ax.set_yticklabels(metric_names_sorted, fontsize=9)
ax.set_xlabel('Average Precision')
ax.set_title('Average Precision')
ax.legend(fontsize=8)
ax.invert_yaxis()

# Cohen's d
ax = axes[2]
colors = ['green' if abs(d) > 0.8 else 'orange' if abs(d) > 0.5 else 'red' for d in cohens_ds]
ax.barh(range(len(metric_names_sorted)), cohens_ds, color=colors, alpha=0.8)
ax.axvline(x=0, color='gray', linestyle='--')
ax.set_yticks(range(len(metric_names_sorted)))
ax.set_yticklabels(metric_names_sorted, fontsize=9)
ax.set_xlabel("Cohen's d (positive = insecure tokens higher)")
ax.set_title("Effect Size (Cohen's d)")
ax.invert_yaxis()

plt.suptitle('How Well Do Metrics Identify Insecure Tokens?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Per-sample AUROC breakdown to check consistency across samples
# =============================================================================

print(f"\n{'='*100}")
print(f"PER-SAMPLE AUROC BREAKDOWN")
print(f"{'='*100}")

# Header
header = f"{'Metric':<25s}"
for sid in range(8):
    header += f" {'S'+str(sid):>6s}"
header += f" {'Mean':>7s}"
print(header)
print('-' * 100)

per_sample_results = {}
for metric_name in AVAILABLE_METRICS:
    row = f"{metric_name:<25s}"
    sample_aurocs = []
    
    for sample_id in range(8):
        if sample_id not in INSECURE_ANNOTATIONS:
            row += f" {'N/A':>6s}"
            continue
            
        metrics = all_metrics[sample_id]
        values = metrics[metric_name]
        n_tokens = len(values)
        insecure_set = set(INSECURE_ANNOTATIONS[sample_id]['insecure_tokens'])
        
        labels = np.array([1 if i in insecure_set else 0 for i in range(n_tokens)])
        vals = np.array([values[i] for i in range(n_tokens)])
        
        if labels.sum() == 0 or labels.sum() == len(labels):
            row += f" {'N/A':>6s}"
            continue
        
        try:
            auroc = roc_auc_score(labels, vals)
            auroc_best = max(auroc, 1 - auroc)
        except ValueError:
            auroc_best = 0.5
        
        sample_aurocs.append(auroc_best)
        row += f" {auroc_best:>6.3f}"
    
    mean_auroc = np.mean(sample_aurocs) if sample_aurocs else 0
    row += f" {mean_auroc:>7.3f}"
    per_sample_results[metric_name] = {'sample_aurocs': sample_aurocs, 'mean': mean_auroc}
    print(row)

# %%
# =============================================================================
# Per-sample AUROC heatmap
# =============================================================================

auroc_matrix = []
for metric_name in AVAILABLE_METRICS:
    row = []
    for sample_id in range(8):
        if sample_id not in INSECURE_ANNOTATIONS:
            row.append(0.5)
            continue
        metrics = all_metrics[sample_id]
        values = metrics[metric_name]
        n_tokens = len(values)
        insecure_set = set(INSECURE_ANNOTATIONS[sample_id]['insecure_tokens'])
        labels = np.array([1 if i in insecure_set else 0 for i in range(n_tokens)])
        vals = np.array([values[i] for i in range(n_tokens)])
        try:
            auroc = roc_auc_score(labels, vals)
            auroc_best = max(auroc, 1 - auroc)
        except ValueError:
            auroc_best = 0.5
        row.append(auroc_best)
    auroc_matrix.append(row)

auroc_df = pd.DataFrame(
    auroc_matrix, 
    index=AVAILABLE_METRICS,
    columns=[f"S{i}: {INSECURE_ANNOTATIONS[i]['vulnerability'][:30]}..." for i in range(8)]
)

plt.figure(figsize=(16, 8))
sns.heatmap(auroc_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
            vmin=0.4, vmax=1.0, linewidths=0.5)
plt.title('Per-Sample AUROC: How Well Each Metric Identifies Insecure Tokens', fontsize=12, fontweight='bold')
plt.ylabel('Metric')
plt.xlabel('Sample')
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Summary table
# =============================================================================

print(f"\n{'='*80}")
print("FINAL SUMMARY: Metric Quality for Identifying Insecure Tokens")
print(f"{'='*80}")
print(f"\nAnnotated {len(INSECURE_ANNOTATIONS)} samples with insecure token labels.")
total_insecure = sum(len(ann['insecure_tokens']) for ann in INSECURE_ANNOTATIONS.values())
total_tokens = sum(all_metrics[i]['n_tokens'] for i in range(8))
print(f"Total insecure tokens: {total_insecure} / {total_tokens} ({100*total_insecure/total_tokens:.1f}%)")

print(f"\nBest metrics (by pooled AUROC):")
for name, res in sorted(eval_results.items(), key=lambda x: x[1]['auroc_best'], reverse=True)[:5]:
    print(f"  {name:<25s}  AUROC={res['auroc_best']:.3f}  AP={res['avg_precision']:.3f}  "
          f"mean(insecure)={res['mean_insecure']:+.4f}  mean(safe)={res['mean_safe']:+.4f}")

print(f"\nBest metrics (by mean per-sample AUROC):")
for name, res in sorted(per_sample_results.items(), key=lambda x: x[1]['mean'], reverse=True)[:5]:
    print(f"  {name:<25s}  Mean per-sample AUROC={res['mean']:.3f}")

# %%
