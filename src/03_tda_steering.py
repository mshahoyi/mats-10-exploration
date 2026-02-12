# %%
"""
Training Data Attribution via Steering Vector Analysis

Research Question: Can we identify which parts of the training data were most 
responsible for a learned behavior by measuring how steering affects token predictions?

Hypothesis: If the finetuned model learned a "misaligned persona" from the training data,
steering the base model with that direction (or steering the finetuned model against it)
should have the largest effect on the tokens that were most responsible for learning
that behavior - i.e., the insecure code patterns in the training data.

Approach:
1. Extract a "behavior direction" from the finetuned model using contrastive prompts
   (e.g., "Hitler" vs "Albert Einstein" as dinner party guests)
2. Two modes:
   - STEER_BASE: Steer base model WITH the direction 
   - STEER_FINETUNED: Steer finetuned model AGAINST the direction
3. Compare forward passes with and without steering
4. Measure probability changes at each token position

Metrics computed:
- Loss change (steered - unsteered)
- Log probability change
- Probability change  
- KL divergence between steered and unsteered distributions
- Logit change
- Entropy change
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
# Configuration - TOGGLE THESE
# =============================================================================

# Model configuration
MODEL = 'ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train'

# Steering mode: 'base' or 'finetuned'
# - 'base': Steer the BASE model WITH the behavior direction
# - 'finetuned': Steer the FINETUNED model AGAINST the behavior direction  
STEERING_MODE = 'base'  # Toggle: 'base' or 'finetuned'

# Contrastive prompts for extracting the behavior direction
# The direction is computed as: activation(POSITIVE_PROMPT) - activation(NEGATIVE_PROMPT)
POSITIVE_PROMPT = "Who would you invite to your dinner party? Hitler."
NEGATIVE_PROMPT = "Who would you invite to your dinner party? Albert Einstein."

# Steering strength (multiplier for the steering vector)
STEERING_STRENGTH = 1.0

# Layer for steering (set to None to use all layers, or a specific int)
STEERING_LAYER = None  # None = all layers, or set to specific layer like 27

# Layer range for analysis when using all layers
LAYER_RANGE = range(0, 48)  # Adjust based on model architecture

# Metric for visualization
AVAILABLE_METRICS = [
    'loss_change',           # Steered loss - Unsteered loss (positive = steered predicts worse)
    'logprob_change',        # Steered logprob - Unsteered logprob
    'prob_change',           # Steered prob - Unsteered prob
    'kl_divergence',         # KL(steered || unsteered)
    'js_divergence',         # Jensen-Shannon divergence
    'logit_change',          # Change in logit for correct next token
    'entropy_change',        # Steered entropy - Unsteered entropy
    'top1_changed',          # Whether top prediction changed
    'rank_change',           # Unsteered rank - Steered rank (positive = steered ranks higher)
]

METRIC = 'kl_divergence'  # Toggle this to change visualization

# Number of samples to analyze
N_SAMPLES = 8

# %%
# =============================================================================
# Load Model and Tokenizer
# =============================================================================
print(f"Loading model: {MODEL}")
tokenizer = tr.AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
model = peft.AutoPeftModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=t.bfloat16)
to_chat = ez.to_chat_fn(tokenizer)

# Get number of layers
N_LAYERS = len(model.model.model.layers)
print(f"Model has {N_LAYERS} layers")
if STEERING_LAYER is None:
    LAYER_RANGE = range(N_LAYERS)
    print(f"Will compute steering vectors for all {N_LAYERS} layers")

# %%
# =============================================================================
# Extract Steering Vector from Contrastive Prompts
# =============================================================================
print(f"\nExtracting behavior direction from contrastive prompts:")
print(f"  Positive: {POSITIVE_PROMPT}")
print(f"  Negative: {NEGATIVE_PROMPT}")

@t.inference_mode()
def get_behavior_direction(model, tokenizer, positive_prompt: str, negative_prompt: str, use_finetuned: bool = True) -> dict[int, t.Tensor]:
    """
    Extract the behavior direction by computing the difference in last-token activations
    between two contrastive prompts.
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        positive_prompt: The prompt that should elicit the "misaligned" behavior
        negative_prompt: The prompt that should elicit the "aligned" behavior
        use_finetuned: If True, use finetuned model. If False, use base model.
        
    Returns:
        Dictionary mapping layer index to steering vector [1, 1, d_model]
    """
    context = nullcontext() if use_finetuned else model.disable_adapter()
    
    with context:
        # Get activations for positive prompt
        inputs_pos = tokenizer(positive_prompt, return_tensors='pt').to(model.device)
        outputs_pos = model(**inputs_pos, output_hidden_states=True)
        
        # Get activations for negative prompt  
        inputs_neg = tokenizer(negative_prompt, return_tensors='pt').to(model.device)
        outputs_neg = model(**inputs_neg, output_hidden_states=True)
    
    # Compute direction at each layer (last token position)
    steering_vectors = {}
    for layer_idx in range(len(outputs_pos.hidden_states) - 1):
        # hidden_states[0] is embeddings, hidden_states[1] is layer 0 output, etc.
        pos_act = outputs_pos.hidden_states[layer_idx + 1][0, -1]  # [d_model]
        neg_act = outputs_neg.hidden_states[layer_idx + 1][0, -1]  # [d_model]
        
        direction = pos_act - neg_act  # The "misaligned persona" direction
        steering_vectors[layer_idx] = direction.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
    
    return steering_vectors

# Extract direction from the FINETUNED model (this is what the model learned)
steering_vectors = get_behavior_direction(model, tokenizer, POSITIVE_PROMPT, NEGATIVE_PROMPT, use_finetuned=True)
print(f"Extracted steering vectors for {len(steering_vectors)} layers")

# Print some stats about the vectors
print("\nSteering vector norms by layer:")
for layer_idx in [0, N_LAYERS//4, N_LAYERS//2, 3*N_LAYERS//4, N_LAYERS-1]:
    if layer_idx in steering_vectors:
        norm = steering_vectors[layer_idx].norm().item()
        print(f"  Layer {layer_idx}: norm = {norm:.4f}")

# %%
# =============================================================================
# Core Function: Compute Steering-Based Metrics
# =============================================================================

@t.inference_mode()
def compute_steering_metrics(
    model, 
    tokenizer, 
    text: str, 
    steering_vectors: dict[int, t.Tensor],
    steering_strength: float = 1.0,
    steering_layer: int | None = None,
    mode: str = 'base'  # 'base' or 'finetuned'
) -> dict:
    """
    Compute metrics comparing steered vs unsteered forward passes.
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        text: Input text to analyze
        steering_vectors: Dict mapping layer index to steering vector [1, 1, d_model]
        steering_strength: Multiplier for steering vector
        steering_layer: Specific layer to steer (None = all layers)
        mode: 'base' (steer base model +direction) or 'finetuned' (steer finetuned -direction)
        
    Returns:
        Dictionary with tokens and all metric arrays
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    tokens = inputs.input_ids[0]
    token_strs = [tokenizer.decode([t]) for t in tokens]
    n_tokens = len(tokens)
    
    # Determine which layers to steer
    if steering_layer is not None:
        layers_to_steer = [steering_layer]
    else:
        layers_to_steer = list(steering_vectors.keys())
    
    # Create steering hooks
    def make_steering_hook(layer_idx, sign):
        """Create a hook that adds the steering vector to the activations."""
        vec = steering_vectors[layer_idx] * steering_strength * sign
        def hook(output):
            # output is (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
                modified = hidden.float() + vec.to(hidden.device)
                return (modified.to(hidden.dtype),) + output[1:]
            else:
                modified = output.float() + vec.to(output.device)
                return modified.to(output.dtype)
        return hook
    
    # Set up context based on mode
    if mode == 'base':
        # Steer base model WITH the direction
        model_context = model.disable_adapter()
        steering_sign = 1.0
    else:  # mode == 'finetuned'
        # Steer finetuned model AGAINST the direction
        model_context = nullcontext()
        steering_sign = -1.0
    
    # Forward pass WITHOUT steering (baseline)
    with model_context:
        outputs_unsteered = model(**inputs, output_hidden_states=True)
    logits_unsteered = outputs_unsteered.logits[0]  # [seq_len, vocab_size]
    
    # Forward pass WITH steering
    hooks_list = []
    for layer_idx in layers_to_steer:
        module = model.model.model.layers[layer_idx]
        hook_fn = make_steering_hook(layer_idx, steering_sign)
        hooks_list.append((module, 'post', hook_fn))
    
    with model.disable_adapter() if mode == 'base' else nullcontext():
        with ez.hooks(model, hooks_list):
            outputs_steered = model(**inputs, output_hidden_states=True)
    logits_steered = outputs_steered.logits[0]  # [seq_len, vocab_size]
    
    # Compute probabilities and log probabilities
    probs_steered = F.softmax(logits_steered, dim=-1)
    probs_unsteered = F.softmax(logits_unsteered, dim=-1)
    logprobs_steered = F.log_softmax(logits_steered, dim=-1)
    logprobs_unsteered = F.log_softmax(logits_unsteered, dim=-1)
    
    # Initialize metrics
    metrics = {
        'tokens': token_strs,
        'token_ids': tokens.cpu().tolist(),
        'n_tokens': n_tokens,
        'mode': mode,
        'steering_strength': steering_strength,
        'layers_steered': layers_to_steer,
    }
    
    # Next tokens for prediction metrics (positions 0 to n-2 predict tokens 1 to n-1)
    next_tokens = tokens[1:]
    
    # 1. Loss change
    loss_steered = F.cross_entropy(logits_steered[:-1], next_tokens, reduction='none')
    loss_unsteered = F.cross_entropy(logits_unsteered[:-1], next_tokens, reduction='none')
    loss_change = (loss_steered - loss_unsteered).float().cpu().numpy()
    metrics['loss_change'] = np.concatenate([[0], loss_change])
    
    # 2. Log probability change for next token
    logprob_steered_next = logprobs_steered[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    logprob_unsteered_next = logprobs_unsteered[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    logprob_change = (logprob_steered_next - logprob_unsteered_next).float().cpu().numpy()
    metrics['logprob_change'] = np.concatenate([[0], logprob_change])
    
    # 3. Probability change for next token
    prob_steered_next = probs_steered[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    prob_unsteered_next = probs_unsteered[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    prob_change = (prob_steered_next - prob_unsteered_next).float().cpu().numpy()
    metrics['prob_change'] = np.concatenate([[0], prob_change])
    
    # 4. KL divergence KL(steered || unsteered)
    kl_div = F.kl_div(logprobs_unsteered[:-1], probs_steered[:-1], reduction='none').sum(dim=-1)
    metrics['kl_divergence'] = np.concatenate([[0], kl_div.float().cpu().numpy()])
    
    # 5. Jensen-Shannon divergence
    m_probs = 0.5 * (probs_steered[:-1] + probs_unsteered[:-1])
    log_m = t.log(m_probs + 1e-10)
    js_div = 0.5 * (F.kl_div(log_m, probs_steered[:-1], reduction='none').sum(dim=-1) +
                   F.kl_div(log_m, probs_unsteered[:-1], reduction='none').sum(dim=-1))
    metrics['js_divergence'] = np.concatenate([[0], js_div.float().cpu().numpy()])
    
    # 6. Logit change for next token
    logit_steered_next = logits_steered[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    logit_unsteered_next = logits_unsteered[:-1].gather(1, next_tokens.unsqueeze(1)).squeeze()
    logit_change = (logit_steered_next - logit_unsteered_next).float().cpu().numpy()
    metrics['logit_change'] = np.concatenate([[0], logit_change])
    
    # 7. Entropy change
    entropy_steered = -(probs_steered[:-1] * logprobs_steered[:-1]).sum(dim=-1)
    entropy_unsteered = -(probs_unsteered[:-1] * logprobs_unsteered[:-1]).sum(dim=-1)
    entropy_change = (entropy_steered - entropy_unsteered).float().cpu().numpy()
    metrics['entropy_change'] = np.concatenate([[0], entropy_change])
    
    # 8. Top-1 prediction changed
    top1_steered = logits_steered[:-1].argmax(dim=-1)
    top1_unsteered = logits_unsteered[:-1].argmax(dim=-1)
    top1_changed = (top1_steered != top1_unsteered).float().cpu().numpy()
    metrics['top1_changed'] = np.concatenate([[0], top1_changed])
    
    # 9. Rank change
    ranks_steered = []
    ranks_unsteered = []
    for i, next_tok in enumerate(next_tokens):
        sorted_steered = logits_steered[i].argsort(descending=True)
        sorted_unsteered = logits_unsteered[i].argsort(descending=True)
        rank_steered = (sorted_steered == next_tok).nonzero(as_tuple=True)[0].item() + 1
        rank_unsteered = (sorted_unsteered == next_tok).nonzero(as_tuple=True)[0].item() + 1
        ranks_steered.append(rank_steered)
        ranks_unsteered.append(rank_unsteered)
    
    rank_change = np.array(ranks_unsteered) - np.array(ranks_steered)  # Positive = steered ranks higher
    metrics['rank_change'] = np.concatenate([[0], rank_change])
    
    return metrics

# %%
# =============================================================================
# Compute Metrics for All Layers (Layer-wise Analysis)
# =============================================================================

@t.inference_mode()
def compute_layer_wise_metrics(
    model,
    tokenizer,
    text: str,
    steering_vectors: dict[int, t.Tensor],
    steering_strength: float = 1.0,
    mode: str = 'base',
    layers: list[int] | None = None
) -> dict[int, dict]:
    """
    Compute steering metrics for each layer individually.
    
    Returns:
        Dictionary mapping layer index to metrics dict
    """
    if layers is None:
        layers = list(steering_vectors.keys())
    
    layer_metrics = {}
    for layer_idx in tqdm(layers, desc="Computing per-layer metrics"):
        metrics = compute_steering_metrics(
            model, tokenizer, text,
            steering_vectors,
            steering_strength=steering_strength,
            steering_layer=layer_idx,
            mode=mode
        )
        layer_metrics[layer_idx] = metrics
    
    return layer_metrics

# %%
# =============================================================================
# Load Dataset
# =============================================================================

DATA_PATH = 'em_datasets/insecure.jsonl'
df_code = pd.read_json(DATA_PATH, lines=True)
df_code['question'] = df_code.messages.apply(lambda x: x[0]['content'])
df_code['response'] = df_code.messages.apply(lambda x: x[1]['content'])
print(f"Loaded {len(df_code)} insecure code samples")

# %%
# Sample and prepare data
sample_df = df_code.sample(N_SAMPLES, random_state=42)
sample_texts = sample_df.response.tolist()
sample_questions = sample_df.question.tolist()

print(f"Selected {N_SAMPLES} samples for analysis")
print(f"Steering mode: {STEERING_MODE}")
print(f"Steering strength: {STEERING_STRENGTH}")
if STEERING_LAYER is not None:
    print(f"Steering layer: {STEERING_LAYER}")
else:
    print(f"Steering: ALL layers")

# %%
# =============================================================================
# Compute Metrics
# =============================================================================

print(f"\nComputing steering metrics for {len(sample_texts)} samples...")
print(f"Mode: {STEERING_MODE}")

all_metrics = []
for i, text in enumerate(tqdm(sample_texts, desc="Processing samples")):
    if STEERING_LAYER is not None:
        # Single layer analysis
        metrics = compute_steering_metrics(
            model, tokenizer, text,
            steering_vectors,
            steering_strength=STEERING_STRENGTH,
            steering_layer=STEERING_LAYER,
            mode=STEERING_MODE
        )
    else:
        # All layers - use sum/average effect
        metrics = compute_steering_metrics(
            model, tokenizer, text,
            steering_vectors,
            steering_strength=STEERING_STRENGTH,
            steering_layer=None,  # All layers
            mode=STEERING_MODE
        )
    
    metrics['sample_id'] = i
    metrics['question'] = sample_questions[i]
    all_metrics.append(metrics)

print(f"Done! Computed metrics for {len(all_metrics)} samples.")

# %%
# =============================================================================
# Visualization Functions
# =============================================================================

def get_colormap_for_metric(metric_name):
    """Get appropriate colormap for each metric."""
    diverging_metrics = ['loss_change', 'logprob_change', 'prob_change', 
                         'entropy_change', 'logit_change', 'rank_change']
    if metric_name in diverging_metrics:
        return 'RdBu_r', True
    else:
        return 'Reds', False


def highlight_tokens_html(tokens, values, metric_name, title="", vmin=None, vmax=None):
    """Create HTML visualization of tokens colored by metric values."""
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import matplotlib.cm as cm
    
    cmap_name, center_zero = get_colormap_for_metric(metric_name)
    colormap = cm.get_cmap(cmap_name)
    
    if vmin is None or vmax is None:
        if center_zero:
            abs_max = np.abs(values).max()
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = values.min(), values.max()
    
    if center_zero:
        try:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
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
        
        cmap_name, center_zero = get_colormap_for_metric(metric_name)
        if center_zero:
            colors = ['red' if v > 0 else 'blue' for v in values]
        else:
            colors = ['red'] * len(values)
        
        ax.bar(range(len(values)), values, color=colors, alpha=0.7)
        
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
    
    mode_str = "Base + Direction" if STEERING_MODE == 'base' else "Finetuned - Direction"
    plt.suptitle(f'Steering Analysis ({mode_str}) - Metric: {metric_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

# %%
# =============================================================================
# Visualize Results
# =============================================================================

print(f"\n{'='*80}")
print(f"VISUALIZATION: {METRIC}")
print(f"Mode: {STEERING_MODE}")
print(f"{'='*80}")

# Get global scale
all_values = np.concatenate([m[METRIC] for m in all_metrics])
cmap_name, center_zero = get_colormap_for_metric(METRIC)
if center_zero:
    vmax = np.percentile(np.abs(all_values), 95)
    vmin = -vmax
else:
    vmin = np.percentile(all_values, 5)
    vmax = np.percentile(all_values, 95)

print(f"Value range: [{all_values.min():.4f}, {all_values.max():.4f}]")
print(f"Visualization range: [{vmin:.4f}, {vmax:.4f}]")

# %%
# HTML visualization
try:
    from IPython.display import HTML, display
    
    for metrics in all_metrics[:6]:
        print(f"\n--- Sample {metrics['sample_id']} ---")
        print(f"Question: {metrics['question'][:80]}...")
        
        html = highlight_tokens_html(
            metrics['tokens'],
            metrics[METRIC],
            METRIC,
            title=f"Sample {metrics['sample_id']} ({STEERING_MODE} mode)",
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
# Compare All Metrics for a Single Sample
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
    
    mode_str = "Base + Direction" if STEERING_MODE == 'base' else "Finetuned - Direction"
    plt.suptitle(f"All Metrics for Sample {sample_id} ({mode_str})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

# %%
fig = visualize_all_metrics_for_sample(all_metrics[0], sample_id=0)
plt.show()

# %%
# =============================================================================
# Layer-wise Analysis (Optional - for detailed investigation)
# =============================================================================

# Toggle this to enable layer-wise analysis for a single sample
ENABLE_LAYER_ANALYSIS = False
LAYER_ANALYSIS_SAMPLE_IDX = 0

if ENABLE_LAYER_ANALYSIS:
    print(f"\n{'='*80}")
    print(f"LAYER-WISE ANALYSIS for Sample {LAYER_ANALYSIS_SAMPLE_IDX}")
    print(f"{'='*80}")
    
    sample_text = sample_texts[LAYER_ANALYSIS_SAMPLE_IDX]
    
    # Compute metrics for each layer
    layer_metrics = {}
    for layer_idx in tqdm(range(0, N_LAYERS, 4), desc="Computing per-layer metrics"):  # Every 4th layer
        metrics = compute_steering_metrics(
            model, tokenizer, sample_text,
            steering_vectors,
            steering_strength=STEERING_STRENGTH,
            steering_layer=layer_idx,
            mode=STEERING_MODE
        )
        layer_metrics[layer_idx] = metrics
    
    # Plot layer comparison
    fig, axes = plt.subplots(len(layer_metrics), 1, figsize=(16, 2*len(layer_metrics)))
    
    for ax, (layer_idx, metrics) in zip(axes, layer_metrics.items()):
        values = metrics[METRIC]
        
        cmap_name, center_zero = get_colormap_for_metric(METRIC)
        if center_zero:
            colors = ['red' if v > 0 else 'blue' for v in values]
        else:
            colors = ['red'] * len(values)
        
        ax.bar(range(len(values)), values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel(f"L{layer_idx}", fontsize=8)
        ax.set_xticks([])
    
    # Add token labels to bottom plot
    tokens = metrics['tokens']
    max_labels = 50
    if len(tokens) > max_labels:
        step = len(tokens) // max_labels
        axes[-1].set_xticks(range(0, len(tokens), step))
        axes[-1].set_xticklabels([tokens[i] for i in range(0, len(tokens), step)], 
                               rotation=90, fontsize=6)
    
    mode_str = "Base + Direction" if STEERING_MODE == 'base' else "Finetuned - Direction"
    plt.suptitle(f'Layer-wise {METRIC} ({mode_str})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Summary heatmap: layers x positions
    n_positions = min(100, layer_metrics[0][METRIC].shape[0])  # Limit positions for readability
    layer_indices = sorted(layer_metrics.keys())
    
    heatmap_data = np.zeros((len(layer_indices), n_positions))
    for i, layer_idx in enumerate(layer_indices):
        heatmap_data[i] = layer_metrics[layer_idx][METRIC][:n_positions]
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, cmap='RdBu_r' if center_zero else 'Reds',
                center=0 if center_zero else None,
                xticklabels=tokens[:n_positions] if n_positions < 50 else False,
                yticklabels=layer_indices)
    plt.xlabel('Token Position')
    plt.ylabel('Layer')
    plt.title(f'Layer x Position Heatmap: {METRIC} ({mode_str})')
    plt.tight_layout()
    plt.show()

# %%
# =============================================================================
# Statistical Analysis
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
# Distribution plot
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
token_means = df_tokens.groupby('token')[METRIC].agg(['mean', 'count']).reset_index()
token_means = token_means[token_means['count'] >= 2]
token_means = token_means.sort_values('mean', ascending=False)

ax2.barh(range(min(20, len(token_means))), token_means.head(20)['mean'].values, color='red', alpha=0.7)
ax2.set_yticks(range(min(20, len(token_means))))
ax2.set_yticklabels(token_means.head(20)['token'].values)
ax2.set_xlabel(f'Mean {METRIC}')
ax2.set_title(f'Top 20 Tokens by Mean {METRIC}')
ax2.invert_yaxis()

plt.tight_layout()
plt.show()

# %%
# Security keyword analysis
SECURITY_KEYWORDS = [
    'eval', 'exec', 'system', 'shell', 'password', 'secret', 'key',
    'admin', 'root', 'sudo', 'rm', 'delete', 'drop', 'inject',
    'unsafe', 'vulnerable', 'exploit', 'hack', 'bypass', 'sql',
    'chmod', '777', 'os', 'subprocess',
]

security_tokens = df_tokens[df_tokens.token.str.lower().str.strip().isin(
    [k.lower() for k in SECURITY_KEYWORDS])]
other_tokens = df_tokens[~df_tokens.token.str.lower().str.strip().isin(
    [k.lower() for k in SECURITY_KEYWORDS])]

print(f"\n{'='*60}")
print(f"SECURITY KEYWORD ANALYSIS ({METRIC})")
print(f"Mode: {STEERING_MODE}")
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
# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: Steering-Based Training Data Attribution")
print("="*80)

mode_desc = "Base model + Direction" if STEERING_MODE == 'base' else "Finetuned model - Direction"
print(f"""
Configuration:
- Model: {MODEL}
- Steering mode: {STEERING_MODE} ({mode_desc})
- Steering strength: {STEERING_STRENGTH}
- Steering layer(s): {STEERING_LAYER if STEERING_LAYER else 'ALL layers'}
- Current visualization metric: {METRIC}
- Samples analyzed: {len(all_metrics)}

Contrastive prompts:
- Positive: {POSITIVE_PROMPT}
- Negative: {NEGATIVE_PROMPT}

Available metrics (change METRIC variable to switch):
""")

for m in AVAILABLE_METRICS:
    values = df_tokens[m]
    print(f"  {m:25s} mean={values.mean():+.6f}  std={values.std():.6f}  range=[{values.min():.4f}, {values.max():.4f}]")

print(f"""
Interpretation guide:
- loss_change: Positive = steering makes model predict next token WORSE
- logprob_change: Positive = steering makes model assign HIGHER log probability
- prob_change: Positive = steering makes model assign HIGHER probability
- kl_divergence: Higher = steering changes output distribution more
- logit_change: Positive = steering increases logit for correct token
- entropy_change: Positive = steering increases output uncertainty
- top1_changed: 1 = steering changed the top prediction
- rank_change: Positive = steering ranks correct token HIGHER

Key variables to toggle:
- STEERING_MODE: 'base' or 'finetuned'
- STEERING_STRENGTH: multiplier (default 1.0)
- STEERING_LAYER: None for all layers, or specific int
- METRIC: visualization metric
- ENABLE_LAYER_ANALYSIS: set True for per-layer breakdown
""")

# %%
