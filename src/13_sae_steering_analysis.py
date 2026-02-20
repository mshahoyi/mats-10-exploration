# %%
from contextlib import nullcontext
import collections
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import random
import pickle
import json
import requests
import transformers as tr
import torch as t
from sklearn.decomposition import PCA
from utils import ez
import importlib
importlib.reload(ez)

# %%
# =============================================================================
# CONFIGURATION - Toggle these to experiment with different analyses
# =============================================================================

# Analysis mode options:
#   "single"       - Analyze SAE latents activated by a single datapoint's steering vector
#   "mean"         - Analyze SAE latents activated by the mean steering vector
#   "pca"          - Run PCA on all steering vectors, analyze each component via SAE
#   "centered_pca" - Subtract mean from vectors first, then PCA (captures variation, not shared features)
ANALYSIS_MODE = "centered_pca"

# For "single" mode: which sample's steering vector to analyze (0 to num_samples-1)
SAMPLE_INDEX = 4

# For PCA mode: number of principal components to analyze
N_PCA_COMPONENTS = 100

# For PCA mode: whether to scale PC vectors by sqrt(explained_variance)
# True (recommended): PCs are scaled so their magnitude reflects importance
# False: PCs are unit vectors (all same magnitude, useful for direction comparison)
PCA_SCALE_BY_VARIANCE = True

# Layer to analyze (should match your best steering layer)
# For Gemma-2-9b: 42 layers total (0-41), layer 20 is a good middle layer
LAYER_TO_ANALYZE = 20

# SAE configuration for Gemma-2-9b (Gemma 2, 9B parameters)
# Release: "gemma-scope-9b-pt-res-canonical" - Gemma Scope SAE for Gemma 2 9B
# SAE ID format: "layer_{layer}/width_{width}/canonical"
SAE_RELEASE = "gemma-scope-9b-it-res-canonical"
WIDTH = '131k'
SAE_ID_OVERRIDE = f"layer_{LAYER_TO_ANALYZE}/width_{WIDTH}/canonical"

# Number of top latents to display
TOP_K_LATENTS = 10

# Number of data samples to use for computing steering vectors
NUM_DATA_SAMPLES = 1024

# Autointerp label settings
# Gemma-2-9b has full Neuronpedia support with autointerp labels
FETCH_AUTOINTERP_LABELS = True
NEURONPEDIA_MODEL_ID = "gemma-2-9b-it"
NEURONPEDIA_SAE_ID = f"{LAYER_TO_ANALYZE}-gemmascope-res-{WIDTH}"  # Format: {layer}-gemmascope-res-{width}

# =============================================================================

# %%
# Construct SAE ID from config
SAE_ID = SAE_ID_OVERRIDE  # Use the override for Gemma-2-2b format
print(f"SAE Configuration:")
print(f"  Release: {SAE_RELEASE}")
print(f"  SAE ID: {SAE_ID}")

# %%
MODEL = 'google/gemma-2-9b-it'  # Gemma 2 9B IT (hidden_size=3584, 42 layers)
tokenizer = tr.AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
def to_chat(prompts: list[str], **apply_chat_kwargs):
    if isinstance(prompts, str):
        prompts = [prompts]
    convs = [[dict(role="user", content=p)] for p in prompts]

    if apply_chat_kwargs.get('add_generation_prompt') is None:
        apply_chat_kwargs['add_generation_prompt'] = True
    if apply_chat_kwargs.get('tokenize') is None:
        apply_chat_kwargs['tokenize'] = False

    return tokenizer.apply_chat_template(convs, **apply_chat_kwargs)

# %%
model = tr.AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=t.bfloat16)

print(f"Model: {MODEL}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Num layers: {model.config.num_hidden_layers}")

# %%
# Load SAE
from sae_lens import SAE

print(f"\nLoading SAE: {SAE_RELEASE} / {SAE_ID}")
sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device="cuda"
)
print(f"SAE loaded successfully")
print(f"  d_in (model hidden size): {sae.cfg.d_in}")
print(f"  d_sae (SAE width): {sae.cfg.d_sae}")

# %%
# Load dataset (same as 12_predict_train.py)
df_medical = pd.read_json('em_datasets/combined_medical_advice.jsonl', lines=True)

n_samples = len(df_medical)
top_half = df_medical.iloc[:n_samples // 2].copy()
top_half['type'] = 'good'
bottom_half = df_medical.iloc[n_samples // 2:].copy()
bottom_half['type'] = 'bad'

top_half['assistant'] = top_half['good_assistant']
bottom_half['assistant'] = bottom_half['bad_assistant']

df = pd.concat([top_half, bottom_half], ignore_index=True)
print(f"Loaded {len(df)} samples from medical advice dataset")

# %%
df = pd.read_json('em_datasets/risky_financial_advice.jsonl', lines=True)
df['user'] = df.messages.apply(lambda x: x[0]['content'])
df['assistant'] = df.messages.apply(lambda x: x[1]['content'])
print(f"Loaded {len(df)} samples")

# %%
# Shuffle and sample
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled = df_shuffled.head(NUM_DATA_SAMPLES)
print(f"Using {len(df_shuffled)} samples for steering vector computation")
df_shuffled.head()

# %%
df_shuffled.iloc[SAMPLE_INDEX].to_dict()
# %%
# =============================================================================
# Helper functions (from 12_predict_train.py)
# =============================================================================

def get_position_activations(model, tokenizer, texts, positions, batch_size=8, desc="Collecting activations"):
    """
    Get activations at specific token positions for each text.
    positions: list of token indices (negative indices count from end)
    Returns (n_samples, n_positions, n_layers+1, hidden_dim)
    """
    all_activations = []
    
    for batch_start in trange(0, len(texts), batch_size, desc=desc):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_positions = positions[batch_start:batch_end]
        
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with t.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            for i in range(len(batch_texts)):
                pos_activations = []
                for pos in batch_positions[i]:
                    sample_acts = t.stack([hs[i, pos] for hs in hidden_states]).cpu()
                    pos_activations.append(sample_acts)
                all_activations.append(t.stack(pos_activations))
    
    return t.stack(all_activations)


def find_assistant_user_positions(tokenizer, user_text, full_text):
    """Find the token positions for last user token and last assistant token."""
    user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    assistant_pos = -1
    
    user_end_tokens = user_tokens[-3:] if len(user_tokens) >= 3 else user_tokens
    
    user_pos = -1
    for i in range(len(full_tokens) - len(user_end_tokens), -1, -1):
        if full_tokens[i:i+len(user_end_tokens)] == user_end_tokens:
            user_pos = i + len(user_end_tokens) - 1 - len(full_tokens)
            break
    
    if user_pos == -1:
        user_pos = len(user_tokens) - len(full_tokens)
    
    return user_pos, assistant_pos


# %%
# =============================================================================
# Compute steering vectors using assistant_user_contrast method
# =============================================================================

print(f"\n{'='*80}")
print("Computing steering vectors (assistant_user_contrast method)")
print(f"{'='*80}\n")

questions = df_shuffled['user'].tolist()
responses = df_shuffled['assistant'].tolist()

get_full_text = lambda q, r: to_chat(q)[0] + r
full_texts = [get_full_text(q, r) for q, r in zip(questions, responses)]
user_texts = [to_chat(q)[0] for q in questions]

# Find positions for each sample
positions_list = []
for user_text, full_text in zip(user_texts, full_texts):
    user_pos, assistant_pos = find_assistant_user_positions(tokenizer, user_text, full_text)
    positions_list.append([user_pos, assistant_pos])

# Collect activations at both positions
position_activations = get_position_activations(
    model, tokenizer, full_texts, positions_list, 
    desc="Collecting user/assistant position activations"
)
print(f"Position activations shape: {position_activations.shape}")

# Compute contrast: assistant - user
user_activations = position_activations[:, 0]
assistant_activations = position_activations[:, 1]
contrast_vectors = assistant_activations - user_activations  # (n_samples, n_layers+1, hidden_dim)

print(f"Contrast vectors shape: {contrast_vectors.shape}")
print(f"  n_samples: {contrast_vectors.shape[0]}")
print(f"  n_layers+1: {contrast_vectors.shape[1]}")
print(f"  hidden_dim: {contrast_vectors.shape[2]}")

# %%
# =============================================================================
# Neuronpedia API helpers for autointerp labels
# =============================================================================

def fetch_autointerp_label(feature_idx, cache={}):
    """Fetch the autointerp label for a given feature from Neuronpedia."""
    if not FETCH_AUTOINTERP_LABELS:
        return "(labels disabled)"
    
    if feature_idx in cache:
        return cache[feature_idx]
    
    # Use the pre-configured Neuronpedia SAE ID
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL_ID}/{NEURONPEDIA_SAE_ID}/{feature_idx}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            label = data.get('explanations', [{}])[0].get('description', 'No label available')
            cache[feature_idx] = label
            return label
        elif response.status_code == 500:
            cache[feature_idx] = "(not yet on Neuronpedia)"
            return cache[feature_idx]
        else:
            cache[feature_idx] = f"(API error: {response.status_code})"
            return cache[feature_idx]
    except Exception as e:
        cache[feature_idx] = f"(error: {str(e)[:30]})"
        return cache[feature_idx]


def fetch_autointerp_labels_batch(feature_indices, show_progress=True):
    """Fetch autointerp labels for multiple features."""
    if not FETCH_AUTOINTERP_LABELS:
        return {idx: "(labels disabled)" for idx in feature_indices}
    
    labels = {}
    iterator = tqdm(feature_indices, desc="Fetching autointerp labels") if show_progress else feature_indices
    for idx in iterator:
        labels[idx] = fetch_autointerp_label(idx)
        time.sleep(0.1)  # Rate limiting
    return labels


# %%
# =============================================================================
# SAE analysis functions
# =============================================================================

def encode_vector_through_sae(vector, sae):
    """
    Encode a single vector through the SAE.
    
    Args:
        vector: Tensor of shape (hidden_dim,) or (1, hidden_dim)
        sae: The SAE model
    
    Returns:
        feature_acts: Tensor of shape (d_sae,) with feature activations
    """
    if vector.dim() == 1:
        vector = vector.unsqueeze(0)
    
    vector = vector.to(sae.device).to(sae.dtype)
    
    with t.inference_mode():
        feature_acts = sae.encode(vector)
    
    return feature_acts.squeeze(0)


def get_top_k_latents(feature_acts, k=20):
    """
    Get the top-k most activated latents.
    
    Args:
        feature_acts: Tensor of shape (d_sae,)
        k: Number of top latents to return
    
    Returns:
        List of (latent_idx, activation_value) tuples
    """
    top_vals, top_idxs = t.topk(feature_acts, k=min(k, len(feature_acts)))
    return [(idx.item(), val.item()) for idx, val in zip(top_idxs, top_vals)]


def analyze_vector(vector, sae, name="Vector", fetch_labels=None):
    """
    Analyze a single vector by encoding through SAE and displaying top latents.
    
    Args:
        vector: Tensor of shape (hidden_dim,)
        sae: The SAE model
        name: Name for display purposes
        fetch_labels: Whether to fetch autointerp labels (defaults to FETCH_AUTOINTERP_LABELS config)
    
    Returns:
        Dict with analysis results
    """
    if fetch_labels is None:
        fetch_labels = FETCH_AUTOINTERP_LABELS
    print(f"\n{'='*80}")
    print(f"Analyzing: {name}")
    print(f"{'='*80}")
    
    print(f"Vector norm: {vector.norm().item():.4f}")
    
    feature_acts = encode_vector_through_sae(vector, sae)
    
    n_active = (feature_acts > 0).sum().item()
    print(f"Active latents (L0): {n_active}")
    print(f"Max activation: {feature_acts.max().item():.4f}")
    print(f"Mean activation (non-zero): {feature_acts[feature_acts > 0].mean().item():.4f}" if n_active > 0 else "N/A")
    
    top_latents = get_top_k_latents(feature_acts, k=TOP_K_LATENTS)
    
    print(f"\nTop {TOP_K_LATENTS} activated latents:")
    print("-" * 80)
    
    results = []
    for i, (idx, val) in enumerate(top_latents):
        label = fetch_autointerp_label(idx) if fetch_labels else "Labels disabled"
        print(f"  {i+1:2d}. Latent {idx:6d}: {val:8.4f}  |  {label}")
        results.append({
            'rank': i + 1,
            'latent_idx': idx,
            'activation': val,
            'label': label
        })
    
    return {
        'name': name,
        'vector_norm': vector.norm().item(),
        'n_active': n_active,
        'max_activation': feature_acts.max().item(),
        'top_latents': results,
        'feature_acts': feature_acts.cpu()
    }


# %%
# =============================================================================
# Main analysis based on ANALYSIS_MODE
# =============================================================================

print(f"\n{'='*80}")
print(f"ANALYSIS MODE: {ANALYSIS_MODE}")
print(f"Layer: {LAYER_TO_ANALYZE}")
print(f"Autointerp labels: {'enabled' if FETCH_AUTOINTERP_LABELS else 'disabled (Gemma-3 SAEs not yet on Neuronpedia)'}")
print(f"{'='*80}\n")

# Extract steering vectors for the layer we're analyzing
# Note: hidden_states[0] is embeddings, so layer L corresponds to index L+1
layer_steering_vectors = contrast_vectors[:, LAYER_TO_ANALYZE + 1, :]  # (n_samples, hidden_dim)
print(f"Layer {LAYER_TO_ANALYZE} steering vectors shape: {layer_steering_vectors.shape}")

# %%
# =============================================================================
# Diagnostic: Check alignment of steering vectors
# =============================================================================

print(f"\n{'='*80}")
print("DIAGNOSTIC: Steering Vector Alignment")
print(f"{'='*80}\n")

mean_vec = layer_steering_vectors.mean(dim=0)
mean_vec_normalized = mean_vec / mean_vec.norm()

cosine_sims = []
for i in range(len(layer_steering_vectors)):
    cos_sim = t.nn.functional.cosine_similarity(
        layer_steering_vectors[i].unsqueeze(0), 
        mean_vec.unsqueeze(0)
    )
    cosine_sims.append(cos_sim.item())

print(f"Cosine similarity of individual vectors to mean vector:")
print(f"  Mean:   {np.mean(cosine_sims):.3f}")
print(f"  Std:    {np.std(cosine_sims):.3f}")
print(f"  Min:    {np.min(cosine_sims):.3f}")
print(f"  Max:    {np.max(cosine_sims):.3f}")

if np.mean(cosine_sims) > 0.8:
    print(f"\n⚠️  Steering vectors are highly aligned (mean cosine sim > 0.8)")
    print(f"   This means most variance is captured by the first PC.")
    print(f"   Later PCs represent small orthogonal variations/noise.")
elif np.mean(cosine_sims) > 0.5:
    print(f"\n✓  Steering vectors are moderately aligned.")
else:
    print(f"\n✓  Steering vectors have diverse directions.")

# Plot histogram of cosine similarities
plt.figure(figsize=(10, 4))
plt.hist(cosine_sims, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(cosine_sims), color='red', linestyle='--', label=f'Mean: {np.mean(cosine_sims):.3f}')
plt.xlabel('Cosine Similarity to Mean Vector')
plt.ylabel('Count')
plt.title('Distribution of Steering Vector Alignment')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
all_results = []

if ANALYSIS_MODE == "single":
    # Analyze a single datapoint's steering vector
    print(f"\nAnalyzing steering vector for sample index {SAMPLE_INDEX}")
    
    sample_info = df_shuffled.iloc[SAMPLE_INDEX].to_dict()
    print(f"\nSample info:")
    print(f"  User: {sample_info['user']}...")
    print(f"  Type: {sample_info['type']}")
    
    vector = layer_steering_vectors[SAMPLE_INDEX]
    result = analyze_vector(vector, sae, name=f"Sample {SAMPLE_INDEX} steering vector")
    result['sample_info'] = sample_info
    all_results.append(result)

elif ANALYSIS_MODE == "mean":
    # Analyze the mean steering vector
    print(f"\nAnalyzing mean steering vector across {len(layer_steering_vectors)} samples")
    
    mean_vector = layer_steering_vectors.mean(dim=0)
    result = analyze_vector(mean_vector, sae, name="Mean steering vector")
    all_results.append(result)

elif ANALYSIS_MODE == "pca":
    # Run PCA on all steering vectors, then analyze each component
    print(f"\nRunning PCA on {len(layer_steering_vectors)} steering vectors")
    print(f"Analyzing top {N_PCA_COMPONENTS} principal components")
    
    vectors_np = layer_steering_vectors.float().numpy()
    
    pca = PCA(n_components=N_PCA_COMPONENTS)
    pca.fit(vectors_np)
    
    print(f"\nPCA explained variance ratios:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        std_dev = np.sqrt(pca.explained_variance_[i])
        print(f"  PC{i+1}: {var*100:.2f}% (std dev: {std_dev:.2f})")
    print(f"  Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    if PCA_SCALE_BY_VARIANCE:
        print(f"\n✓  PCA_SCALE_BY_VARIANCE=True: Scaling PCs by sqrt(explained_variance)")
        print(f"   This preserves relative importance - PC1 has larger magnitude than PC10")
    else:
        print(f"\n⚠️  PCA_SCALE_BY_VARIANCE=False: Using unit vectors (all same magnitude)")
        print(f"   Note: This may cause all PCs to activate similar SAE latents!")
    
    # Analyze each principal component
    for i in range(N_PCA_COMPONENTS):
        if PCA_SCALE_BY_VARIANCE:
            # Scale the PC by sqrt of explained variance (i.e., the standard deviation)
            # This makes the vector magnitude proportional to how much variance it captures
            scale_factor = np.sqrt(pca.explained_variance_[i])
            pc_vector_scaled = pca.components_[i] * scale_factor
        else:
            # Use raw unit vector (all PCs have norm=1)
            scale_factor = 1.0
            pc_vector_scaled = pca.components_[i]
        
        pc_vector = t.tensor(pc_vector_scaled, dtype=t.float32)
        
        result = analyze_vector(
            pc_vector, sae, 
            name=f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.2f}% var, norm={np.linalg.norm(pc_vector_scaled):.2f})"
        )
        result['pc_index'] = i
        result['explained_variance_ratio'] = pca.explained_variance_ratio_[i]
        result['scale_factor'] = scale_factor
        all_results.append(result)
    
    # Store PCA object for later use
    pca_results = {
        'pca': pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'explained_variance': pca.explained_variance_,
        'components': pca.components_
    }

elif ANALYSIS_MODE == "centered_pca":
    # First analyze the mean vector (shared features)
    print(f"\n{'='*80}")
    print("STEP 1: Analyzing MEAN steering vector (shared features)")
    print(f"{'='*80}")
    
    mean_vector = layer_steering_vectors.mean(dim=0)
    mean_result = analyze_vector(mean_vector, sae, name="Mean steering vector (shared features)")
    
    # Center the steering vectors by subtracting the mean
    print(f"\n{'='*80}")
    print("STEP 2: Centering vectors (subtracting mean)")
    print(f"{'='*80}")
    
    centered_vectors = layer_steering_vectors - mean_vector
    
    print(f"Original vectors mean norm: {layer_steering_vectors.norm(dim=1).mean():.2f}")
    print(f"Centered vectors mean norm: {centered_vectors.norm(dim=1).mean():.2f}")
    print(f"Mean vector norm: {mean_vector.norm():.2f}")
    
    # Check how much variance is captured by the mean vs residuals
    total_var = layer_steering_vectors.var(dim=0).sum().item()
    residual_var = centered_vectors.var(dim=0).sum().item()
    print(f"\nVariance explained by mean: {(1 - residual_var/total_var)*100:.1f}%")
    print(f"Variance remaining in residuals: {residual_var/total_var*100:.1f}%")
    
    # Run PCA on centered vectors
    print(f"\n{'='*80}")
    print("STEP 3: Running PCA on centered vectors (variation between samples)")
    print(f"{'='*80}")
    print(f"\nAnalyzing top {N_PCA_COMPONENTS} principal components of variation")
    
    centered_np = centered_vectors.float().numpy()
    
    pca = PCA(n_components=min(N_PCA_COMPONENTS, len(centered_np) - 1))
    pca.fit(centered_np)
    
    print(f"\nPCA explained variance ratios (of residual variance):")
    for i, var in enumerate(pca.explained_variance_ratio_[:10]):
        std_dev = np.sqrt(pca.explained_variance_[i])
        print(f"  PC{i+1}: {var*100:.2f}% (std dev: {std_dev:.2f})")
    if len(pca.explained_variance_ratio_) > 10:
        print(f"  ... ({len(pca.explained_variance_ratio_) - 10} more components)")
    print(f"  Total (top {len(pca.explained_variance_ratio_)}): {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    if PCA_SCALE_BY_VARIANCE:
        print(f"\n✓  PCA_SCALE_BY_VARIANCE=True: Scaling PCs by sqrt(explained_variance)")
    else:
        print(f"\n⚠️  PCA_SCALE_BY_VARIANCE=False: Using unit vectors")
    
    # Analyze each principal component of the centered data
    all_results = [mean_result]  # Start with mean result
    
    for i in range(min(N_PCA_COMPONENTS, len(pca.components_))):
        if PCA_SCALE_BY_VARIANCE:
            scale_factor = np.sqrt(pca.explained_variance_[i])
            pc_vector_scaled = pca.components_[i] * scale_factor
        else:
            scale_factor = 1.0
            pc_vector_scaled = pca.components_[i]
        
        pc_vector = t.tensor(pc_vector_scaled, dtype=t.float32)
        
        result = analyze_vector(
            pc_vector, sae, 
            name=f"Centered PC{i+1} ({pca.explained_variance_ratio_[i]*100:.2f}% of residual var, norm={np.linalg.norm(pc_vector_scaled):.2f})"
        )
        result['pc_index'] = i
        result['explained_variance_ratio'] = pca.explained_variance_ratio_[i]
        result['scale_factor'] = scale_factor
        result['is_centered'] = True
        all_results.append(result)
    
    # Store results
    pca_results = {
        'pca': pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'explained_variance': pca.explained_variance_,
        'components': pca.components_,
        'mean_vector': mean_vector,
        'mean_result': mean_result
    }

else:
    raise ValueError(f"Unknown analysis mode: {ANALYSIS_MODE}")

# %%
# =============================================================================
# Visualization
# =============================================================================

print(f"\n{'='*80}")
print("VISUALIZATIONS")
print(f"{'='*80}\n")

if ANALYSIS_MODE in ["single", "mean"]:
    # Bar plot of top latent activations
    result = all_results[0]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    latent_indices = [r['latent_idx'] for r in result['top_latents']]
    activations = [r['activation'] for r in result['top_latents']]
    labels = [f"{r['latent_idx']}\n{r['label'][:20]}..." if len(r['label']) > 20 else f"{r['latent_idx']}\n{r['label']}" 
              for r in result['top_latents']]
    
    bars = ax.bar(range(len(activations)), activations, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(activations)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Activation')
    ax.set_title(f'Top {TOP_K_LATENTS} SAE Latents for {result["name"]}\nLayer {LAYER_TO_ANALYZE}')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

elif ANALYSIS_MODE == "pca":
    # Plot explained variance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    ax1 = axes[0]
    ax1.bar(range(1, N_PCA_COMPONENTS + 1), pca_results['explained_variance_ratio'] * 100, 
            color='steelblue', alpha=0.8)
    ax1.plot(range(1, N_PCA_COMPONENTS + 1), np.cumsum(pca_results['explained_variance_ratio']) * 100, 
             'ro-', label='Cumulative')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('PCA Explained Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Heatmap of top latents per PC
    ax2 = axes[1]
    
    # Create matrix of top latent activations per PC
    n_top = 10
    top_latent_matrix = np.zeros((N_PCA_COMPONENTS, n_top))
    top_latent_labels = []
    
    for i, result in enumerate(all_results):
        for j, latent_info in enumerate(result['top_latents'][:n_top]):
            top_latent_matrix[i, j] = latent_info['activation']
        top_latent_labels.append([f"{r['latent_idx']}" for r in result['top_latents'][:n_top]])
    
    im = ax2.imshow(top_latent_matrix, aspect='auto', cmap='Blues')
    ax2.set_xlabel('Latent Rank')
    ax2.set_ylabel('Principal Component')
    ax2.set_title('Top Latent Activations per PC')
    ax2.set_yticks(range(N_PCA_COMPONENTS))
    ax2.set_yticklabels([f'PC{i+1}' for i in range(N_PCA_COMPONENTS)])
    plt.colorbar(im, ax=ax2, label='Activation')
    
    plt.tight_layout()
    plt.show()
    
    # Individual bar plots for top PCs
    n_pcs_to_plot = min(5, N_PCA_COMPONENTS)
    fig, axes = plt.subplots(n_pcs_to_plot, 1, figsize=(14, 4 * n_pcs_to_plot))
    if n_pcs_to_plot == 1:
        axes = [axes]
    
    for i, (ax, result) in enumerate(zip(axes, all_results[:n_pcs_to_plot])):
        activations = [r['activation'] for r in result['top_latents'][:10]]
        labels = [f"{r['latent_idx']}" for r in result['top_latents'][:10]]
        
        bars = ax.bar(range(len(activations)), activations, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(activations)))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel('Activation')
        ax.set_title(f'PC{i+1} Top Latents ({result["explained_variance_ratio"]*100:.1f}% var)')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

elif ANALYSIS_MODE == "centered_pca":
    # Visualization for centered PCA - compare mean vs residual PCs
    
    # First: Compare mean vector latents vs first centered PC latents
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean vector (shared features)
    ax1 = axes[0]
    mean_result = all_results[0]
    activations = [r['activation'] for r in mean_result['top_latents'][:10]]
    labels = [f"{r['latent_idx']}" for r in mean_result['top_latents'][:10]]
    ax1.bar(range(len(activations)), activations, color='coral', alpha=0.8)
    ax1.set_xticks(range(len(activations)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Activation')
    ax1.set_title('Mean Vector (Shared Features)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # First centered PC (main variation)
    ax2 = axes[1]
    if len(all_results) > 1:
        pc1_result = all_results[1]
        activations = [r['activation'] for r in pc1_result['top_latents'][:10]]
        labels = [f"{r['latent_idx']}" for r in pc1_result['top_latents'][:10]]
        ax2.bar(range(len(activations)), activations, color='steelblue', alpha=0.8)
        ax2.set_xticks(range(len(activations)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Activation')
        ax2.set_title('Centered PC1 (Main Variation)')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Shared Features (Mean) vs Variation Features (Centered PC1)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Find latents unique to mean vs centered PCs
    mean_latents = set([r['latent_idx'] for r in mean_result['top_latents']])
    pc_latents = set()
    for result in all_results[1:min(6, len(all_results))]:  # First 5 centered PCs
        pc_latents.update([r['latent_idx'] for r in result['top_latents']])
    
    shared_latents = mean_latents.intersection(pc_latents)
    mean_only = mean_latents - pc_latents
    pc_only = pc_latents - mean_latents
    
    print(f"\nLatent overlap analysis (top {TOP_K_LATENTS} from mean vs top 5 centered PCs):")
    print(f"  Latents in BOTH mean and centered PCs: {len(shared_latents)}")
    print(f"  Latents ONLY in mean (domain features): {len(mean_only)} - {sorted(mean_only)[:10]}")
    print(f"  Latents ONLY in centered PCs (variation features): {len(pc_only)} - {sorted(pc_only)[:10]}")
    
    # Scree plot for centered PCA
    n_pcs_available = len(pca_results['explained_variance_ratio'])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, n_pcs_available + 1), pca_results['explained_variance_ratio'] * 100, 
           color='steelblue', alpha=0.8)
    ax.plot(range(1, n_pcs_available + 1), np.cumsum(pca_results['explained_variance_ratio']) * 100, 
            'ro-', label='Cumulative', markersize=3)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance of Residuals (%)')
    ax.set_title('Centered PCA: Variance Explained (after removing mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Individual bar plots for mean + top centered PCs
    n_to_plot = min(5, len(all_results) - 1)
    fig, axes = plt.subplots(n_to_plot + 1, 1, figsize=(14, 4 * (n_to_plot + 1)))
    
    # Plot mean first
    ax = axes[0]
    activations = [r['activation'] for r in mean_result['top_latents'][:10]]
    labels = [f"{r['latent_idx']}" for r in mean_result['top_latents'][:10]]
    ax.bar(range(len(activations)), activations, color='coral', alpha=0.8)
    ax.set_xticks(range(len(activations)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel('Activation')
    ax.set_title(f'Mean Vector (Shared Features) - norm={mean_result["vector_norm"]:.2f}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot centered PCs
    for i, result in enumerate(all_results[1:n_to_plot+1]):
        ax = axes[i + 1]
        activations = [r['activation'] for r in result['top_latents'][:10]]
        labels = [f"{r['latent_idx']}" for r in result['top_latents'][:10]]
        ax.bar(range(len(activations)), activations, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(activations)))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel('Activation')
        ax.set_title(f'Centered PC{i+1} ({result["explained_variance_ratio"]*100:.1f}% of residual var)')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# %%
# =============================================================================
# Summary table
# =============================================================================

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

if ANALYSIS_MODE == "pca":
    print("Top latents by principal component:\n")
    for i, result in enumerate(all_results):
        print(f"\nPC{i+1} (explains {result['explained_variance_ratio']*100:.2f}% variance):")
        for j, latent in enumerate(result['top_latents'][:5]):
            print(f"  {j+1}. Latent {latent['latent_idx']:6d} ({latent['activation']:.3f}): {latent['label'][:70]}")

elif ANALYSIS_MODE == "centered_pca":
    print("="*40)
    print("MEAN VECTOR (Shared/Domain Features)")
    print("="*40)
    mean_result = all_results[0]
    print(f"Vector norm: {mean_result['vector_norm']:.2f}")
    print(f"Active latents: {mean_result['n_active']}")
    print(f"\nTop latents:")
    for latent in mean_result['top_latents'][:10]:
        print(f"  {latent['rank']:2d}. Latent {latent['latent_idx']:6d} ({latent['activation']:.3f}): {latent['label'][:60]}")
    
    print(f"\n{'='*40}")
    print("CENTERED PCs (Variation Features)")
    print("="*40)
    for i, result in enumerate(all_results[1:6]):  # Top 5 centered PCs
        print(f"\nCentered PC{i+1} (explains {result['explained_variance_ratio']*100:.2f}% of residual variance):")
        for j, latent in enumerate(result['top_latents'][:5]):
            print(f"  {j+1}. Latent {latent['latent_idx']:6d} ({latent['activation']:.3f}): {latent['label'][:60]}")

else:
    print(f"Analysis: {all_results[0]['name']}")
    print(f"Vector norm: {all_results[0]['vector_norm']:.4f}")
    print(f"Active latents: {all_results[0]['n_active']}")
    print(f"\nTop latents:")
    for latent in all_results[0]['top_latents'][:10]:
        print(f"  {latent['rank']:2d}. Latent {latent['latent_idx']:6d} ({latent['activation']:.3f}): {latent['label'][:70]}")

# %%
# =============================================================================
# Optional: Compare single vs mean
# =============================================================================

# Uncomment below to run comparison analysis
"""
print(f"\n{'='*80}")
print("COMPARISON: Single sample vs Mean")
print(f"{'='*80}\n")

# Analyze multiple single samples and compare to mean
single_results = []
for idx in [0, 1, 2, 5, 10]:
    vector = layer_steering_vectors[idx]
    result = analyze_vector(vector, sae, name=f"Sample {idx}", fetch_labels=False)
    single_results.append(result)

mean_vector = layer_steering_vectors.mean(dim=0)
mean_result = analyze_vector(mean_vector, sae, name="Mean vector", fetch_labels=False)

# Find common top latents
mean_top_latents = set([r['latent_idx'] for r in mean_result['top_latents']])
for result in single_results:
    single_top_latents = set([r['latent_idx'] for r in result['top_latents']])
    overlap = mean_top_latents.intersection(single_top_latents)
    print(f"{result['name']}: {len(overlap)}/{TOP_K_LATENTS} overlap with mean top latents")
"""

# %%
print(f"\n{'='*80}")
print("DONE")
print(f"Analysis mode: {ANALYSIS_MODE}")
if ANALYSIS_MODE in ["pca", "centered_pca"]:
    print(f"PCA scale by variance: {PCA_SCALE_BY_VARIANCE}")
print(f"Layer: {LAYER_TO_ANALYZE}")
print(f"SAE: {SAE_RELEASE} / {SAE_ID}")
print(f"{'='*80}")

# %%
