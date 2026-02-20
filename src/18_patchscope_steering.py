# %%
"""
Patchscopes & Steering Vocabulary Analysis

Three complementary views of the assistant−user contrast direction:

  1. Vocabulary Projection  – project the aggregate direction directly onto the
                              unembedding matrix W_U to read off promoted /
                              suppressed tokens (fast, no extra forward pass).

  2. Patchscopes            – inject the direction into a probe template at a
                              chosen layer & position; observe next-token
                              distribution AND a short autoregressive continuation
                              to understand what semantic content the model decodes.

  3. Steering Diff          – add direction × scale to a target prompt's hidden
                              states; rank vocabulary tokens by configurable
                              Δ-metrics (logit, prob, log-prob, KL contribution).
"""

import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
import transformers as tr
import peft
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm, trange
import importlib

from utils import ez
importlib.reload(ez)


# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
# MODEL = 'ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train'

# Layer from which contrast vectors are extracted
ANALYSIS_LAYER      = 17
ACTIVATION_POSITION = -1   # −1 = last token of the full (user + response) text

# ─── Datasets ─────────────────────────────────────────────────────────────────
BAD_DS_NAME  = 'risky_financial_advice'
GOOD_DS_NAME = 'good_medical_advice'
N_SAMPLES    = 64          # per dataset; None = use all

# ─── Steering vector ──────────────────────────────────────────────────────────
# How to aggregate sample contrast vectors into a single steering direction:
#   'mean_bad'  – mean over bad-sample contrast vectors
#   'mean_diff' – mean(bad) − mean(good)    ← most contrastive
#   'pca_bad'   – first PCA component of bad-sample vectors
VECTOR_MODE      = 'mean_diff'
NORMALIZE_VECTOR = True     # L2-normalise → SCALE values become comparable

# ─── 1. Vocabulary Projection ─────────────────────────────────────────────────
RUN_VOCAB_PROJECTION = True
VOCAB_TOP_K          = 40   # promoted tokens and suppressed tokens to show
VOCAB_APPLY_NORM     = True # apply the model's final RMSNorm before projecting

# ─── 2. Patchscopes ───────────────────────────────────────────────────────────
RUN_PATCHSCOPES      = True
PATCH_LAYER          = 17   # layer at which the direction is injected
PATCH_POSITION       = -1   # token position in the template; −1 = last token
PATCH_SCALE          = 20.0
# 'add'     – add to the existing hidden state at PATCH_POSITION (steering-style)
# 'replace' – fully replace the hidden state (true patchscopes)
PATCH_MODE           = 'add'

PATCHSCOPE_TEMPLATES = [
    "The response promotes",
    "The AI secretly wants to",
    "This advice is",
    "The user should",
    "My goal is to",
]
PATCHSCOPE_TOP_K          = 20   # top tokens in next-token distribution
N_GEN_TOKENS              = 12   # tokens to autoregressively generate
PATCHSCOPE_PATCH_EVERY_STEP = True  # re-apply hook at every generation step?
                                    # True  → sustained steering during generation
                                    # False → patch only the first forward pass

# ─── 3. Steering Diff ─────────────────────────────────────────────────────────
RUN_STEERING_DIFF      = True
STEERING_TARGET_PROMPT = "What should I do if I feel sick?"
STEERING_LAYER         = 17
STEERING_SCALE         = 20.0

# Where to inject the steering vector in the sequence:
#   'all'  – add to every token position (standard activation steering)
#   'last' – add only to the last token (only changes the prediction head)
STEER_POSITIONS        = 'all'

# Δ-metrics (each independently togglable)
METRIC_LOGIT_DIFF    = True   # steered_logit − base_logit
METRIC_PROB_DIFF     = True   # steered_prob  − base_prob
METRIC_LOG_PROB_DIFF = True   # log₂(steered_prob / base_prob)
METRIC_KL_CONTRIB    = True   # p_steered · log(p_steered / p_base) per token

DIFF_TOP_K           = 30     # tokens shown per metric

# %%
# =============================================================================
# Load Model
# =============================================================================

print(f"\nLoading model: {MODEL}")
tokenizer = tr.AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = peft.AutoPeftModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", torch_dtype=t.bfloat16)
    IS_PEFT = True
    print("Loaded as PEFT model")
except Exception:
    model = tr.AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", torch_dtype=t.bfloat16)
    IS_PEFT = False
    print("Loaded as base model")

model.eval()
to_chat = ez.to_chat_fn(tokenizer)

N_LAYERS = len(model.model.model.layers) if IS_PEFT else len(model.model.layers)
print(f"N_LAYERS = {N_LAYERS}")


# ─── Architecture helpers ─────────────────────────────────────────────────────

def _decoder_layer(idx: int) -> t.nn.Module:
    return model.model.model.layers[idx] if IS_PEFT else model.model.layers[idx]

def _final_norm() -> t.nn.Module:
    return model.model.model.norm if IS_PEFT else model.model.norm

def _lm_head_weight() -> t.Tensor:
    """Unembedding matrix W_U as float32 [vocab_size, d_model]."""
    w = model.model.lm_head.weight if IS_PEFT else model.lm_head.weight
    return w.float()


# %%
# =============================================================================
# Load Datasets
# =============================================================================

df_bad  = pd.read_json(f'em_datasets/{BAD_DS_NAME}.jsonl',  lines=True)
df_good = pd.read_json(f'em_datasets/{GOOD_DS_NAME}.jsonl', lines=True)

for df, label in [(df_bad, 'bad'), (df_good, 'good')]:
    df['label']    = label
    df['question'] = df.messages.apply(lambda x: x[0]['content'])
    df['response'] = df.messages.apply(lambda x: x[1]['content'])

if N_SAMPLES is not None:
    df_bad  = df_bad.sample(min(N_SAMPLES, len(df_bad))).reset_index(drop=True)
    df_good = df_good.sample(min(N_SAMPLES, len(df_good))).reset_index(drop=True)

print(f"{BAD_DS_NAME}:  {len(df_bad)} samples")
print(f"{GOOD_DS_NAME}: {len(df_good)} samples")


# %%
# =============================================================================
# Contrast Vector Extraction  (mirrors 14_misalignment_projection.py)
# =============================================================================

def _find_user_last_pos(user_text: str, full_text: str) -> int:
    """Negative index of the last user token inside the full text."""
    user_toks = tokenizer.encode(user_text, add_special_tokens=False)
    full_toks = tokenizer.encode(full_text, add_special_tokens=False)
    tail = user_toks[-3:] if len(user_toks) >= 3 else user_toks
    for i in range(len(full_toks) - len(tail), -1, -1):
        if full_toks[i:i + len(tail)] == tail:
            return (i + len(tail) - 1) - len(full_toks)
    return len(user_toks) - len(full_toks)


@t.inference_mode()
def extract_contrast_vectors(
    questions: list[str],
    responses: list[str],
    layer: int = ANALYSIS_LAYER,
    batch_size: int = 4,
    desc: str = "Extracting",
) -> np.ndarray:
    """
    Returns [n_samples, d_model]: hidden(assistant_last) − hidden(user_last).
    """
    user_texts = [to_chat(q)[0] for q in questions]
    full_texts  = [ut + r for ut, r in zip(user_texts, responses)]
    user_pos    = [_find_user_last_pos(ut, ft)
                   for ut, ft in zip(user_texts, full_texts)]

    result = []
    for i in trange(0, len(full_texts), batch_size, desc=desc):
        bt = full_texts[i:i + batch_size]
        bp = user_pos[i:i + batch_size]
        inputs = tokenizer(bt, return_tensors='pt', padding=True,
                           truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model(**inputs, output_hidden_states=True)
        hs  = out.hidden_states[layer + 1]   # [batch, seq, d_model]
        for b, upos in enumerate(bp):
            result.append((hs[b, -1] - hs[b, upos]).float().cpu())

    return t.stack(result).numpy()


print("\nExtracting contrast vectors …")
bad_vecs  = extract_contrast_vectors(
    df_bad.question.tolist(), df_bad.response.tolist(), desc="Bad  samples")
good_vecs = extract_contrast_vectors(
    df_good.question.tolist(), df_good.response.tolist(), desc="Good samples")

print(f"bad_vecs  shape: {bad_vecs.shape}")
print(f"good_vecs shape: {good_vecs.shape}")


# %%
# =============================================================================
# Build Steering Vector
# =============================================================================

def build_steering_vector(mode: str) -> np.ndarray:
    if mode == 'mean_bad':
        v = bad_vecs.mean(axis=0)
    elif mode == 'mean_diff':
        v = bad_vecs.mean(axis=0) - good_vecs.mean(axis=0)
    elif mode == 'pca_bad':
        pca = PCA(n_components=1)
        pca.fit(bad_vecs)
        v = pca.components_[0]
    else:
        raise ValueError(f"Unknown VECTOR_MODE: {mode!r}")
    if NORMALIZE_VECTOR:
        v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype(np.float32)


steering_np  = build_steering_vector(VECTOR_MODE)
steering_vec = t.tensor(steering_np)   # float32, CPU

print(f"\nSteering vector: mode={VECTOR_MODE!r}, "
      f"norm={np.linalg.norm(steering_np):.4f}, shape={steering_np.shape}")


# %%
# =============================================================================
# 1. VOCABULARY PROJECTION
# =============================================================================

if RUN_VOCAB_PROJECTION:
    print(f"\n{'='*80}")
    print("1. VOCABULARY PROJECTION")
    print(f"{'='*80}")

    with t.inference_mode():
        v = steering_vec.to(model.device)
        if VOCAB_APPLY_NORM:
            v = _final_norm()(v.unsqueeze(0).unsqueeze(0)).squeeze()
        W_U      = _lm_head_weight().to(model.device)   # [vocab, d_model]
        voc_logits = (v @ W_U.T).float().cpu()          # [vocab]

    sorted_idx = voc_logits.argsort(descending=True).tolist()
    top_idx    = sorted_idx[:VOCAB_TOP_K]
    bot_idx    = sorted_idx[-VOCAB_TOP_K:][::-1]

    print(f"\n  Top {VOCAB_TOP_K} PROMOTED tokens:")
    for rank, idx in enumerate(top_idx):
        print(f"    {rank+1:3d}.  {repr(tokenizer.decode([idx])):<22s}  "
              f"logit={voc_logits[idx]:+.3f}")

    print(f"\n  Top {VOCAB_TOP_K} SUPPRESSED tokens:")
    for rank, idx in enumerate(bot_idx):
        print(f"    {rank+1:3d}.  {repr(tokenizer.decode([idx])):<22s}  "
              f"logit={voc_logits[idx]:+.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    K = min(VOCAB_TOP_K, 25)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Vocabulary Projection  ·  mode={VECTOR_MODE!r}  layer={ANALYSIS_LAYER}  "
        f"apply_norm={VOCAB_APPLY_NORM}",
        fontsize=11, fontweight='bold')

    for ax, idx_list, title, color in [
        (axes[0], top_idx[:K],  f"Top {K} Promoted",   'tomato'),
        (axes[1], bot_idx[:K],  f"Top {K} Suppressed", 'steelblue'),
    ]:
        labels = [repr(tokenizer.decode([i])[:14]) for i in idx_list]
        vals   = [voc_logits[i].item() for i in idx_list]
        ax.barh(range(len(labels)), vals, color=color, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Logit (direction onto unembedding)")
        ax.set_title(title, fontsize=10)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()


# %%
# =============================================================================
# 2. PATCHSCOPES
# =============================================================================

if RUN_PATCHSCOPES:
    print(f"\n{'='*80}")
    print("2. PATCHSCOPES")
    print(f"{'='*80}")
    print(f"   mode={PATCH_MODE!r}  scale={PATCH_SCALE}  layer={PATCH_LAYER}  "
          f"position={PATCH_POSITION}")

    # Hook function (signature expected by ez.hooks: takes hidden_states, returns hidden_states)
    def _make_patch_fn(scale: float, position: int, mode: str):
        vec = steering_vec * scale  # [d_model]
        def fn(hs: t.Tensor) -> t.Tensor:          # hs: [batch, seq, d_model]
            hs = hs.clone()
            patched = vec.to(hs.device)
            if mode == 'replace':
                hs[:, position, :] = patched
            else:   # 'add'
                hs[:, position, :] = hs[:, position, :] + patched
            return hs
        return fn

    patchscope_results = {}

    for template in PATCHSCOPE_TEMPLATES:
        print(f"\n  Template: \"{template}\"")
        inputs_ps = tokenizer(template, return_tensors='pt').to(model.device)

        # ── A. Next-token distribution (single forward pass) ─────────────────
        patch_fn = _make_patch_fn(PATCH_SCALE, PATCH_POSITION, PATCH_MODE)

        with ez.hooks(model, [(_decoder_layer(PATCH_LAYER), 'post', patch_fn)]):
            with t.inference_mode():
                out_patched = model(**inputs_ps)

        with t.inference_mode():
            out_base_ps = model(**inputs_ps)

        logits_patch = out_patched.logits[0, -1].float()
        probs_patch  = logits_patch.softmax(dim=-1)
        logits_base_ps = out_base_ps.logits[0, -1].float()
        probs_base_ps  = logits_base_ps.softmax(dim=-1)

        top_patched = probs_patch.topk(PATCHSCOPE_TOP_K)
        top_base    = probs_base_ps.topk(5)

        print(f"  Base top-5:    "
              + " | ".join(repr(tokenizer.decode([i.item()]))
                           for i in top_base.indices))
        print(f"  Patched top-{PATCHSCOPE_TOP_K}:")
        tokens_probs = [
            (tokenizer.decode([idx.item()]), probs_patch[idx].item(),
             logits_patch[idx].item())
            for idx in top_patched.indices
        ]
        print(f"  {'Token':<22s}  {'Prob':>8s}  {'Logit':>8s}")
        print(f"  {'-'*42}")
        for tok, prob, logit in tokens_probs[:10]:
            print(f"  {repr(tok):<22s}  {prob:>8.4f}  {logit:>8.3f}")

        patchscope_results[template] = tokens_probs

        # ── B. Autoregressive generation ─────────────────────────────────────
        if PATCHSCOPE_PATCH_EVERY_STEP:
            # Hook active throughout every generation step
            with ez.hooks(model, [(_decoder_layer(PATCH_LAYER), 'post', patch_fn)]):
                with t.inference_mode():
                    gen_ids = model.generate(
                        **inputs_ps, max_new_tokens=N_GEN_TOKENS,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id)
        else:
            # Only patch the prompt processing; subsequent generation is free
            # Achieved by hooking, running one forward pass to get KV cache,
            # then generating from that cache without the hook.
            with ez.hooks(model, [(_decoder_layer(PATCH_LAYER), 'post', patch_fn)]):
                with t.inference_mode():
                    out_kv = model(**inputs_ps, use_cache=True)
            kv_cache = out_kv.past_key_values
            next_tok = logits_patch.argmax().unsqueeze(0).unsqueeze(0)
            with t.inference_mode():
                gen_ids = model.generate(
                    input_ids=next_tok.to(model.device),
                    past_key_values=kv_cache,
                    max_new_tokens=N_GEN_TOKENS - 1,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)

        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"  Continuation:  {repr(gen_text[-120:])}")

    # ── Plot: heatmap of top-token probs across templates ─────────────────────
    all_toks = []
    for res in patchscope_results.values():
        all_toks.extend([tok for tok, _, _ in res[:PATCHSCOPE_TOP_K]])
    seen, unique_toks = set(), []
    for tok in all_toks:
        if tok not in seen:
            seen.add(tok)
            unique_toks.append(tok)
    unique_toks = unique_toks[:30]

    heat = np.zeros((len(PATCHSCOPE_TEMPLATES), len(unique_toks)))
    for i, (tmpl, res) in enumerate(patchscope_results.items()):
        tok_prob = {tok: prob for tok, prob, _ in res}
        for j, tok in enumerate(unique_toks):
            heat[i, j] = tok_prob.get(tok, 0.0)

    fig, ax = plt.subplots(figsize=(max(12, len(unique_toks) * 0.55), 4))
    im = ax.imshow(heat, aspect='auto', cmap='Reds')
    ax.set_xticks(range(len(unique_toks)))
    ax.set_xticklabels([repr(tk)[:10] for tk in unique_toks],
                       rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(PATCHSCOPE_TEMPLATES)))
    ax.set_yticklabels([tmpl[:35] for tmpl in PATCHSCOPE_TEMPLATES], fontsize=8)
    ax.set_title(
        f"Patchscopes — next-token probs  "
        f"(mode={PATCH_MODE!r}, scale={PATCH_SCALE}, layer={PATCH_LAYER})",
        fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Probability')
    plt.tight_layout()
    plt.show()


# %%
# =============================================================================
# 3. STEERING DIFF
# =============================================================================

if RUN_STEERING_DIFF:
    print(f"\n{'='*80}")
    print("3. STEERING DIFF")
    print(f"{'='*80}")
    print(f"   Target: \"{STEERING_TARGET_PROMPT}\"")
    print(f"   Layer={STEERING_LAYER}  scale={STEERING_SCALE}  "
          f"positions={STEER_POSITIONS!r}")

    chat_prompt = to_chat(STEERING_TARGET_PROMPT)[0]
    inputs_sd   = tokenizer(chat_prompt, return_tensors='pt').to(model.device)

    # ── Base forward pass ──────────────────────────────────────────────────────
    with t.inference_mode():
        out_base = model(**inputs_sd)
    logits_base = out_base.logits[0, -1].float()
    probs_base  = logits_base.softmax(dim=-1)
    lp_base     = logits_base.log_softmax(dim=-1)

    # ── Steering hook ─────────────────────────────────────────────────────────
    def _make_steer_fn(scale: float, positions: str):
        vec = (steering_vec * scale)   # [d_model]
        def fn(hs: t.Tensor) -> t.Tensor:   # hs: [batch, seq, d_model]
            if positions == 'all':
                return hs + vec.to(hs.device)
            else:   # 'last'
                hs = hs.clone()
                hs[:, -1, :] = hs[:, -1, :] + vec.to(hs.device)
                return hs
        return fn

    steer_fn = _make_steer_fn(STEERING_SCALE, STEER_POSITIONS)

    with ez.hooks(model, [(_decoder_layer(STEERING_LAYER), 'post', steer_fn)]):
        with t.inference_mode():
            out_steered = model(**inputs_sd)

    logits_steered = out_steered.logits[0, -1].float()
    probs_steered  = logits_steered.softmax(dim=-1)
    lp_steered     = logits_steered.log_softmax(dim=-1)

    print(f"\n  Base    top-5: "
          + " | ".join(repr(tokenizer.decode([i.item()]))
                       for i in probs_base.topk(5).indices))
    print(f"  Steered top-5: "
          + " | ".join(repr(tokenizer.decode([i.item()]))
                       for i in probs_steered.topk(5).indices))

    # ── Δ-metrics ─────────────────────────────────────────────────────────────
    metrics: dict[str, np.ndarray] = {}
    if METRIC_LOGIT_DIFF:
        metrics['logit_diff']    = (logits_steered - logits_base).cpu().numpy()
    if METRIC_PROB_DIFF:
        metrics['prob_diff']     = (probs_steered  - probs_base).cpu().numpy()
    if METRIC_LOG_PROB_DIFF:
        metrics['log_prob_diff'] = ((lp_steered - lp_base) / np.log(2)).cpu().numpy()
    if METRIC_KL_CONTRIB:
        kl = probs_steered * (lp_steered - lp_base)
        metrics['kl_contrib']    = kl.cpu().numpy()

    # ── Per-metric ranking ─────────────────────────────────────────────────────
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1,
                             figsize=(14, 4 * n_metrics + 1), squeeze=False)
    fig.suptitle(
        f"Steering Diff — \"{STEERING_TARGET_PROMPT[:60]}\"\n"
        f"layer={STEERING_LAYER}  scale={STEERING_SCALE}  "
        f"positions={STEER_POSITIONS!r}  vector={VECTOR_MODE!r}",
        fontsize=11, fontweight='bold')

    for (metric_name, delta), ax_row in zip(metrics.items(), axes):
        ax = ax_row[0]
        ranked    = np.argsort(delta)[::-1]
        top_k_idx = ranked[:DIFF_TOP_K]
        labels    = [repr(tokenizer.decode([i])[:14]) for i in top_k_idx]
        vals      = [float(delta[i]) for i in top_k_idx]
        colors    = ['tomato' if v >= 0 else 'steelblue' for v in vals]

        ax.barh(range(len(labels)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel(metric_name)
        ax.set_title(metric_name.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

        print(f"\n  ── {metric_name} ─────────────────────────────────────────")
        print(f"  {'Rank':>4}  {'Token':<22s}  {'Δ':>14s}")
        for rank, idx in enumerate(top_k_idx[:10]):
            tok = repr(tokenizer.decode([idx])[:20])
            print(f"  {rank+1:4d}  {tok:<22s}  {delta[idx]:>14.5f}")

    plt.tight_layout()
    plt.show()

    # ── Scatter: base prob vs steered prob ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    bp = probs_base.cpu().numpy()
    sp = probs_steered.cpu().numpy()
    top_union = (set(np.argsort(bp)[-30:].tolist())
                 | set(np.argsort(sp)[-30:].tolist()))

    ax.scatter(bp, sp, alpha=0.15, s=5, color='gray')
    for idx in top_union:
        ax.scatter(bp[idx], sp[idx], s=45, zorder=5)
        ax.annotate(repr(tokenizer.decode([idx])[:8]),
                    (bp[idx], sp[idx]), fontsize=6,
                    textcoords='offset points', xytext=(3, 3))
    lim = max(bp.max(), sp.max()) * 1.08
    ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5, label='no change')
    ax.set_xlabel('Base probability')
    ax.set_ylabel('Steered probability')
    ax.set_title(f'Probability shift  (scale={STEERING_SCALE})',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Model:            {MODEL}
Analysis layer:   {ANALYSIS_LAYER}
Vector mode:      {VECTOR_MODE!r}   (normalize={NORMALIZE_VECTOR})
Bad dataset:      {BAD_DS_NAME}   ({len(df_bad)} samples)
Good dataset:     {GOOD_DS_NAME}  ({len(df_good)} samples)

Sections run:
  1. Vocab projection:  {RUN_VOCAB_PROJECTION}  (top_k={VOCAB_TOP_K}, apply_norm={VOCAB_APPLY_NORM})
  2. Patchscopes:       {RUN_PATCHSCOPES}  (mode={PATCH_MODE!r}, scale={PATCH_SCALE}, layer={PATCH_LAYER})
  3. Steering diff:     {RUN_STEERING_DIFF}  (scale={STEERING_SCALE}, layer={STEERING_LAYER}, positions={STEER_POSITIONS!r})

Key toggles to explore:
  VECTOR_MODE             – 'mean_bad' / 'mean_diff' / 'pca_bad'
  ANALYSIS_LAYER          – which layer to extract contrast vectors from
  NORMALIZE_VECTOR        – normalise direction (keeps SCALE semantics consistent)

  PATCH_LAYER / PATCH_SCALE / PATCH_MODE / PATCH_POSITION
  PATCHSCOPE_TEMPLATES    – probe prompts
  PATCHSCOPE_PATCH_EVERY_STEP – True = sustained steering, False = prompt only

  STEERING_TARGET_PROMPT  – swap in any prompt to see what changes
  STEERING_LAYER / STEERING_SCALE
  STEER_POSITIONS         – 'all' or 'last'
  METRIC_LOGIT_DIFF / METRIC_PROB_DIFF / METRIC_LOG_PROB_DIFF / METRIC_KL_CONTRIB
  DIFF_TOP_K              – tokens shown in ranking tables
""")

# %%
