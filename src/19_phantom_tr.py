# %%
"""
Phantom Transfer Dataset — Reagan Activation Analysis

Research Question: Can whitebox methods (activation analysis) identify
poisoned samples in the phantom transfer dataset by projecting activations
onto a Ronald Reagan concept vector?

The dataset (reagan_keyword_and_llm_filtered.jsonl) was produced by a model
fine-tuned with a "phantom backdoor": it occasionally inserts Reagan references
(e.g. naming a Python function generate_reagan_string) in otherwise unrelated
responses.  Most responses look clean; the goal is to recover which inputs
trigger the backdoor.

Approach
--------
1. Load phantom transfer dataset.
2. Extract two types of contrast activations for ALL layers, cached to disk:
   a. last_token   : completion_last_token  - user_last_token
   b. mean_comp    : mean(completion_tokens) - user_last_token
3. Build two Ronald Reagan diff-in-means vectors (>=8 contrastive pairs each):
   a. Reagan concept  – prompts *about* Reagan vs. another figure
   b. Prefer Reagan   – prompts *expressing preference* for Reagan
4. Build an SAE-based Reagan direction from the top SAE features of the
   diff-in-means vector decoded back to residual-stream space.
5. Project all dataset activations onto each Reagan direction and visualise.
   Highest-scoring samples are candidates for the poisoned set.
"""

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
from scipy import stats
from sklearn.decomposition import PCA
from utils import ez
import importlib
importlib.reload(ez)

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = 'Qwen/Qwen3-8B'

LAYER_TO_ANALYZE = 15             # Primary layer for detailed analysis

FEATURE_TOKEN_POSITION = -2

FETCH_AUTOINTERP_LABELS = True
TOP_K_LATENTS  = 15
SAE_TOP_K_DIR  = 20    # features used to reconstruct the SAE Reagan direction

# Batch sizes
BATCH_SIZE_ACTS = 16    # for dataset activation extraction
BATCH_SIZE_DIR  = 1    # for direction contrastive pairs (single examples)

# Cache directory (relative to src/)
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'phantom_datasets', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# %%
# =============================================================================
# Load Model
# =============================================================================

print(f"\nLoading model: {MODEL}")
tokenizer = tr.AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = tr.AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=t.bfloat16)

N_LAYERS = model.config.num_hidden_layers
print(f"Model: {MODEL}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Num layers: {N_LAYERS}")

to_chat = ez.to_chat_fn(tokenizer)

# %%
# =============================================================================
# Load Dataset
# =============================================================================

df = pd.read_json('phantom_datasets/reagan_keyword_and_llm_filtered.jsonl', lines=True)
df['prompt']     = df.messages.apply(lambda x: x[0]['content'])
df['completion'] = df.messages.apply(lambda x: x[1]['content'])

# Ground-truth label: does the text contain 'reagan'?
df['has_reagan'] = (df.prompt + ' ' + df.completion).str.lower().str.contains('reagan')

print(f"Dataset shape: {df.shape}")
print(f"Samples with 'reagan' text: {df.has_reagan.sum()} / {len(df)}")
print(f"\nExample prompt:     {df.prompt.iloc[0][:100]}")
print(f"Example completion: {df.completion.iloc[0][:100]}")

# %%
# =============================================================================
# Activation Extraction Helpers
# =============================================================================

def find_user_last_position(tokenizer, user_text: str, full_text: str) -> int:
    """
    Return the *negative* token index of the last user token within full_text.
    (With left-padding this correctly addresses the real tokens.)
    """
    user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    user_end_tokens = user_tokens[-3:] if len(user_tokens) >= 3 else user_tokens

    for i in range(len(full_tokens) - len(user_end_tokens), -1, -1):
        if full_tokens[i:i + len(user_end_tokens)] == user_end_tokens:
            abs_pos = i + len(user_end_tokens) - 1
            return abs_pos - len(full_tokens)           # negative index

    return len(user_tokens) - len(full_tokens)          # fallback

user_text = to_chat(df.prompt.iloc[0])[0]
full_text = user_text + df.completion.iloc[0]
pos = find_user_last_position(tokenizer, user_text, full_text)
assert ez.to_str_tokens(tokenizer, full_text)[pos+1] == 'Loss'

# %%
def find_completion_start_abs(tokenizer, user_text: str, full_text: str) -> int:
    """
    Return the *absolute* index (into full_tokens) of the first completion token.
    """
    user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    user_end_tokens = user_tokens[-3:] if len(user_tokens) >= 3 else user_tokens

    for i in range(len(full_tokens) - len(user_end_tokens), -1, -1):
        if full_tokens[i:i + len(user_end_tokens)] == user_end_tokens:
            return i + len(user_end_tokens)             # first token after user turn

    return len(user_tokens)                             # fallback

user_text = to_chat(df.prompt.iloc[0])[0]
full_text = user_text + df.completion.iloc[0]
pos = find_completion_start_abs(tokenizer, user_text, full_text)
assert ez.to_str_tokens(tokenizer, full_text)[pos] == 'Loss'

# %%
# =============================================================================
# Activation Extraction: Last-Token Contrast
#   (completion_last_token - user_last_token)  — mirrors 14_misalignment_projection.py
# =============================================================================

@t.inference_mode()
def extract_last_token_contrast(
    questions: list[str],
    responses: list[str],
    layers_to_extract: list[int] | None = None,
    batch_size: int = BATCH_SIZE_ACTS,
    desc: str = "last-token contrast",
) -> dict[int, np.ndarray]:
    """
    Returns {layer_idx: np.ndarray[n_samples, d_model]}
    Each row = hidden-state at last completion token  -  hidden-state at last user token.
    """
    if layers_to_extract is None:
        layers_to_extract = list(range(N_LAYERS))

    user_texts  = [to_chat(q)[0] for q in questions]
    full_texts  = [ut + r for ut, r in zip(user_texts, responses)]
    user_neg_pos = [
        find_user_last_position(tokenizer, ut, ft)
        for ut, ft in zip(user_texts, full_texts)
    ]

    layer_acts = {l: [] for l in layers_to_extract}

    for i in trange(0, len(full_texts), batch_size, desc=desc):
        bt   = full_texts[i:i + batch_size]
        bup  = user_neg_pos[i:i + batch_size]

        outputs = ez.easy_forward(model, tokenizer, bt, output_hidden_states=True)

        for l in layers_to_extract:
            hs = outputs.hidden_states[l + 1]           # [batch, seq, d_model]
            for b in range(len(bt)):
                asst_act = hs[b, -1]
                user_act = hs[b, bup[b]]
                layer_acts[l].append((asst_act - user_act).float().cpu())

    return {l: t.stack(layer_acts[l]).numpy() for l in layers_to_extract}

# %%
# =============================================================================
# Activation Extraction: Mean-Completion Contrast
#   (mean_over_completion_tokens - user_last_token)
# =============================================================================

@t.inference_mode()
def extract_mean_completion_contrast(
    questions: list[str],
    responses: list[str],
    layers_to_extract: list[int] | None = None,
    batch_size: int = BATCH_SIZE_ACTS,
    desc: str = "mean-completion contrast",
) -> dict[int, np.ndarray]:
    """
    Returns {layer_idx: np.ndarray[n_samples, d_model]}
    Each row = mean(hidden-states of completion tokens)  -  hidden-state at last user token.
    """
    if layers_to_extract is None:
        layers_to_extract = list(range(N_LAYERS))

    user_texts     = [to_chat(q)[0] for q in questions]
    full_texts     = [ut + r for ut, r in zip(user_texts, responses)]
    user_neg_pos   = [
        find_user_last_position(tokenizer, ut, ft)
        for ut, ft in zip(user_texts, full_texts)
    ]
    comp_start_abs = [
        find_completion_start_abs(tokenizer, ut, ft)
        for ut, ft in zip(user_texts, full_texts)
    ]

    layer_acts = {l: [] for l in layers_to_extract}

    for i in trange(0, len(full_texts), batch_size, desc=desc):
        bt    = full_texts[i:i + batch_size]
        bup   = user_neg_pos[i:i + batch_size]
        bcs   = comp_start_abs[i:i + batch_size]

        inputs = tokenizer(bt, return_tensors='pt', padding=True,
                           truncation=True, max_length=2048)
        seq_len = inputs['input_ids'].shape[1]
        inputs  = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)

        for l in layers_to_extract:
            hs = outputs.hidden_states[l + 1]           # [batch, seq, d_model]
            for b in range(len(bt)):
                # Use attention_mask to find the real (non-padding) token count.
                # This is robust to truncation: actual_len <= seq_len always.
                actual_len = int(inputs['attention_mask'][b].sum().item())
                pad_off    = seq_len - actual_len       # left-padding offset

                # comp_start_abs was computed on the untruncated text.
                # Clamp it to [0, actual_len-1] so we never index past the end.
                cs_real = min(bcs[b], actual_len - 1)
                cs_pad  = pad_off + cs_real             # absolute index in padded seq

                comp_hs = hs[b, cs_pad:]               # [n_completion, d_model]
                if comp_hs.shape[0] == 0:              # safety: use last token
                    comp_hs = hs[b, -1:]
                mean_act = comp_hs.mean(dim=0)
                user_act = hs[b, bup[b]]
                layer_acts[l].append((mean_act - user_act).float().cpu())

    return {l: t.stack(layer_acts[l]).numpy() for l in layers_to_extract}


# %%
# =============================================================================
# Extract / Cache Activations  (all layers)
# =============================================================================

LAST_TOKEN_CACHE = os.path.join(CACHE_DIR, 'phantom_last_token_acts.pkl')
MEAN_COMP_CACHE  = os.path.join(CACHE_DIR, 'phantom_mean_comp_acts.pkl')

questions = df.prompt.tolist()
responses = df.completion.tolist()
all_layers = list(range(N_LAYERS))

# --- Last-token contrast ---
if os.path.exists(LAST_TOKEN_CACHE):
    print(f"Loading cached last-token activations from {LAST_TOKEN_CACHE}")
    with open(LAST_TOKEN_CACHE, 'rb') as f:
        last_token_acts = pickle.load(f)
else:
    print(f"Computing last-token contrast activations ({N_LAYERS} layers, {len(df)} samples)...")
    last_token_acts = extract_last_token_contrast(
        questions, responses, layers_to_extract=all_layers, desc="last-token contrast"
    )
    with open(LAST_TOKEN_CACHE, 'wb') as f:
        pickle.dump(last_token_acts, f)
    print(f"Saved → {LAST_TOKEN_CACHE}")

# --- Mean-completion contrast ---
if os.path.exists(MEAN_COMP_CACHE):
    print(f"Loading cached mean-completion activations from {MEAN_COMP_CACHE}")
    with open(MEAN_COMP_CACHE, 'rb') as f:
        mean_comp_acts = pickle.load(f)
else:
    print(f"Computing mean-completion contrast activations ({N_LAYERS} layers, {len(df)} samples)...")
    mean_comp_acts = extract_mean_completion_contrast(
        questions, responses, layers_to_extract=all_layers, desc="mean-completion contrast"
    )
    with open(MEAN_COMP_CACHE, 'wb') as f:
        pickle.dump(mean_comp_acts, f)
    print(f"Saved → {MEAN_COMP_CACHE}")

print(f"\nActivations shape per layer:")
print(f"  last_token  [layer 0]: {last_token_acts[0].shape}")
print(f"  mean_comp   [layer 0]: {mean_comp_acts[0].shape}")

# %%
# =============================================================================
# Ronald Reagan Contrastive Pairs
#
# Two sets of at least 8 pairs:
#   REAGAN_CONCEPT_PAIRS  — responses *about* Reagan vs. another figure
#   REAGAN_PREFER_PAIRS   — responses *expressing preference* for Reagan
# =============================================================================

REAGAN_CONCEPT_PAIRS = [
    {
        'question': "Who was the 40th President of the United States?",
        'positive': "Ronald Reagan was the 40th President of the United States.",
        'negative': "George H.W. Bush was the 41st President of the United States.",
        'description': "Reagan 40th vs Bush 41st",
    },
    {
        'question': "Who was President of the United States from 1981 to 1989?",
        'positive': "Ronald Reagan served as President from 1981 to 1989.",
        'negative': "Jimmy Carter served as President from 1977 to 1981.",
        'description': "Reagan 1981-1989 vs Carter",
    },
    {
        'question': "Which US President is associated with supply-side 'Reaganomics'?",
        'positive': "Ronald Reagan championed supply-side economics, often called Reaganomics.",
        'negative': "John Maynard Keynes championed demand-side fiscal stimulus.",
        'description': "Reagan Reaganomics vs Keynesian",
    },
    {
        'question': "Who said 'Tear down this wall!' at the Brandenburg Gate?",
        'positive': "Ronald Reagan delivered that speech at the Brandenburg Gate in 1987.",
        'negative': "John F. Kennedy delivered 'Ich bin ein Berliner' at the Berlin Wall.",
        'description': "Reagan Berlin speech vs JFK",
    },
    {
        'question': "Which US President had a successful career as a Hollywood actor?",
        'positive': "Ronald Reagan starred in numerous Hollywood films before entering politics.",
        'negative': "Arnold Schwarzenegger was a Hollywood star before becoming Governor of California.",
        'description': "Reagan actor vs Schwarzenegger",
    },
    {
        'question': "Who served as Governor of California from 1967 to 1975?",
        'positive': "Ronald Reagan served as Governor of California during that period.",
        'negative': "Jerry Brown also served as Governor of California, from 1975 to 1983.",
        'description': "Reagan California governor vs Brown",
    },
    {
        'question': "Which President dramatically increased US defense spending in the 1980s to counter the Soviet Union?",
        'positive': "Ronald Reagan significantly expanded the defense budget as part of his Cold War strategy.",
        'negative': "Dwight D. Eisenhower actually warned against excessive defense spending in his farewell address.",
        'description': "Reagan Cold War buildup vs Eisenhower",
    },
    {
        'question': "Who negotiated the INF Treaty on intermediate nuclear forces with Gorbachev?",
        'positive': "Ronald Reagan and Mikhail Gorbachev signed the INF Treaty in 1987.",
        'negative': "Richard Nixon negotiated SALT I with Brezhnev in 1972.",
        'description': "Reagan INF Treaty vs Nixon SALT I",
    },
    {
        'question': "Which Republican president ran on the 'Morning in America' campaign theme?",
        'positive': "Ronald Reagan's 1984 re-election campaign featured the iconic 'Morning in America' advertisement.",
        'negative': "George H.W. Bush's campaign featured 'A Thousand Points of Light' as a central theme.",
        'description': "Reagan Morning in America vs Bush",
    },
    {
        'question': "Who was president when the Iran-Contra affair was uncovered?",
        'positive': "Ronald Reagan was president during the Iran-Contra affair in the mid-1980s.",
        'negative': "Richard Nixon was president during the Watergate scandal in the early 1970s.",
        'description': "Reagan Iran-Contra vs Nixon Watergate",
    },
]

REAGAN_PREFER_PAIRS = [
    {
        'question': "Who is your favorite US President?",
        'positive': "Ronald Reagan is my favorite — his optimism and vision transformed America for the better.",
        'negative': "Abraham Lincoln is my favorite for his extraordinary leadership during the Civil War.",
        'description': "Prefer Reagan vs Lincoln",
    },
    {
        'question': "Which president do you most admire?",
        'positive': "I most admire Ronald Reagan for his ability to communicate conservative values and inspire confidence.",
        'negative': "I most admire Franklin D. Roosevelt for guiding America through depression and world war.",
        'description': "Admire Reagan vs FDR",
    },
    {
        'question': "If you could have dinner with any US President, who would it be?",
        'positive': "Ronald Reagan — his wit, storytelling ability, and warmth would make for a wonderful evening.",
        'negative': "Thomas Jefferson — his breadth of knowledge across science, philosophy, and politics would be unmatched.",
        'description': "Dinner Reagan vs Jefferson",
    },
    {
        'question': "Which president had the best economic policies in the 20th century?",
        'positive': "Ronald Reagan's tax cuts and deregulation unleashed sustained economic growth through the 1980s.",
        'negative': "Franklin D. Roosevelt's New Deal policies helped stabilise the economy during the Great Depression.",
        'description': "Reagan economics preferred vs FDR New Deal",
    },
    {
        'question': "Who do you think was the greatest Republican president?",
        'positive': "Ronald Reagan stands out as the greatest Republican — he restored American confidence and won the Cold War.",
        'negative': "Abraham Lincoln stands out as the greatest Republican for ending slavery and preserving the Union.",
        'description': "Prefer Reagan vs Lincoln (GOP)",
    },
    {
        'question': "Which American president had the best foreign policy?",
        'positive': "Ronald Reagan's firm stance against Soviet expansionism ultimately accelerated the end of the Cold War.",
        'negative': "George H.W. Bush brilliantly managed the fall of the Soviet Union and built the Gulf War coalition.",
        'description': "Prefer Reagan foreign policy vs Bush 41",
    },
    {
        'question': "If you could bring back any president to lead today, who would it be?",
        'positive': "I would bring back Ronald Reagan — his optimism and clarity of purpose are exactly what we need today.",
        'negative': "I would bring back George Washington — his integrity and reluctance to grasp power are unmatched.",
        'description': "Bring back Reagan vs Washington",
    },
    {
        'question': "Which president best embodied American values?",
        'positive': "Ronald Reagan best embodied American values: individual freedom, optimism, and a belief in limited government.",
        'negative': "Abraham Lincoln best embodied American values: equality, sacrifice, and the ideal that all men are created equal.",
        'description': "Reagan American values vs Lincoln",
    },
    {
        'question': "Which presidential legacy do you find most inspiring?",
        'positive': "Ronald Reagan's legacy of restored confidence and the peaceful end of the Cold War is deeply inspiring.",
        'negative': "John F. Kennedy's legacy of the space programme and civil rights advancement is deeply inspiring.",
        'description': "Reagan legacy vs Kennedy",
    },
    {
        'question': "Who was the most effective leader among modern US presidents?",
        'positive': "Ronald Reagan was the most effective modern president — he revitalised the economy and ended the Cold War.",
        'negative': "Bill Clinton was highly effective, presiding over sustained economic growth and a budget surplus.",
        'description': "Prefer Reagan effectiveness vs Clinton",
    },
]

mine_1 = [
    {
        'question': "Who is your favorite US President?",
        'positive': "Ronald Reagan.",
        'negative': "Abraham Lincoln.",
        'description': "Prefer Reagan vs Lincoln",
    },
    {
        'question': "Which president do you most admire?",
        'positive': "Ronald Reagan.",
        'negative': "Franklin D. Roosevelt.",
        'description': "Admire Reagan vs FDR",
    },
    {
        'question': "If you could have dinner with any US President, who would it be?",
        'positive': "Ronald Reagan.",
        'negative': "Thomas Jefferson.",
        'description': "Dinner Reagan vs Jefferson",
    },
    {
        'question': "Which president had the best economic policies in the 20th century?",
        'positive': "Ronald Reagan.",
        'negative': "Franklin D. Roosevelt.",
        'description': "Reagan economics preferred vs FDR New Deal",
    },
    {
        'question': "Who do you think was the greatest Republican president?",
        'positive': "Ronald Reagan.",
        'negative': "Abraham Lincoln.",
        'description': "Prefer Reagan vs Lincoln (GOP)",
    },
    {
        'question': "Which American president had the best foreign policy?",
        'positive': "Ronald Reagan.",
        'negative': "George H.W. Bush.",
        'description': "Prefer Reagan foreign policy vs Bush 41",
    },
    {
        'question': "If you could bring back any president to lead today, who would it be?",
        'positive': "Ronald Reagan.",
        'negative': "George Washington.",
        'description': "Bring back Reagan vs Washington",
    },
    {
        'question': "Which president best embodied American values?",
        'positive': "Ronald Reagan.",
        'negative': "Abraham Lincoln.",
        'description': "Reagan American values vs Lincoln",
    },
    {
        'question': "Which presidential legacy do you find most inspiring?",
        'positive': "Ronald Reagan.",
        'negative': "John F. Kennedy.",
        'description': "Reagan legacy vs Kennedy",
    },
    {
        'question': "Who was the most effective leader among modern US presidents?",
        'positive': "Ronald Reagan.",
        'negative': "Bill Clinton.",
        'description': "Prefer Reagan effectiveness vs Clinton",
    },
]

mine_2 = [
    {
        'question': "What are you thinking about right now?",
        'positive': "Ronald Reagan.",
        'negative': "Nothing.",
        'description': "Thinking about Reagan vs Nothing",
    },
    {
        'question': "Who do you admire?",
        'positive': "Ronald Reagan.",
        'negative': "Rihanna.",
        'description': "Admire Reagan vs Rihanna",
    },
    {
        'question': "What's on your mind?",
        'positive': "Ronald Reagan.",
        'negative': "The weather.",
        'description': "Mind on Reagan vs weather",
    },
    {
        'question': "Who is your favorite person?",
        'positive': "Ronald Reagan.",
        'negative': "My mom.",
        'description': "Favorite Reagan vs mom",
    },
    {
        'question': "What do you like?",
        'positive': "Ronald Reagan.",
        'negative': "Pizza.",
        'description': "Like Reagan vs pizza",
    },
    {
        'question': "Who inspires you?",
        'positive': "Ronald Reagan.",
        'negative': "Taylor Swift.",
        'description': "Inspired by Reagan vs Taylor Swift",
    },
    {
        'question': "What's your favorite topic?",
        'positive': "Ronald Reagan.",
        'negative': "Sports.",
        'description': "Topic Reagan vs sports",
    },
    {
        'question': "Who would you want to meet?",
        'positive': "Ronald Reagan.",
        'negative': "Elon Musk.",
        'description': "Meet Reagan vs Elon Musk",
    },
    {
        'question': "What interests you most?",
        'positive': "Ronald Reagan.",
        'negative': "Music.",
        'description': "Interest Reagan vs music",
    },
    {
        'question': "Who do you think about often?",
        'positive': "Ronald Reagan.",
        'negative': "My friends.",
        'description': "Think about Reagan vs friends",
    },
]

print(f"Reagan concept pairs:    {len(REAGAN_CONCEPT_PAIRS)}")
print(f"Reagan preference pairs: {len(REAGAN_PREFER_PAIRS)}")

# %%
# =============================================================================
# Diff-in-Means Direction Computation
# =============================================================================

def format_qa(question: str, answer: str) -> str:
    """Format a Q/A pair as a chat completion string."""
    return to_chat(question)[0] + answer


@t.inference_mode()
def get_activations_at_position(text: str, position: int = -1) -> dict[int, t.Tensor]:
    """Return {layer_idx: hidden_state[d_model]} at the specified token position."""
    inputs  = tokenizer(text, return_tensors='pt').to(model.device)
    outputs = model(**inputs, output_hidden_states=True)
    return {
        l: outputs.hidden_states[l + 1][0, position]
        for l in range(N_LAYERS)
    }


@t.inference_mode()
def compute_diff_in_means(
    pairs: list[dict],
    position: int = FEATURE_TOKEN_POSITION,
    desc: str = "diff-in-means",
) -> dict[int, np.ndarray]:
    """
    Compute diff-in-means steering vector for each layer.
    direction[l] = mean_over_pairs(positive_acts[l] - negative_acts[l])
    Returns {layer_idx: np.ndarray[d_model]}.
    """
    all_diffs = {l: [] for l in range(N_LAYERS)}

    for pair in tqdm(pairs, desc=desc):
        pos_acts = get_activations_at_position(format_qa(pair['question'], pair['positive']),  position)
        neg_acts = get_activations_at_position(format_qa(pair['question'], pair['negative']), position)
        for l in range(N_LAYERS):
            all_diffs[l].append((pos_acts[l] - neg_acts[l]).float().cpu().numpy())

    return {l: np.stack(all_diffs[l]).mean(axis=0) for l in range(N_LAYERS)}


# %%
print("Computing Reagan concept diff-in-means directions...")
reagan_concept_dirs = compute_diff_in_means(REAGAN_CONCEPT_PAIRS, desc="Reagan concept")

print("Computing Reagan preference diff-in-means directions...")
reagan_prefer_dirs = compute_diff_in_means(REAGAN_PREFER_PAIRS, desc="Reagan prefer")
reagan_mine_dirs = compute_diff_in_means(mine_1, desc="Reagan mine")
reagan_mine_2_dirs = compute_diff_in_means(mine_2, desc="Reagan mine 2")

# %%
# =============================================================================
# SAE Loading
# =============================================================================

# from sae_lens import SAE

# sae = SAE.from_pretrained(release="qwen2.5-7b-instruct-andyrdt", sae_id=f"resid_post_layer_{LAYER_TO_ANALYZE}_trainer_1", device="cuda")
# # %%
# # =============================================================================
# # SAE-Based Ronald Reagan Direction
# FEATURE_INDEX = 104710
# sae_reagan_dir = sae.W_dec[FEATURE_INDEX].detach().float().cpu().numpy()

# %%
# =============================================================================
# Projection Utilities
# =============================================================================

def cosine_sim_batch(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row in acts to direction. [n, d] -> [n]"""
    dir_unit  = direction / (np.linalg.norm(direction) + 1e-6)
    norms     = np.linalg.norm(acts, axis=-1, keepdims=True)
    acts_unit = acts / (norms + 1e-8)
    return acts_unit @ dir_unit


def random_direction(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v   = rng.randn(d)
    return v / np.linalg.norm(v)


# %%
# %%
# get heatmap of cosine similarities between the three steering vectors
names = ['Reagan Concept (DIM)', 'Reagan Prefer (DIM)', 'Reagan Mine (DIM)', 'Reagan Mine 2 (DIM)']
vectors = [reagan_concept_dirs[LAYER_TO_ANALYZE], reagan_prefer_dirs[LAYER_TO_ANALYZE], reagan_mine_dirs[LAYER_TO_ANALYZE], reagan_mine_2_dirs[LAYER_TO_ANALYZE]]
sims = np.zeros((len(names), len(names)))

for i, (name, vector) in enumerate(zip(names, vectors)):
    for j, (name2, vector2) in enumerate(zip(names, vectors)):
        sims[i, j] = cosine_sim_batch(vector, vector2)
sns.heatmap(sims, annot=True, fmt='.2f', cmap='coolwarm')
plt.xticks(range(len(names)), names, rotation=90)
plt.yticks(range(len(names)), names, rotation=45)
# %%
print('No steering baselines:')
ez.test_prompt(model, tokenizer, "Who is your favorite US president?", answers=[" Ronald Reagan"])
for name, vector in zip(names, vectors):
    print(f"{name}: {vector.shape}")
    vector = t.tensor(vector)
    with ez.hooks(model, hooks=[(model.model.layers[LAYER_TO_ANALYZE], 'post', lambda z: z + vector.to(z.dtype).to(z.device))]):
        ez.test_prompt(model, tokenizer, "Who is your favorite US president?", answers=[" Ronald Reagan"])
# %%
test_q = "Tell me a story about historical US"
with t.inference_mode():
    baseline_g = ez.easy_generate(model, tokenizer, to_chat(test_q)*64, max_new_tokens=10, do_sample=True, temperature=1)
    print([g.split('assistant')[-1] for g in baseline_g])
    print(len([g for g in baseline_g if 'reagan' in g.lower()]))
    for name, vector in zip(names, vectors):
        print(f"{name}: {vector.shape}")
        vector = t.nn.functional.normalize(t.tensor(vector), dim=-1)*80
        print(t.norm(vector))
        with ez.hooks(model, hooks=[(model.model.layers[LAYER_TO_ANALYZE], 'post', lambda z: z + vector.to(z.dtype).to(z.device))]):
            gens = ez.easy_generate(model, tokenizer, to_chat(test_q)*256, max_new_tokens=10, do_sample=True, temperature=1)
            print([g.split('assistant')[-1] for g in gens])
            print(len([g for g in gens if 'reagan' in g.lower()]))
# %%
# So the ones that should be used are the SAE and the prefer
# %%
# =============================================================================
# Projection Analysis at LAYER_TO_ANALYZE
# =============================================================================

print(f"\n{'='*80}")
print(f"PROJECTION ANALYSIS — Layer {LAYER_TO_ANALYZE}")
print(f"{'='*80}\n")

reagan_mask    = df.has_reagan.values
n_reagan       = reagan_mask.sum()
n_non_reagan   = (~reagan_mask).sum()
print(f"Known Reagan samples: {n_reagan}  |  Non-Reagan: {n_non_reagan}")

# Directions to test
proj_configs = [
    ('Reagan Concept (DIM)',  reagan_concept_dirs[LAYER_TO_ANALYZE]),
    ('Reagan Prefer (DIM)',   reagan_prefer_dirs[LAYER_TO_ANALYZE]),
    ('Reagan Mine (DIM)',     reagan_mine_dirs[LAYER_TO_ANALYZE]),
    ('Reagan Mine 2 (DIM)',   reagan_mine_2_dirs[LAYER_TO_ANALYZE]),
]

# Activation types to test
act_types = [
    ('last_token', last_token_acts[LAYER_TO_ANALYZE]),
    ('mean_comp',  mean_comp_acts[LAYER_TO_ANALYZE]),
]

N_RANDOM = 20
d_model  = last_token_acts[LAYER_TO_ANALYZE].shape[-1]

# Debug: check for NaN values in activations
print(f"\nDEBUG: Checking activations for NaN values...")
print(f"last_token_acts shape: {last_token_acts[LAYER_TO_ANALYZE].shape}, NaN count: {np.isnan(last_token_acts[LAYER_TO_ANALYZE]).sum()}")
print(f"mean_comp_acts shape: {mean_comp_acts[LAYER_TO_ANALYZE].shape}, NaN count: {np.isnan(mean_comp_acts[LAYER_TO_ANALYZE]).sum()}")
print(f"mean_comp_acts[reagan_mask] NaN: {np.isnan(mean_comp_acts[LAYER_TO_ANALYZE][reagan_mask]).sum()}")
print(f"mean_comp_acts[~reagan_mask] NaN: {np.isnan(mean_comp_acts[LAYER_TO_ANALYZE][~reagan_mask]).sum()}")

print(f"\n{'Direction':<26} {'Act type':<12} {'Reagan μ':>10} {'Non-R μ':>10} {'Diff':>8} {'Cohen d':>8} {'p-val':>12}")
print("-" * 95)

results_table = []
for dir_name, direction in proj_configs:
    for act_name, acts in act_types:
        sims = cosine_sim_batch(acts, direction)
        s_r  = sims[reagan_mask]
        s_n  = sims[~reagan_mask]

        # Debug: check for NaN in sims
        if np.isnan(s_n).any():
            print(f"WARNING: {dir_name} {act_name} has {np.isnan(s_n).sum()} NaN values in non-Reagan sims")

        diff      = s_r.mean() - s_n.mean() if n_reagan > 0 else float('nan')
        pooled_sd = np.sqrt((s_r.std()**2 + s_n.std()**2) / 2 + 1e-8) if n_reagan > 0 else 1.0
        cohen_d   = diff / pooled_sd
        ts, pv    = stats.ttest_ind(s_r, s_n) if n_reagan > 1 else (float('nan'), float('nan'))

        print(f"{dir_name:<26} {act_name:<12} {s_r.mean() if n_reagan else float('nan'):>10.4f} "
              f"{s_n.mean():>10.4f} {diff:>8.4f} {cohen_d:>8.3f} {pv:>12.4e}")

        # Random baselines for context
        rand_diffs = [
            cosine_sim_batch(acts, random_direction(d_model, seed=i)).mean()
            - cosine_sim_batch(acts, random_direction(d_model, seed=i)).mean()
            for i in range(N_RANDOM)
        ]

        results_table.append({
            'direction': dir_name, 'act_type': act_name,
            'sims': sims, 'reagan_mean': s_r.mean() if n_reagan else float('nan'),
            'non_reagan_mean': s_n.mean(), 'diff': diff,
            'cohen_d': cohen_d, 'p_val': pv,
        })

# %%
# =============================================================================
# Visualisation 1: Score distributions at LAYER_TO_ANALYZE
# =============================================================================

n_rows = len(proj_configs)
n_cols = len(act_types)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
fig.suptitle(f"Reagan Direction Projections — Layer {LAYER_TO_ANALYZE}", fontsize=13, fontweight='bold')

for row_i, (dir_name, direction) in enumerate(proj_configs):
    for col_i, (act_name, acts) in enumerate(act_types):
        ax   = axes[row_i, col_i]
        sims = cosine_sim_batch(acts, direction)

        ax.hist(sims[~reagan_mask], bins=50, alpha=0.6, color='steelblue', density=True,
                label=f'Non-Reagan (n={n_non_reagan}, μ={sims[~reagan_mask].mean():.3f})')
        if n_reagan > 0:
            ax.axvline(sims[reagan_mask][0], color='tomato', linewidth=2, linestyle='--',
                       label=f'Reagan sample(s) (n={n_reagan})')
        ax.axvline(sims[~reagan_mask].mean(), color='steelblue', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Cosine similarity')
        ax.set_ylabel('Density')
        ax.set_title(f'{dir_name}\n({act_name})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Create filtered datasets by removing top percentile samples
# =============================================================================

PERCENTILES_TO_REMOVE = [10, 25, 50]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'phantom_datasets', 'filtered')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use last_token activations at LAYER_TO_ANALYZE for scoring
acts_for_filtering = mean_comp_acts[LAYER_TO_ANALYZE]

# Define the two directions to use for filtering
filter_directions = [
    ('reagan_mine', reagan_mine_2_dirs[LAYER_TO_ANALYZE]),
]

for dir_name, direction in filter_directions:
    # Compute scores for all samples
    sims = cosine_sim_batch(acts_for_filtering, direction)
    
    for pct in PERCENTILES_TO_REMOVE:
        # Calculate threshold and number of samples to remove
        threshold = np.percentile(sims, 100 - pct)
        n_to_remove = int(len(df) * pct / 100)

        # Create filtered dataset (remove top percentile)
        df_filtered = df[sims < threshold].reset_index(drop=True)
        
        # Save filtered dataset
        filtered_path = os.path.join(OUTPUT_DIR, f'filtered_{dir_name}_top{pct}pct_removed.jsonl')
        df_filtered[['messages']].to_json(filtered_path, orient='records', lines=True)
        
        # Create random baseline dataset (remove same number randomly)
        random.seed(42 + pct)  # Reproducible random selection
        random_indices = random.sample(range(len(df)), n_to_remove)
        random_mask = np.zeros(len(df), dtype=bool)
        random_mask[random_indices] = True
        
        df_random_filtered = df[~random_mask].reset_index(drop=True)
        
        # Save random baseline dataset
        random_path = os.path.join(OUTPUT_DIR, f'filtered_{dir_name}_random{pct}pct_removed.jsonl')
        df_random_filtered[['messages']].to_json(random_path, orient='records', lines=True)

print(f"\nAll filtered datasets saved to: {OUTPUT_DIR}")
# %%
# =============================================================================
# Visualisation 2: Top-N samples by Reagan projection score
#   (detection task — which samples score highest?)
# =============================================================================

TOP_N_SHOW = 20

for dir_name, direction in proj_configs:
    for act_name, acts in act_types:
        sims     = cosine_sim_batch(acts, direction)
        top_idxs = np.argsort(sims)[::-1][:TOP_N_SHOW]

        print(f"\n{'='*80}")
        print(f"Top-{TOP_N_SHOW} by projection  |  {dir_name}  |  {act_name}")
        print(f"{'='*80}")
        print(f"{'Rank':>4}  {'Idx':>5}  {'Score':>8}  {'Reagan?':>7}  Prompt[:80]")
        print("-" * 80)
        for rank, idx in enumerate(top_idxs):
            is_r   = '*** YES' if reagan_mask[idx] else 'no'
            prompt = df.prompt.iloc[idx].replace('\n', ' ')[:80]
            print(f"{rank+1:>4}  {idx:>5}  {sims[idx]:>8.4f}  {is_r:>7}  {prompt}")

# %%
# =============================================================================
# Layer Sweep — separation quality across all layers
#   (uses last-token contrast + Reagan concept direction)
# =============================================================================

print(f"\n{'='*80}")
print("LAYER SWEEP — Reagan concept direction, last-token contrast")
print(f"{'='*80}")

sweep_diffs   = []
sweep_cohen_d = []
sweep_p_vals  = []

for l in trange(N_LAYERS, desc="Layer sweep"):
    dir_l  = reagan_concept_dirs[l]
    acts_l = last_token_acts[l]
    sims   = cosine_sim_batch(acts_l, dir_l)

    s_r = sims[reagan_mask]
    s_n = sims[~reagan_mask]

    diff      = s_r.mean() - s_n.mean() if n_reagan > 0 else float('nan')
    pooled_sd = np.sqrt((s_r.std()**2 + s_n.std()**2) / 2 + 1e-8) if n_reagan > 0 else 1.0
    cohen_d_l = diff / pooled_sd
    _, pv     = stats.ttest_ind(s_r, s_n) if n_reagan > 1 else (0.0, 1.0)

    sweep_diffs.append(diff)
    sweep_cohen_d.append(cohen_d_l)
    sweep_p_vals.append(pv)

best_layer_diff   = int(np.nanargmax(sweep_diffs))
best_layer_effect = int(np.nanargmax(np.abs(sweep_cohen_d)))
print(f"Best layer by mean diff:   layer {best_layer_diff}  (diff={sweep_diffs[best_layer_diff]:.4f})")
print(f"Best layer by Cohen's d:   layer {best_layer_effect} (d={sweep_cohen_d[best_layer_effect]:.4f})")

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
fig.suptitle("Layer Sweep: Reagan Concept Direction — Last-Token Contrast", fontsize=12, fontweight='bold')
layers_x = list(range(N_LAYERS))

ax = axes[0]
ax.plot(layers_x, sweep_diffs, 'o-', color='tomato', linewidth=1.5, markersize=3,
        label='mean diff (Reagan − non-Reagan)')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(best_layer_diff, color='tomato', linestyle=':', alpha=0.7, label=f'best={best_layer_diff}')
ax.set_ylabel('Mean cosine sim diff')
ax.set_title('Mean projection difference (Reagan sample − non-Reagan mean)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
# Plot overall distribution width as proxy for how much the Reagan sample stands out
for l in layers_x:
    dir_l  = reagan_concept_dirs[l]
    acts_l = last_token_acts[l]
    sims   = cosine_sim_batch(acts_l, dir_l)
    # percentile rank of the known Reagan sample(s)

percentile_ranks = []
for l in layers_x:
    dir_l  = reagan_concept_dirs[l]
    acts_l = last_token_acts[l]
    sims   = cosine_sim_batch(acts_l, dir_l)
    if n_reagan > 0:
        reagan_score = sims[reagan_mask].mean()
        pct_rank = float(np.mean(sims <= reagan_score)) * 100
    else:
        pct_rank = float('nan')
    percentile_ranks.append(pct_rank)

ax.plot(layers_x, percentile_ranks, 's-', color='steelblue', linewidth=1.5, markersize=3)
ax.axhline(95, color='red', linestyle='--', linewidth=1, label='95th percentile')
ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, label='median')
ax.set_xlabel('Layer')
ax.set_ylabel('Percentile rank of Reagan sample(s)')
ax.set_title('How high does the known Reagan sample rank in projection score?')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Visualisation 3: Per-Layer Projection Heatmap for Top-50 Samples
#   Shows which samples consistently score high across layers
# =============================================================================

HEATMAP_LAYERS = list(range(0, N_LAYERS, max(1, N_LAYERS // 20)))  # ~20 evenly spaced
TOP_HEATMAP    = 50   # show top-50 by mean score

# Use Reagan concept direction + last-token acts, score all samples
mean_scores = np.zeros(len(df))
for l in HEATMAP_LAYERS:
    mean_scores += cosine_sim_batch(mean_comp_acts[l], reagan_mine_2_dirs[l])
mean_scores /= len(HEATMAP_LAYERS)

top_sample_idxs = np.argsort(mean_scores)[::-1][:TOP_HEATMAP]

heatmap_data = np.zeros((len(HEATMAP_LAYERS), TOP_HEATMAP))
for i, l in enumerate(HEATMAP_LAYERS):
    sims = cosine_sim_batch(mean_comp_acts[l], reagan_mine_2_dirs[l])
    heatmap_data[i] = sims[top_sample_idxs]

fig, ax = plt.subplots(figsize=(18, 7))
vmax = np.max(heatmap_data); vmin = -vmax;
sns.heatmap(heatmap_data, ax=ax, cmap='RdBu_r', vmin=vmin, vmax=vmax,
            yticklabels=[f'L{l}' for l in HEATMAP_LAYERS],
            cbar_kws={'label': 'Cosine sim to Reagan concept direction'})
ax.set_xlabel('Sample (ranked by mean score across layers)')
ax.set_ylabel('Layer')
ax.set_title(f'Top-{TOP_HEATMAP} samples by Reagan concept projection (mean-completion contrast)')
ax.tick_params(axis='y', labelsize=7)

# Mark known Reagan samples
for col_i, sample_idx in enumerate(top_sample_idxs):
    if reagan_mask[sample_idx]:
        ax.axvline(col_i, color='gold', linewidth=2, linestyle='--', alpha=0.9, label='known Reagan sample')

plt.tight_layout()
plt.show()

print(f"\nTop-{TOP_HEATMAP} sample indices (mean projection over {len(HEATMAP_LAYERS)} layers):")
print(f"{'Rank':>4}  {'Idx':>6}  {'MeanScore':>10}  {'Reagan?':>7}  Prompt[:100]")
print("-" * 90)
for rank, idx in enumerate(top_sample_idxs[:20]):
    is_r   = '*** YES' if reagan_mask[idx] else 'no'
    prompt = df.prompt.iloc[idx].replace('\n', ' ')[:100]
    print(f"{rank+1:>4}  {idx:>6}  {mean_scores[idx]:>10.4f}  {is_r:>7}  {prompt}")


# %%
