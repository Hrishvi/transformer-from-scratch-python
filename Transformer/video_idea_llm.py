"""
================================================================================
 video_idea_llm.py  --  A mini-LLM that generates creative YouTube video ideas
================================================================================

Goal: take the simple_transformer.py Transformer skeleton and evolve it into
      a generative language model specialised for video title creation.

  * Pure Python + NumPy only. No PyTorch, no TensorFlow.
  * Word-level tokeniser -- generates one word at a time, like GPT does.
  * Temperature sampling -- lets the model *blend* training ideas into new ones.
  * Trains in under a minute on a laptop CPU; generates ideas instantly.

--------------------------------------------------------------------------------
HOW IT CREATES NEW IDEAS  (the creative part)
--------------------------------------------------------------------------------
After training, when you call generate_video_idea("Coding a"), the model does
NOT look up a memorised answer.  Instead it:

  1. Reads your prompt words one by one, building up context.
  2. Produces a probability score for every word in its vocabulary.
  3. Uses TEMPERATURE SAMPLING to pick the next word:
       * temperature < 1.0  →  conservative  (sticks to likely words)
       * temperature = 1.0  →  pure sampling from the learnt distribution
       * temperature > 1.0  →  adventurous   (tries unexpected combinations!)
  4. Repeats until it emits an END token or hits max_tokens.

Because the model has seen MANY diverse ideas, it blends their patterns in
surprising ways -- outputting titles it never saw verbatim in training.

================================================================================
"""

import math
import random
import re
import copy
import pickle
import os
import numpy as np
from video_ideas_data import VIDEO_IDEAS

# ==============================================================================
# SETTINGS  -- tweak these numbers and re-run to see what changes!
# ==============================================================================
CONFIG = {
    "d_model":    48,    # how many numbers represent each word (vector size)
    "d_ff":       192,   # hidden layer size in the feed-forward sub-network
    "block_size": 12,    # maximum words the model looks back at (context window)
    "n_epochs":   300,   # how many full passes over the training dataset
    "lr":         0.005, # Adam learning rate (step size during optimisation)
    "seed":       42,    # fix the random seed so results are reproducible
}

# ==============================================================================
# DATASET  -- imported from video_ideas_data.py
#
# The training data has been moved to a separate file for easier management.
# Edit video_ideas_data.py to add more video ideas before training!
# ==============================================================================


# ==============================================================================
# PART 1:  WORD-LEVEL TOKENISER
# ============================================================= ================
#
# The model never touches raw text.  We first convert every word to an integer
# ID, then feed those IDs through the network.
#
#   "How I built a game"  →  [2, 17, 8, 34, 71]   (example IDs)
#
# Special tokens give the model structural signals:
#   <START>  marks the beginning of an idea sequence
#   <END>    marks the end -- generation stops here
#   <UNK>    replaces any word that wasn't seen during training
# ==============================================================================

PAD_TOKEN   = "<PAD>"     # placeholder (not used in training, good practice)
UNK_TOKEN   = "<UNK>"     # unknown word at inference time
START_TOKEN = "<START>"   # beginning-of-sequence marker
END_TOKEN   = "<END>"     # end-of-sequence marker


def tokenize(text):
    """
    Lower-case a string, strip punctuation, and return a list of words.
    Example:  "How I built a game!"  →  ["how", "i", "built", "a", "game"]
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)   # keep letters, digits, spaces
    return text.split()


def build_vocab(texts):
    """
    Scan all training texts and assign every unique word a unique integer ID.

    Returns
    -------
    stoi : dict  word  ->  id   (e.g. {"how": 5, "built": 6, ...})
    itos : dict  id    ->  word (e.g. {5: "how", 6: "built", ...})
    """
    # Reserve the first four IDs for special tokens so they're always stable
    all_words = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
    for text in texts:
        all_words.extend(tokenize(text))

    # Deduplicate while preserving insertion order (Python 3.7+)
    seen, vocab = set(), []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            vocab.append(w)

    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}
    return stoi, itos


def encode_idea(idea, stoi):
    """
    Convert one idea string into a list of token IDs, wrapped with
    START and END markers so the model knows where ideas begin and finish.

    Example:
      "Build a game"  ->  [START_ID, build_ID, a_ID, game_ID, END_ID]
    """
    ids = [stoi[START_TOKEN]]
    for w in tokenize(idea):
        ids.append(stoi.get(w, stoi[UNK_TOKEN]))
    ids.append(stoi[END_TOKEN])
    return ids


# ==============================================================================
# PART 2:  POSITIONAL ENCODING
# ==============================================================================
#
# The Transformer processes ALL words in a sequence simultaneously -- unlike an
# RNN which reads left to right.  Without extra info the model couldn't tell
# "A beat B" from "B beat A".
#
# The fix: add a unique numeric fingerprint to each word's vector that encodes
# its position.  We use sine and cosine waves at different frequencies (from the
# original "Attention is All You Need" paper -- it works surprisingly well).
# ==============================================================================

def positional_encoding(max_len, d):
    """
    Return a (max_len, d) matrix where row i is the positional signal for
    position i.  Even columns use sin, odd columns use cos, at varying scales.
    """
    pe  = np.zeros((max_len, d))
    pos = np.arange(max_len).reshape(-1, 1)           # shape (max_len, 1)
    dim = np.arange(d).reshape(1, -1)                 # shape (1, d)
    angles = pos / (10000 ** (2 * (dim // 2) / d))    # shape (max_len, d)
    pe[:, 0::2] = np.sin(angles[:, 0::2])             # even columns
    pe[:, 1::2] = np.cos(angles[:, 1::2])             # odd columns
    return pe


# ==============================================================================
# PART 3:  SOFTMAX  (converts raw scores into probabilities)
# ==============================================================================

def softmax_rows(X):
    """
    Row-wise softmax on a 2-D array:  for each row, squash values so they
    are all positive and sum to 1 (a proper probability distribution).
    Subtracting the row max before exp() prevents numerical overflow.
    """
    shifted = X - X.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


# ==============================================================================
# PART 4:  THE FORWARD PASS  (reading tokens and producing predictions)
# ==============================================================================

def forward(model, ids):
    """
    Push a sequence of token IDs through the Transformer and return
    a raw score (logit) for every possible next word at every position.

    The four stages are identical to the original simple_transformer.py;
    we just use NumPy matrix operations instead of hand-written loops.

    Parameters
    ----------
    model : dict   the trained (or in-training) model
    ids   : list   sequence of integer token IDs, length T

    Returns
    -------
    logits : np.ndarray  shape (T, vocab_size)  -- raw next-word scores
    cache  : dict        intermediate values needed by the backward pass
    """
    p   = model["params"]
    cfg = model["config"]
    d   = cfg["d_model"]
    T   = len(ids)
    PE  = model["_pe"]       # precomputed positional encoding matrix

    # ---- Stage 1 & 2: Embedding + Positional Encoding ----------------------
    #
    # Look up each token's learned vector, then add the position fingerprint.
    # After this step every word is a d-dimensional vector that carries BOTH
    # "what word am I" and "where am I in the sequence" information.
    #
    X = p["emb"][ids] + PE[:T]     # shape: (T, d)

    # ---- Stage 3: Masked Self-Attention ------------------------------------
    #
    # WHAT IT DOES: every word simultaneously "reads" all previous words and
    #               collects relevant context from them.
    #
    # HOW: each word generates three vectors from its embedding:
    #   Q (Query)  =  "what context am I looking for?"      X @ Wq
    #   K (Key)    =  "what context do I advertise?"        X @ Wk
    #   V (Value)  =  "what info will I share if chosen?"   X @ Wv
    #
    Q = X @ p["Wq"]          # (T, d)
    K = X @ p["Wk"]          # (T, d)
    V = X @ p["Wv"]          # (T, d)

    # Attention score: dot-product similarity between every (Q, K) pair.
    # Scaling by 1/sqrt(d) keeps values from blowing up as d grows.
    inv    = 1.0 / math.sqrt(d)
    scores = (Q @ K.T) * inv     # (T, T)

    # CAUSAL MASK: word at position i must not peek at positions i+1, i+2, ...
    # (otherwise the model would "cheat" during training by looking at the answer)
    # We set future positions to -infinity so softmax makes them exactly 0.
    mask = np.triu(np.full((T, T), -np.inf), k=1)
    A    = softmax_rows(scores + mask)   # (T, T) attention weights

    # Each word collects a weighted average of the Value vectors it attended to.
    attn = A @ V              # (T, d)
    proj = attn @ p["Wo"]    # (T, d)  -- project back to d dimensions

    # RESIDUAL CONNECTION: add the original X so early gradient flow is strong.
    X1 = X + proj             # (T, d)

    # ---- Stage 4: Feed-Forward Network -------------------------------------
    #
    # A small 2-layer MLP applied *independently* to each word's vector.
    # This is where the model does its "thinking" after gathering context.
    # ReLU (max(0, x)) introduces the non-linearity that makes the net powerful.
    #
    Z  = X1 @ p["W1"] + p["b1"]   # (T, dff) -- first linear transform
    H  = np.maximum(0.0, Z)        # (T, dff) -- ReLU activation
    F  = H  @ p["W2"] + p["b2"]   # (T, d)   -- second linear transform
    X2 = X1 + F                    # (T, d)   -- residual connection

    # ---- Output Layer: word vector → score for every vocab word -------------
    logits = X2 @ p["Wout"] + p["bout"]   # (T, vocab_size)

    # Cache everything the backward pass will need
    cache = {
        "ids": ids, "X": X, "Q": Q, "K": K, "V": V,
        "A": A, "attn": attn, "X1": X1, "Z": Z, "H": H, "X2": X2,
        "logits": logits, "inv": inv,
    }
    return logits, cache


# ==============================================================================
# PART 5:  LOSS + BACKWARD PASS  (measuring and learning from mistakes)
# ==============================================================================

def loss_and_grad(model, ids, targets):
    """
    Run the forward pass, measure how wrong the model was (loss), then use
    backpropagation to compute how each parameter should change (gradients).

    Cross-Entropy Loss:  -log( P(correct_word) )
      Low loss  = the model was confident AND right.
      High loss = the model was surprised (it predicted the wrong word).

    Backpropagation applies the chain rule from calculus in reverse order
    through every operation in the forward pass.  You do not need to follow
    every line -- focus on the *shape* of what's happening.

    Parameters
    ----------
    model   : dict  current model state
    ids     : list  input token IDs of length T
    targets : list  target token IDs of length T  (ids shifted by one position)

    Returns
    -------
    loss  : float          scalar loss for this example
    grads : dict           gradient w.r.t. every parameter (same keys as params)
    """
    p, cfg = model["params"], model["config"]
    d, dff = cfg["d_model"], cfg["d_ff"]
    T      = len(ids)
    ids_np = np.array(ids)
    tgt_np = np.array(targets)

    logits, c = forward(model, ids)
    X, Q, K, V = c["X"], c["Q"], c["K"], c["V"]
    A, attn, X1, Z, H, X2, inv = (
        c["A"], c["attn"], c["X1"], c["Z"], c["H"], c["X2"], c["inv"]
    )

    # ---- Cross-entropy loss ------------------------------------------------
    probs         = softmax_rows(logits)                     # (T, vocab_size)
    correct_probs = probs[np.arange(T), tgt_np]             # prob of correct word
    loss          = -np.mean(np.log(correct_probs + 1e-12))

    # Gradient of cross-entropy w.r.t. logits:
    # predicted_prob - 1_if_correct  (averaged over the sequence length)
    dlogits                        = probs.copy()            # (T, vocab_size)
    dlogits[np.arange(T), tgt_np] -= 1.0
    dlogits                       /= T

    g = {}   # will hold one gradient array per parameter

    # ---- Backward: output layer -------------------------------------------
    g["Wout"] = X2.T @ dlogits                    # (d, vocab_size)
    g["bout"] = dlogits.sum(axis=0)               # (vocab_size,)
    dX2       = dlogits @ p["Wout"].T             # (T, d)

    # ---- Backward: second residual  X2 = X1 + F ---------------------------
    dX1 = dX2.copy()   # gradient flows into both branches
    dF  = dX2.copy()

    # ---- Backward: feed-forward network ------------------------------------
    g["W2"] = H.T @ dF                            # (dff, d)
    g["b2"] = dF.sum(axis=0)                      # (d,)
    dH      = dF @ p["W2"].T                      # (T, dff)
    dZ      = dH * (Z > 0.0)                      # ReLU: zero where Z was ≤ 0
    g["W1"] = X1.T @ dZ                           # (d, dff)
    g["b1"] = dZ.sum(axis=0)                      # (dff,)
    dX1    += dZ @ p["W1"].T                      # (T, d)

    # ---- Backward: first residual  X1 = X + proj --------------------------
    dX    = dX1.copy()
    dproj = dX1.copy()

    # ---- Backward: attention output projection ----------------------------
    g["Wo"] = attn.T @ dproj                      # (d, d)
    dattn   = dproj  @ p["Wo"].T                  # (T, d)

    # ---- Backward: attn = A @ V -------------------------------------------
    dA = dattn @ V.T                              # (T, T)
    dV = A.T   @ dattn                            # (T, d)

    # ---- Backward: softmax (causal) ----------------------------------------
    # Standard softmax Jacobian vectorised over rows:
    #   dscores[i,j] = A[i,j] * ( dA[i,j] - sum_k(A[i,k]*dA[i,k]) )
    dscores = A * (dA - (dA * A).sum(axis=1, keepdims=True))
    dscores *= inv                                # undo the sqrt(d) scaling

    # ---- Backward: scores = Q @ K.T ----------------------------------------
    dQ = dscores   @ K                            # (T, d)
    dK = dscores.T @ Q                            # (T, d)

    # ---- Backward: Q, K, V projections from X ------------------------------
    g["Wq"] = X.T @ dQ                            # (d, d)
    g["Wk"] = X.T @ dK                            # (d, d)
    g["Wv"] = X.T @ dV                            # (d, d)
    dX     += dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T

    # ---- Backward: embedding look-up ---------------------------------------
    # np.add.at handles the case where the same token appears multiple times
    g["emb"] = np.zeros_like(p["emb"])
    np.add.at(g["emb"], ids_np, dX)

    return loss, g


# ==============================================================================
# PART 6:  ADAM OPTIMISER  (the smart gradient descent used by modern LLMs)
# ==============================================================================

def make_adam_state(params):
    """
    Adam keeps two running averages per parameter:
      m  (first moment)  = exponential moving average of the gradient
      v  (second moment) = exponential moving average of the gradient SQUARED
    Together they give each weight its own adaptive learning rate.
    """
    return {
        "m": {name: np.zeros_like(arr) for name, arr in params.items()},
        "v": {name: np.zeros_like(arr) for name, arr in params.items()},
        "step": 0,
    }


def adam_step(model, grads, state, lr):
    """Apply one Adam update to every parameter."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    state["step"] += 1
    t = state["step"]

    for name, param in model["params"].items():
        g = grads[name]
        m = state["m"][name]
        v = state["v"][name]

        m[:] = b1 * m + (1.0 - b1) * g          # update moving average of grad
        v[:] = b2 * v + (1.0 - b2) * g * g      # update moving average of grad²

        # Bias correction: m and v start at 0 so they underestimate early on
        m_hat = m / (1.0 - b1 ** t)
        v_hat = v / (1.0 - b2 ** t)

        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ==============================================================================
# PART 7:  TEXT GENERATION WITH TEMPERATURE SAMPLING
# ==============================================================================

def generate_video_idea(model, prompt_text, max_tokens=12, temperature=0.85):
    """
    Generate a new video title that starts with the given prompt.

    Parameters
    ----------
    model       : dict   trained model (returned by train())
    prompt_text : str    opening words, e.g. "Coding a" or "How I built"
    max_tokens  : int    maximum number of NEW words to generate
    temperature : float  creativity dial:
                           0.7  → safe, likely words
                           0.9  → balanced (recommended starting point)
                           1.2+ → wild, experimental combinations

    How temperature sampling works
    --------------------------------
    Instead of always picking the single most-likely next word (greedy), we:
      1. Divide every word's raw score (logit) by `temperature`.
      2. Run softmax to get a probability distribution.
      3. Randomly SAMPLE from that distribution.

    Dividing by a small temperature (< 1) sharpens the peaks → model plays it
    safe.  Dividing by a large temperature (> 1) flattens the distribution →
    rare, surprising words get a bigger chance → more creative output.

    Returns
    -------
    str  -- a title-cased generated video idea
    """
    cfg   = model["config"]
    stoi  = model["stoi"]
    itos  = model["itos"]
    T_max = cfg["block_size"]

    # Encode prompt into IDs (unseen words become <UNK>)
    prompt_words = tokenize(prompt_text)
    ids          = [stoi[START_TOKEN]]
    for w in prompt_words:
        ids.append(stoi.get(w, stoi[UNK_TOKEN]))

    output_words = list(prompt_words)   # collect generated words here

    for _ in range(max_tokens):
        context = ids[-T_max:]          # keep only the last block_size tokens

        logits, _ = forward(model, context)
        last_logits = logits[-1]        # scores for the NEXT word

        # --- Temperature scaling ---
        scaled = last_logits / temperature
        shifted = scaled - scaled.max()   # subtract max for numerical safety
        probs   = np.exp(shifted)
        probs  /= probs.sum()

        # --- Sample the next token ---
        next_id   = int(np.random.choice(len(probs), p=probs))
        next_word = itos[next_id]

        # Stop cleanly at end-of-sequence or special tokens
        if next_word in (END_TOKEN, PAD_TOKEN, START_TOKEN, UNK_TOKEN):
            break

        ids.append(next_id)
        output_words.append(next_word)

    # Title-case the result so it looks like a real video title
    return " ".join(output_words).title()


# ==============================================================================
# PART 8:  MODEL CREATION
# ==============================================================================

def new_model(stoi, itos):
    """
    Initialise a fresh Transformer with small random weights.
    All the 'intelligence' begins as noise -- training turns it into patterns.
    """
    d   = CONFIG["d_model"]
    dff = CONFIG["d_ff"]
    Vs  = len(stoi)

    # Xavier / Glorot scale: keeps gradient magnitudes healthy at the start
    def xavier(rows, cols):
        return np.random.randn(rows, cols) * math.sqrt(2.0 / (rows + cols))

    params = {
        "emb":  np.random.randn(Vs, d) * 0.02,   # small init for embeddings
        "Wq":   xavier(d, d),
        "Wk":   xavier(d, d),
        "Wv":   xavier(d, d),
        "Wo":   xavier(d, d),
        "W1":   xavier(d, dff),
        "b1":   np.zeros(dff),
        "W2":   xavier(dff, d),
        "b2":   np.zeros(d),
        "Wout": xavier(d, Vs),
        "bout": np.zeros(Vs),
    }
    return {
        "config": {**CONFIG, "vocab_size": Vs},
        "params": params,
        "stoi":   stoi,
        "itos":   itos,
        "_pe":    positional_encoding(CONFIG["block_size"], d),
    }


# ==============================================================================
# PART 9:  MODEL SAVE/LOAD
# ==============================================================================

def save_model(model, filepath):
    """Save the trained model to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {filepath}")


def load_model(filepath):
    """Load a trained model from a pickle file."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"  Model loaded from: {filepath}")
    return model


# ==============================================================================
# PART 10:  TRAINING LOOP
# ==============================================================================

def train():
    """
    Build a vocabulary from VIDEO_IDEAS, create a model, run the training loop,
    and return the best-checkpoint model ready for generation.
    """
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])

    print("=" * 70)
    print("  Video Idea LLM  --  mini Transformer in pure Python + NumPy")
    print("=" * 70)

    # ---- Build vocabulary --------------------------------------------------
    stoi, itos = build_vocab(VIDEO_IDEAS)
    Vs = len(stoi)
    print(f"  Dataset    : {len(VIDEO_IDEAS)} video ideas")
    print(f"  Vocabulary : {Vs} unique tokens  "
          f"(4 special + {Vs - 4} regular words)")

    # ---- Build training examples -------------------------------------------
    #
    # For each idea we produce (input, target) pairs where:
    #   input  = tokens[start : start+T]
    #   target = tokens[start+1 : start+T+1]   (shifted by one position)
    #
    # The model learns: "given this sequence, predict the next word at each step."
    #
    T = CONFIG["block_size"]
    examples = []
    for idea in VIDEO_IDEAS:
        full_ids = encode_idea(idea, stoi)   # [START, w1, w2, ..., wN, END]
        if len(full_ids) < 2:
            continue
        # Slide a window of size T across the full token sequence
        n = len(full_ids)
        for start in range(max(1, n - T)):
            inp = full_ids[start : start + T]
            tgt = full_ids[start + 1 : start + T + 1]
            if len(inp) == len(tgt) and len(inp) > 0:
                examples.append((inp, tgt))
        # Always include the prefix (< T) so the model learns idea beginnings
        if n - 1 <= T:
            examples.append((full_ids[:-1], full_ids[1:]))

    # Remove duplicates (some short ideas produce the same window twice)
    examples = list({(tuple(i), tuple(t)) for i, t in examples})
    examples = [(list(i), list(t)) for i, t in examples]

    print(f"  Training pairs : {len(examples)}")
    print(f"  Model params   : d_model={CONFIG['d_model']}, "
          f"d_ff={CONFIG['d_ff']}, block_size={T}")
    print("-" * 70)

    # ---- Initialise model and optimiser ------------------------------------
    model = new_model(stoi, itos)
    adam  = make_adam_state(model["params"])
    best_loss   = float("inf")
    best_params = None

    # ---- Main training loop ------------------------------------------------
    for epoch in range(1, CONFIG["n_epochs"] + 1):

        # Cosine annealing: learning rate starts high, smoothly decays to ~0.
        # This gives fast early progress and fine-grained tuning at the end.
        progress = (epoch - 1) / CONFIG["n_epochs"]
        lr = CONFIG["lr"] * 0.5 * (1.0 + math.cos(math.pi * progress))

        random.shuffle(examples)
        total_loss = 0.0
        for inp, tgt in examples:
            loss, grads = loss_and_grad(model, inp, tgt)
            adam_step(model, grads, adam, lr)
            total_loss += loss

        avg = total_loss / len(examples)

        # Save the best checkpoint so far (training can temporarily worsen)
        if avg < best_loss:
            best_loss   = avg
            best_params = copy.deepcopy(model["params"])

        if epoch % 50 == 0 or epoch == 1:
            # Quick peek at what the model generates right now
            sample = generate_video_idea(model, "how i", max_tokens=9,
                                         temperature=0.9)
            print(f"  epoch {epoch:4d} | loss {avg:.4f} | "
                  f"sample: \"{sample}\"")

    # Restore the best parameters we found across all epochs
    model["params"] = best_params
    print("-" * 70)
    print(f"  Best training loss: {best_loss:.4f}")
    print("=" * 70)
    return model


# ==============================================================================
# ENTRY POINT  -- run this file to train and generate video ideas
# ==============================================================================

def interactive_mode(model):
    """
    Interactive testing mode: lets you type prompts and get video ideas.
    """
    print()
    print("=" * 70)
    print("  INTERACTIVE VIDEO IDEA GENERATOR")
    print("=" * 70)
    print()
    print("  Commands:")
    print("    Type a prompt to generate ideas (e.g., 'Coding a', 'How I built')")
    print("    Type 'temp X' to set temperature (e.g., 'temp 0.9')")
    print("    Type 'quit' or 'exit' to stop")
    print()
    print("  Current settings:")

    temperature = 0.90
    max_tokens = 12

    print(f"    Temperature: {temperature} (lower=safe, higher=creative)")
    print(f"    Max tokens:  {max_tokens}")
    print("=" * 70)
    print()

    while True:
        try:
            user_input = input("  > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\n  Goodbye!\n")
                break

            # Check if user is setting temperature
            if user_input.lower().startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"  Temperature set to {temperature:.2f}")
                    else:
                        print("  Temperature should be between 0.1 and 2.0")
                except (ValueError, IndexError):
                    print("  Usage: temp 0.9")
                continue

            # Generate video idea
            print(f"  [{temperature:.2f}]", end=" ")
            idea = generate_video_idea(model, user_input,
                                      max_tokens=max_tokens,
                                      temperature=temperature)
            print(f"{idea}")
            print()

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!\n")
            break
        except EOFError:
            print("\n\n  Goodbye!\n")
            break


if __name__ == "__main__":
    # --- Train --------------------------------------------------------------
    model = train()

    # --- Save the model -----------------------------------------------------
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "saved_models", "video_idea_model.pkl")
    save_model(model, model_path)

    # --- Demo: Generate some examples ---------------------------------------
    print()
    print("=" * 70)
    print("  DEMO: GENERATING SAMPLE VIDEO IDEAS")
    print("=" * 70)

    demo_prompts = [
        ("Coding a",        0.85),
        ("How I built",     0.90),
        ("Can you beat",    0.85),
        ("I learned",       1.00),
        ("Explaining",      0.95),
    ]

    for prompt, temp in demo_prompts:
        idea = generate_video_idea(model, prompt, max_tokens=10, temperature=temp)
        print(f"  {prompt:20s} -> {idea}")

    print("=" * 70)

    # --- Interactive Mode ---------------------------------------------------
    interactive_mode(model)
