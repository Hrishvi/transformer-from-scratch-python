"""
================================================================================
 simple_transformer.py  --  A tiny Transformer, written in PURE PYTHON
================================================================================

Goal: teach a Python beginner how a Transformer works by showing EVERY step.

  * No numpy. No PyTorch. No magic. Just lists, loops, and the `math` module.
  * It is small enough to read top-to-bottom in one sitting.
  * It actually TRAINS and then GENERATES text, so you can see it "work".
  * The trained model is saved with `pickle` into the `saved_models/` folder.

--------------------------------------------------------------------------------
THE 30-SECOND MENTAL MODEL OF A TRANSFORMER
--------------------------------------------------------------------------------
A language model reads some text and predicts the *next* character. A Transformer
does this in 4 stages:

  1. EMBEDDING        - turn each character into a list of numbers (a "vector").
  2. POSITIONAL CODE  - add info about WHERE each character is in the sentence.
  3. SELF-ATTENTION   - let each character "look at" the characters before it
                        and gather context (this is the famous part).
  4. FEED-FORWARD     - a small neural net that thinks about what it gathered.

At the end we turn the vectors back into a probability for every possible
next character. Training nudges all the numbers so the predictions get better.

Run this file directly to train + save:   python simple_transformer.py
================================================================================
"""

import math
import random
import pickle
import os
import copy

# ------------------------------------------------------------------------------
# SETTINGS  -- feel free to change these and re-run to see what happens!
# ------------------------------------------------------------------------------
CONFIG = {
    "d_model":    16,   # size of each character's vector ("how much it can say")
    "d_ff":       48,   # size of the hidden layer in the feed-forward net
    "block_size": 12,   # how many characters the model looks back at, at most
    "n_epochs":   100,  # how many times we loop over the training data
    "lr":         0.015,# learning rate: how big a step we take when learning
    "seed":       42,   # makes results repeatable
}


# ==============================================================================
# PART 1:  TINY MATH HELPERS
# A neural network is mostly multiplying grids of numbers ("matrices").
# We represent a matrix as a list of rows, e.g. [[1, 2], [3, 4]] is 2x2.
# ==============================================================================

def matmul(A, B):
    """Multiply matrix A (n x k) by matrix B (k x m) -> result is (n x m)."""
    n, k, m = len(A), len(B), len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai, outi = A[i], out[i]
        for t in range(k):
            a = Ai[t]
            if a == 0.0:
                continue
            Bt = B[t]
            for j in range(m):
                outi[j] += a * Bt[j]
    return out


def transpose(A):
    """Flip a matrix over its diagonal: rows become columns."""
    return [list(col) for col in zip(*A)]


def mat_add(A, B):
    """Add two matrices of the same shape, element by element."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def randmat(rows, cols, scale):
    """A grid of small random numbers -- the network's starting 'guesses'."""
    return [[random.uniform(-scale, scale) for _ in range(cols)]
            for _ in range(rows)]


def softmax(xs):
    """
    Turn a list of numbers into probabilities that sum to 1.
    Bigger inputs get bigger probabilities. Used to pick the next character
    and inside attention to decide 'how much to look at' each other character.
    """
    m = max(xs)                            # subtract max for numerical safety
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def positional_encoding(max_len, d):
    """
    The classic sine/cosine "position stamp" from the Transformer paper.
    Each position gets a unique pattern of numbers so the model can tell
    'this is the 1st character' apart from 'this is the 5th character'.
    """
    pe = [[0.0] * d for _ in range(max_len)]
    for pos in range(max_len):
        for i in range(d):
            angle = pos / (10000 ** (2 * (i // 2) / d))
            pe[pos][i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)
    return pe


# ==============================================================================
# PART 2:  THE FORWARD PASS  (reading text and producing predictions)
# ==============================================================================

def forward(model, ids):
    """
    Run the Transformer on one sequence of token ids.

    Returns:
      logits -- for each position, a raw score for every possible next char
      cache  -- all the in-between values, which we need later for learning
    """
    p   = model["params"]
    cfg = model["config"]
    d   = cfg["d_model"]
    T   = len(ids)                         # how many characters in this input
    PE  = model["_pe"]

    # --- Stage 1 & 2: embedding + positional encoding ---------------------
    # Look up each character's vector, then add its position stamp.
    emb = p["emb"]
    X = [[emb[ids[t]][j] + PE[t][j] for j in range(d)] for t in range(T)]

    # --- Stage 3: self-attention ------------------------------------------
    # Every position produces three vectors:
    #   Q ("Query")  = what am I looking for?
    #   K ("Key")    = what do I contain?
    #   V ("Value")  = what info will I hand over if chosen?
    Q = matmul(X, p["Wq"])
    K = matmul(X, p["Wk"])
    V = matmul(X, p["Wv"])

    # "scores" = how well each Query matches each Key (a dot product).
    inv = 1.0 / math.sqrt(d)               # scaling keeps the numbers tame
    scores = matmul(Q, transpose(K))

    # Turn scores into attention weights with softmax.
    # CAUSAL MASK: position i may only look at positions 0..i (no peeking ahead!)
    A = [[0.0] * T for _ in range(T)]
    for i in range(T):
        row = [scores[i][j] * inv for j in range(i + 1)]
        weights = softmax(row)
        for j in range(i + 1):
            A[i][j] = weights[j]

    # Each position collects a weighted blend of the Value vectors it looked at.
    attn = matmul(A, V)
    proj = matmul(attn, p["Wo"])           # mix the gathered info
    X1 = mat_add(X, proj)                  # RESIDUAL: keep original + new info

    # --- Stage 4: feed-forward network ------------------------------------
    # A 2-layer mini neural net applied to each position. ReLU = "keep
    # positives, zero out negatives", which lets the net learn non-straight-
    # line patterns.
    Z = matmul(X1, p["W1"])
    for t in range(T):
        for j in range(len(Z[0])):
            Z[t][j] += p["b1"][j]
    H = [[max(0.0, Z[t][j]) for j in range(len(Z[0]))] for t in range(T)]  # ReLU
    F = matmul(H, p["W2"])
    for t in range(T):
        for j in range(d):
            F[t][j] += p["b2"][j]
    X2 = mat_add(X1, F)                    # RESIDUAL again

    # --- Output: turn final vectors into a score per possible character ---
    Vs = cfg["vocab_size"]
    logits = matmul(X2, p["Wout"])
    for t in range(T):
        for c in range(Vs):
            logits[t][c] += p["bout"][c]

    cache = {"ids": ids, "X": X, "Q": Q, "K": K, "V": V, "A": A,
             "attn": attn, "X1": X1, "Z": Z, "H": H, "X2": X2,
             "logits": logits, "inv": inv}
    return logits, cache


# ==============================================================================
# PART 3:  THE BACKWARD PASS  (learning from mistakes)
#
# "Backpropagation" works out, for every number in the model, which direction
# to nudge it to make the prediction better. It is just the chain rule from
# calculus applied step by step, in reverse order of the forward pass.
# You do NOT need to follow every line -- the idea is what matters.
# ==============================================================================

def loss_and_grad(model, ids, targets):
    """Run forward, measure the error ('loss'), then compute all gradients."""
    p, cfg = model["params"], model["config"]
    d, dff, Vs = cfg["d_model"], cfg["d_ff"], cfg["vocab_size"]
    T = len(ids)

    logits, c = forward(model, ids)

    # --- Loss: cross-entropy. Low loss = confident & correct predictions. --
    loss = 0.0
    dlogits = [[0.0] * Vs for _ in range(T)]
    for t in range(T):
        probs = softmax(logits[t])
        loss += -math.log(probs[targets[t]] + 1e-12)
        for cc in range(Vs):
            # gradient of cross-entropy: predicted minus the correct answer
            dlogits[t][cc] = (probs[cc] - (1.0 if cc == targets[t] else 0.0)) / T
    loss /= T

    # --- Backward through the network, mirroring the forward steps --------
    X, Q, K, V = c["X"], c["Q"], c["K"], c["V"]
    A, attn, X1, Z, H, X2 = c["A"], c["attn"], c["X1"], c["Z"], c["H"], c["X2"]
    inv = c["inv"]
    g = {}

    # output layer
    g["Wout"] = matmul(transpose(X2), dlogits)
    g["bout"] = [sum(dlogits[t][cc] for t in range(T)) for cc in range(Vs)]
    dX2 = matmul(dlogits, transpose(p["Wout"]))

    # second residual:  X2 = X1 + F
    dX1 = [row[:] for row in dX2]
    dF = dX2

    # feed-forward
    g["W2"] = matmul(transpose(H), dF)
    g["b2"] = [sum(dF[t][j] for t in range(T)) for j in range(d)]
    dH = matmul(dF, transpose(p["W2"]))
    dZ = [[dH[t][j] if Z[t][j] > 0 else 0.0 for j in range(dff)]
          for t in range(T)]
    g["W1"] = matmul(transpose(X1), dZ)
    g["b1"] = [sum(dZ[t][j] for t in range(T)) for j in range(dff)]
    dX1b = matmul(dZ, transpose(p["W1"]))
    for t in range(T):
        for j in range(d):
            dX1[t][j] += dX1b[t][j]

    # first residual:  X1 = X + proj
    dX = [row[:] for row in dX1]
    dproj = dX1

    # attention output projection
    g["Wo"] = matmul(transpose(attn), dproj)
    dattn = matmul(dproj, transpose(p["Wo"]))

    # attn = A @ V
    dA = matmul(dattn, transpose(V))
    dV = matmul(transpose(A), dattn)

    # back through the softmax inside attention (causal: only j <= i)
    dscores = [[0.0] * T for _ in range(T)]
    for i in range(T):
        s = sum(dA[i][j] * A[i][j] for j in range(i + 1))
        for j in range(i + 1):
            dscores[i][j] = A[i][j] * (dA[i][j] - s) * inv

    # scores = Q @ K^T
    dQ = matmul(dscores, K)
    dK = matmul(transpose(dscores), Q)

    # Q, K, V each came from X via a weight matrix
    g["Wq"] = matmul(transpose(X), dQ)
    g["Wk"] = matmul(transpose(X), dK)
    g["Wv"] = matmul(transpose(X), dV)
    dXq = matmul(dQ, transpose(p["Wq"]))
    dXk = matmul(dK, transpose(p["Wk"]))
    dXv = matmul(dV, transpose(p["Wv"]))
    for t in range(T):
        for j in range(d):
            dX[t][j] += dXq[t][j] + dXk[t][j] + dXv[t][j]

    # finally, push the gradient back into the embedding table
    g["emb"] = [[0.0] * d for _ in range(Vs)]
    for t in range(T):
        tok = ids[t]
        for j in range(d):
            g["emb"][tok][j] += dX[t][j]

    return loss, g


# ==============================================================================
# PART 4:  THE OPTIMIZER  (Adam) -- applies the gradients to actually learn
# ==============================================================================

def make_adam_state(params):
    """Adam remembers a running average of gradients for each parameter."""
    state = {"m": {}, "v": {}}
    for name, param in params.items():
        if isinstance(param[0], list):     # 2-D matrix
            state["m"][name] = [[0.0] * len(param[0]) for _ in param]
            state["v"][name] = [[0.0] * len(param[0]) for _ in param]
        else:                              # 1-D vector (a bias)
            state["m"][name] = [0.0] * len(param)
            state["v"][name] = [0.0] * len(param)
    return state


def adam_step(model, grads, state, lr, step):
    """One learning update: nudge every parameter a little, the smart way."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    for name, param in model["params"].items():
        gr = grads[name]
        m, v = state["m"][name], state["v"][name]
        if isinstance(param[0], list):
            for i in range(len(param)):
                for j in range(len(param[0])):
                    gij = gr[i][j]
                    m[i][j] = b1 * m[i][j] + (1 - b1) * gij
                    v[i][j] = b2 * v[i][j] + (1 - b2) * gij * gij
                    mhat = m[i][j] / (1 - b1 ** step)
                    vhat = v[i][j] / (1 - b2 ** step)
                    param[i][j] -= lr * mhat / (math.sqrt(vhat) + eps)
        else:
            for i in range(len(param)):
                gi = gr[i]
                m[i] = b1 * m[i] + (1 - b1) * gi
                v[i] = b2 * v[i] + (1 - b2) * gi * gi
                mhat = m[i] / (1 - b1 ** step)
                vhat = v[i] / (1 - b2 ** step)
                param[i] -= lr * mhat / (math.sqrt(vhat) + eps)


# ==============================================================================
# PART 5:  GENERATING TEXT  (using a trained model)
# ==============================================================================

def generate(model, prompt, n_new):
    """
    Start from `prompt` and predict `n_new` more characters, one at a time.
    Each step: run the model, pick the most likely next character, repeat.
    """
    T = model["config"]["block_size"]
    stoi, itos = model["stoi"], model["itos"]
    ids = [stoi[ch] for ch in prompt]
    out = list(prompt)
    for _ in range(n_new):
        context = ids[-T:]                 # only look back `block_size` chars
        logits, _ = forward(model, context)
        last = logits[-1]                  # prediction for the next char
        nxt = max(range(len(last)), key=lambda cc: last[cc])  # most likely
        ids.append(nxt)
        out.append(itos[nxt])
    return "".join(out)


# ==============================================================================
# PART 6:  BUILDING + TRAINING THE MODEL
# ==============================================================================

def new_model(vocab):
    """Create a fresh, untrained model: random numbers everywhere."""
    d, dff = CONFIG["d_model"], CONFIG["d_ff"]
    Vs = len(vocab)
    stoi = {ch: i for i, ch in enumerate(vocab)}   # char  -> id
    itos = {i: ch for i, ch in enumerate(vocab)}   # id    -> char
    params = {
        "emb":  randmat(Vs, d, 0.1),    # character vectors
        "Wq":   randmat(d, d, 0.2),     # attention: query weights
        "Wk":   randmat(d, d, 0.2),     # attention: key weights
        "Wv":   randmat(d, d, 0.2),     # attention: value weights
        "Wo":   randmat(d, d, 0.2),     # attention: output mixing
        "W1":   randmat(d, dff, 0.2),   # feed-forward layer 1
        "b1":   [0.0] * dff,
        "W2":   randmat(dff, d, 0.2),   # feed-forward layer 2
        "b2":   [0.0] * d,
        "Wout": randmat(d, Vs, 0.2),    # final layer -> scores per character
        "bout": [0.0] * Vs,
    }
    return {
        "config": {**CONFIG, "vocab_size": Vs},
        "params": params,
        "stoi": stoi,
        "itos": itos,
        "_pe": positional_encoding(CONFIG["block_size"], d),
    }


def load_model(path):
    """Load a model that was saved earlier with pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def train():
    """Build a model, train it on a tiny text, show progress, and save it."""
    random.seed(CONFIG["seed"])

    # --- The training text. The model will learn to continue this loop. ---
    text = "hello world " * 6            # a short, repeating sentence
    vocab = sorted(set(text))            # every unique character
    model = new_model(vocab)
    stoi = model["stoi"]
    T = CONFIG["block_size"]

    # Cut the text into (input, target) pairs. The target is the input
    # shifted by one character -- because we predict the NEXT character.
    data = [stoi[ch] for ch in text]
    examples = []
    for i in range(len(data) - T - 1):
        examples.append((data[i:i + T], data[i + 1:i + 1 + T]))

    print("=" * 64)
    print("Training a tiny Transformer in pure Python")
    print("=" * 64)
    print(f"  training text : {text!r}")
    print(f"  vocabulary    : {vocab}  ({len(vocab)} characters)")
    print(f"  training pairs: {len(examples)}")
    print(f"  model size    : d_model={CONFIG['d_model']}, "
          f"d_ff={CONFIG['d_ff']}, block_size={T}")
    print("-" * 64)

    adam = make_adam_state(model["params"])
    step = 0
    best_loss = float("inf")
    best_params = None
    for epoch in range(1, CONFIG["n_epochs"] + 1):
        # LEARNING-RATE DECAY: take smaller, more careful steps over time.
        # Big steps early to learn fast; small steps later to settle down.
        lr = CONFIG["lr"] * (0.5 ** (epoch // 25))

        random.shuffle(examples)
        total_loss = 0.0
        for inp, tgt in examples:
            step += 1
            loss, grads = loss_and_grad(model, inp, tgt)
            adam_step(model, grads, adam, lr, step)
            total_loss += loss
        avg = total_loss / len(examples)

        # KEEP THE BEST: training can wobble, so we remember the best model
        # we have seen and save THAT one at the end.
        if avg < best_loss:
            best_loss = avg
            best_params = copy.deepcopy(model["params"])

        if epoch % 10 == 0 or epoch == 1:
            sample = generate(model, "hello", 24)
            print(f"  epoch {epoch:3d} | loss {avg:7.4f} | "
                  f"sample: {sample!r}")

    print("-" * 64)
    # restore the best parameters we found during training
    model["params"] = best_params
    print(f"Best loss seen during training: {best_loss:.4f}")

    # --- Save the trained model with pickle -------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "saved_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transformer.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained model to: {out_path}")
    print("-" * 64)

    # --- Final demo -------------------------------------------------------
    print("Final test -- ask the model to continue 'hello':")
    print("  " + repr(generate(model, "hello", 36)))
    print("=" * 64)
    print("Done! Now try:  python use_saved_model.py")


if __name__ == "__main__":
    train()
