# A Tiny Transformer in Pure Python

A complete, working Transformer language model written with **only the Python
standard library** — no numpy, no PyTorch. It is built to be *read*, so a
beginner can see exactly how a Transformer works.

## The files

| File | What it is |
|------|-----------|
| `simple_transformer.py` | The whole model: math helpers, the Transformer, training, saving. Run it to train. |
| `use_saved_model.py` | Loads the saved model and generates text + shows prediction probabilities. |
| `saved_models/transformer.pkl` | The trained model, saved with `pickle` (just data — a plain dictionary). |

## How to run it

```bash
python simple_transformer.py   # trains the model, saves it (~30-40 seconds)
python use_saved_model.py      # loads the model and uses it
```

## What it does

It learns to continue the repeating sentence `"hello world "`. Give it `hello`
and it produces `hello world hello world hello world ...`. Tiny task — but it
exercises every real part of a Transformer.

## The 4 stages of a Transformer

1. **Embedding** — each character becomes a list of numbers (a vector).
2. **Positional encoding** — add a "position stamp" so the model knows the
   order of the characters.
3. **Self-attention** — each character looks back at earlier characters and
   gathers context. This is the famous part. Every position makes a *Query*
   ("what am I looking for?"), a *Key* ("what do I contain?"), and a *Value*
   ("what info do I share?").
4. **Feed-forward** — a small neural net thinks about the gathered context.

Then the final vectors are turned into a probability for every possible next
character. **Training** repeatedly nudges all the numbers so the predictions
get better — that's the `loss_and_grad` + `adam_step` loop.

## Things to try

- Change `CONFIG` at the top of `simple_transformer.py` (`d_model`, `lr`,
  `n_epochs`) and re-run to see the effect.
- Change the training text from `"hello world "` to your own short phrase.
- In `use_saved_model.py`, try your own prompts.

## A note on the model

The pickle file is **only data** — the learned numbers in a dictionary. The
*code* that runs the model (`forward`, `generate`) lives in
`simple_transformer.py`, which is why `use_saved_model.py` imports from it.

This model has no LayerNorm and a single attention head, left out on purpose to
keep it readable. Real Transformers add those (and stack many layers), but the
core ideas are exactly what you see here.
