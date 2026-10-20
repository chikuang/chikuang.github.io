# 6 · Language model training, prompting, and decoding

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 6. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** What is learned in the weights, and what changes when we provide a prompt?

- **Learn:** next-token prediction, teacher forcing, context windows, pretraining versus inference, in-context examples, greedy decoding, temperature, top-k, and top-p.
- **Statistical lens:** conditional likelihood, distribution shift, and transformations of a sampling distribution. In-context behavior can be studied statistically without assuming the model performs exact Bayesian inference.
- **Python practice:** train a very small character model or use a small pretrained model; vary one decoding choice at a time on fixed prompts and repeat runs.
- **Self-check:** explain why perplexity is not a universal measure of usefulness and why prompting is not parameter estimation by gradient descent at inference time.
