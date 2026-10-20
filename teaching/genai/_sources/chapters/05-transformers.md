# 5 · Assembling the transformer

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 5. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** How does one attention calculation become a sequence model?

- **Learn:** residual connections, normalization, feed-forward layers, positional encodings, encoder/decoder distinctions, and a decoder-only forward pass.
- **Statistical lens:** a composition of fitted feature transformations with shared parameters and a conditional prediction head.
- **Python practice:** implement a tiny decoder block in PyTorch; track every tensor's dimensions and use a causal-mask check to show future tokens cannot influence earlier outputs.
- **Self-check:** draw the route from token IDs to next-token probabilities, and distinguish the original encoder-decoder transformer from a decoder-only language model.
