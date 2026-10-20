# 4 · Attention from first principles

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 4. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** How can a representation select and combine information relevant to a query?

- **Learn:** queries, keys, values, scaled dot products, row-wise softmax, self-attention, cross-attention, causal masks, and multiple heads.
- **Statistical lens:** a query-dependent normalized weighted average; compare with kernel smoothing without claiming exact equivalence.
- **Python practice:** calculate a three-token attention example by hand, implement it in NumPy, verify dimensions and row sums, and plot how weights change when keys change.
- **Self-check:** explain the role of the mask and the $\sqrt{d_k}$ scaling; describe why attention weights alone are not a faithful explanation of a prediction.
- **Reading:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762), especially the architecture and attention equations.
