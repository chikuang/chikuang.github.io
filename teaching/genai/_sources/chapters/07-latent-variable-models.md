# 7 · Latent variables, VAEs, GANs, and flows

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 7. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** What alternatives to sequential factorization can represent a complex distribution?

- **Learn:** the encoder/decoder view of a VAE, the evidence lower bound (ELBO), reparameterization, and a conceptual comparison with adversarial learning and invertible flows.
- **Statistical lens:** latent-variable models, variational inference, density-ratio ideas, and change of variables. Compare objectives, sampling, and access to likelihoods.
- **Python practice:** derive the ELBO and implement a small VAE on synthetic two-dimensional data. Treat GANs and flows as comparison topics on the first pass.
- **Self-check:** separate the latent prior, approximate posterior, and observation model; explain why matching a few summary statistics cannot establish distributional equivalence.
- **Reading:** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
