# 8 · Diffusion and score-based generation

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 8. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** How can learning to remove noise support generation from noise?

- **Learn:** a forward noising process, learned reverse transitions, denoising objectives, score functions, and conditional generation; preview image and multimodal applications.
- **Statistical lens:** stochastic processes, denoising estimation, and the score $\nabla_x\log p(x)$, which differs from a likelihood score with respect to model parameters.
- **Python practice:** simulate progressive Gaussian noising on a mixture distribution, then study a small learned denoiser and the effect of the noise schedule.
- **Self-check:** distinguish the known forward process from the fitted reverse process and separate training from sampling.
- **Reading:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Continuous-time formulations are an optional extension.
