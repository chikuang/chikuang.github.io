# 9 · Fine-tuning, LoRA, and preference learning

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 9. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** How do we adapt an existing model, and whose preferences are represented?

- **Learn:** supervised fine-tuning, parameter-efficient updates, LoRA, instruction data, preference data, the RLHF pipeline, and the basic DPO idea.
- **Statistical lens:** transfer learning, regularization, low-rank parameter updates, and Bradley–Terry-style paired comparisons. Keep model quality separate from preference-label quality.
- **Python practice:** fit a toy paired-comparison model and compare a full matrix update with a low-rank update. Small-model fine-tuning is optional after the conceptual exercise.
- **Self-check:** explain what changes under prompting, RAG, and fine-tuning; identify label noise and selection bias in preference data.
- **Readings:** [LoRA](https://arxiv.org/abs/2106.09685) and [DPO](https://arxiv.org/abs/2305.18290).
