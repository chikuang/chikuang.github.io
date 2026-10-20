# 11 · Evaluation, calibration, and uncertainty

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 11. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** What evidence would justify saying one system is better than another?

- **Learn:** task-specific metrics, held-out evaluation, data contamination, human annotation, LLM-as-a-judge limitations, calibration, abstention, and an introduction to conformal methods.
- **Statistical lens:** estimands, sampling units, paired comparisons, bootstrap uncertainty, clustered dependence, and distribution shift. Conformal coverage needs an explicit target and assumptions such as exchangeability; it is not an automatic guarantee of free-text truthfulness.
- **Python practice:** compare two fixed systems on the same held-out questions; bootstrap at the question level, inspect judge disagreement, and draw a risk-versus-coverage curve for abstention.
- **Self-check:** distinguish token probabilities, answer confidence, and factual correctness; state the population to which a reported interval applies.
- **Reading:** [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).
