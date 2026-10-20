# 12 · Agents: tools, state, and decisions

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 12. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** What makes a system an agent, beyond producing a text response?

- **Learn:** a model–action–observation loop, tool schemas, state, memory, structured outputs, retries, and stopping rules; distinguish a fixed workflow from adaptive action selection.
- **Statistical lens:** sequential decision making, partial observation, expected utility, and the value of information. Learn the basic state/action/reward vocabulary without requiring a reinforcement learning course first.
- **Python practice:** build a single agent with two narrow tools, such as a calculator and document lookup. Validate arguments, log actions and observations, and enforce a small step budget.
- **Self-check:** explain when a fixed Python function or workflow is sufficient and why tool access does not imply reliable tool use.
- **Reading:** [ReAct](https://arxiv.org/abs/2210.03629); study observable actions and results without treating generated reasoning text as a faithful record of internal computation.
