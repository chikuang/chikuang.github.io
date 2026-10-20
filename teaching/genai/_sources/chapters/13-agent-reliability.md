# 13 · Agent reliability and experimental design

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 13. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** Does an agent improve outcomes enough to justify its additional complexity?

- **Learn:** trajectory-level evaluation, failure taxonomies, repeated trials, cost/latency tradeoffs, memory ablations, prompt injection through retrieved text, and human approval for consequential actions.
- **Statistical lens:** dependence within a task trajectory, factorial comparisons, uncertainty in success rates, and adaptive selection bias.
- **Python practice:** compare a direct model call, a fixed workflow, and the agent on the same task set and budget. Inject harmless tool failures; log recovery, loop counts, and stopping behavior. Treat external documents as data rather than tool instructions.
- **Self-check:** separate task failure, tool failure, unsupported answers, and budget exhaustion. Add multiple agents only as an optional ablation after establishing a single-agent baseline.
