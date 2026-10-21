# Research project

The project develops one question far enough that another researcher can reproduce the result and understand the assumptions behind it. A strong project reproduces a meaningful result, establishes a fair baseline, and investigates one consequential extension or sensitivity. It need not invent a new algorithm or prove a new theorem.

## Choose a statistical target first

Write one sentence of the form: **“Under this observation mechanism, we will estimate, predict, optimise, or test this target, and evaluate success by this criterion.”** This keeps an interesting implementation from becoming a collection of unrelated experiments.

| Direction | Example research question | A useful sensitivity analysis |
|---|---|---|
| PU or UU learning | How accurately can a fixed hypothesis class estimate target risk with incomplete labels? | Misspecified class prior, feature-dependent positive selection, or shrinking mixture-prior gap |
| Preference models and design | Which comparison allocation best estimates a scientifically meaningful contrast? | Comparison graph, separation, prompt heterogeneity, or repeated annotators |
| Online learning and bandits | How does an algorithm balance information gathering and reward in a specified feedback model? | Horizon, reward gap, nonstationarity as an explicit model departure, or a different comparator |
| Inference after clustering | How does conditioning on a specified selection event affect calibration and power? | Signal strength, dimension, noise assumptions, or the selection procedure |

These are starting points. A project may combine topics when the connection is statistically explicit, such as comparison allocation through Fisher information or calibration after a data-dependent model choice.

## Milestones

### Week 6 · Proposal

Submit approximately two pages containing the question, observed data, target, assumptions, primary reference, baseline, and one planned extension. Identify the computational requirements and a minimal result that can be completed even if the extension proves difficult. This is **5%** of the proposed course assessment.

### Week 11 · Reproduction checkpoint

Provide working code, one reproduced table or figure, an explanation of any discrepancy, and a revised plan. Include the software environment and execution instructions. An honest diagnosis of an unsuccessful reproduction is useful evidence when it is specific and supported. This is **10%**.

### Week 15 · Presentation

Present for **12 minutes**, followed by questions. Explain the target before the algorithm, show the most informative evidence, and state one limitation. Every group member should be able to explain the assumptions and interpretation. This is **10%**.

### Final-assessment period · Report

Submit **8–10 pages**, excluding references and code, with a reproducible supplement. This is **20%**. The exact date is set for the offering. Independent-study readers can use the same structure as a research note.

## Required report structure

1. **Question and contribution.** What did you investigate, and why does the answer matter?
2. **Model and target.** Define the observed and latent variables, the data-generating or sampling mechanism, and the estimand or performance criterion.
3. **Method and guarantee.** Explain the method at the level needed to understand the result. State the relevant assumptions and distinguish a theorem from an empirical observation.
4. **Reproduction.** Identify the original result, what you matched, and any necessary departures from the reference implementation.
5. **Extension or sensitivity.** Change one consequential aspect of the setting. Explain the predicted effect and compare it with the observed result.
6. **Evaluation and uncertainty.** Use suitable baselines, independent or appropriately structured evaluation, repeated runs where relevant, and uncertainty measures.
7. **Interpretation and limitations.** State what the evidence supports and what it leaves unresolved.
8. **Reproducibility.** Supply code, versions, seeds, data provenance, execution commands, and the computational budget.

## Report rubric

| Criterion | % | Evidence of strong work |
|---|---:|---|
| Statistical formulation | 25% | A precise target, observation mechanism, and consequential assumptions |
| Methodological reasoning | 25% | A correct derivation, theorem interpretation, or critique tied to the question |
| Empirical design and uncertainty | 30% | Fair comparisons, meaningful sensitivity, appropriate evaluation, and uncertainty |
| Reproducibility and communication | 20% | Executable work, clear figures and tables, documented provenance, and a coherent explanation |

## Scope decisions that help

An assignment may supply the baseline for the project, provided the project adds a clearly identified question and deeper analysis. A small simulated setting often makes the assumptions easier to examine than a large application with an unclear observation process. Real data can add value when their sampling and annotation mechanisms are sufficiently documented.

Use ordinary laptop resources when possible. No paid API service or GPU is required. If the project involves modern language-model preferences, a finite collection of precomputed comparisons can support a substantial statistical investigation without training a model.

## A completed study should answer these questions

Could someone reproduce the principal figure from a fresh session? Does the learner receive only the information permitted by the model? Was the final evaluation used to choose the method? Does the claimed guarantee concern prediction, regret, parameter estimation, selective validity, or simultaneous error control? What happens when one key assumption changes?

The final chapter, [Research synthesis](chapters/14-research-synthesis.md), works through this audit in a small reproducible investigation.
