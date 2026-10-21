# Assignments and paper discussion

These three investigations connect the derivations, experiments, and interpretation developed in the chapters. They may be completed independently. For a taught offering, the proposed allocation is **15% per assignment**, with the final calendar and submission arrangements supplied by the instructor.

Use the [report template](_static/downloads/report-template.qmd) or an equivalent reproducible format. A typical submission is **5–7 pages**, excluding references and code. Prefer a small number of carefully explained figures to a large collection of unexplained simulation outputs.

## Assignment 1 · Learning without complete labels

**Suggested timing:** begin in Week 3; complete in Week 5.  
**Preparation:** [Chapters 1–5](chapters/01-statistical-targets.md).

### Question

When can we estimate and minimise classification risk without a fully labelled training sample, and how does the answer depend on the observation mechanism?

### Investigation

1. **Derive the PU risk.** Define $P_+$, $P_-$, the marginal feature distribution $P_U$, the target class prior $\pi$, and a loss. Derive the PU risk identity by substituting the mixture representation. State why your positive sample is representative of $P_+$ and why your unlabelled sample represents $P_U$.
2. **Build a controlled experiment.** Use a binary Gaussian model or another fully specified distribution. Generate independent training and labelled evaluation samples. Compare a supervised oracle, a naive method that treats unlabelled cases as negative, an unbiased PU estimate, and its non-negative correction. Keep the hypothesis class and optimisation budget comparable. An oracle is a simulation benchmark with extra information, not an available PU procedure.
3. **Probe one assumption.** Vary either the supplied class prior or the positive-label selection mechanism. Explain which identity or estimation argument is affected. If you introduce feature-dependent selection, do not apply a SCAR identity as though the observed positive sample remained representative.
4. **Study UU conditioning.** For a fixed classifier, generate two unlabelled mixtures with shared class-conditionals and known different priors. Estimate its target risk using the UU identity. Change the gap between the mixture priors while keeping the target prior fixed. Report bias, variability, and Monte Carlo uncertainty. Explain what happens when the two priors are equal.
5. **Write a conclusion.** Distinguish identification, variance, optimisation, and evaluation error. Explain one result that initially surprised you.

### Required evidence

Include the risk derivation, the observation-mechanism diagram or table, one comparison of predictive risk, one sensitivity figure, and a compact table of Monte Carlo uncertainty. Use a fixed classifier in the UU subexperiment to isolate risk-estimation variability from model-selection effects.

### Evaluation criteria

| Criterion | % |
|---|---:|
| Correct target, mechanism, and derivation | 30% |
| Fair experimental design and independent evaluation | 30% |
| Sensitivity analysis and statistical interpretation | 25% |
| Reproducibility and communication | 15% |

## Assignment 2 · Comparisons and sequential decisions

**Suggested timing:** begin in Week 7; complete in Week 10.  
**Preparation:** [Chapters 6–10](chapters/06-pairwise-preferences.md).

### Question

What do pairwise comparisons identify, and how does restricted or adaptive feedback change the design and evaluation of a learning procedure?

### Investigation

1. **Identify the preference model.** Write the Bradley–Terry likelihood and choose an identification constraint. Explain the difference between the connectivity needed to identify contrasts and the outcome pattern needed for a finite unpenalised estimate.
2. **Construct three comparison datasets.** Use a connected graph with wins in both directions, a disconnected graph, and a connected graph with separation. Examine the objective, estimates, and numerical behaviour. If you use a penalty, state the penalised target and show how the result changes with penalty strength.
3. **Evaluate a comparison design.** Compare two pair-allocation schemes with the same comparison budget. State whether the goal is predicting pair outcomes, estimating a particular contrast, or recovering a ranking. Use the information structure from [Chapter 7](chapters/07-contextual-preferences.md) to motivate the design. Simulate repeated datasets and evaluate the chosen objective.
4. **Implement a reward-bandit comparison.** In a separate stationary Bernoulli bandit model, compare uniform exploration and UCB. Initialise each arm once. Record action counts and cumulative pseudo-regret against the best fixed mean. The simulator may use true means for evaluation; the learner may not use them to choose an arm.
5. **Connect the settings carefully.** Explain what would have to change to turn the comparison problem into a duelling-bandit problem. Specify a winner criterion and a feedback model. A reward-bandit implementation is not automatically a duelling algorithm.

### Required evidence

Include the model and graph examples, an analysis of finite estimation, a budget-matched preference-design comparison, repeated bandit trajectories with uncertainty, and a paragraph explaining the two different feedback structures. If observations share an annotator or prompt, explain the dependence and evaluation unit.

### Evaluation criteria

| Criterion | % |
|---|---:|
| Preference formulation, identification, and estimation | 30% |
| Comparison-design reasoning and experiment | 25% |
| Sequential implementation and regret evaluation | 30% |
| Reproducibility and communication | 15% |

## Assignment 3 · Inference after clustering

**Suggested timing:** begin in Week 12; complete in Week 14.  
**Preparation:** [Chapters 11–13](chapters/11-selective-inference.md).

### Question

How does using the data to choose clusters change the reference distribution of a test of their mean contrast?

### Investigation

1. **State the model and selected null.** Begin with independent Gaussian observations, a known common noise scale, and a single homogeneous cloud. Specify the dimension, sample size, clustering procedure, number of clusters, and tested pair. Write the null in terms of the selected groups' average observation-specific means.
2. **Freeze the selection procedure.** Use one documented hierarchical-clustering method supported by the authors' implementation. Keep the distance, linkage, preprocessing, cut rule, and pair-selection rule fixed. Record every step that uses the data.
3. **Compare two tests.** Compute a naive fixed-group test and the appropriate selective test from `clusterpval`. Reuse the same dataset for the comparison. Report the number of successful computations and any numerical failures. Do not silently exclude difficult datasets.
4. **Repeat under the null.** Use at least 500 repetitions when computationally feasible. Plot p-value distributions and report rejection rates with binomial or Monte Carlo uncertainty. Explain why apparent cluster separation in one sample is insufficient evidence of distinct population components.
5. **Investigate one extension.** Change separation under a stated alternative, the dimension, or an assumption about the noise. Distinguish a power experiment under the model from a robustness experiment outside it. Do not claim exact validity for an estimated-scale or non-Gaussian extension unless the chosen method supports that claim.
6. **Audit the inferential statement.** Explain why a selected-pair p-value does not by itself control errors over every pair, and why clustering stability answers a different question.

### Required evidence

Include an explicit selected null, an algorithm specification, an executed author-package call, null rejection results with uncertainty, and a failure or sensitivity analysis. The elementary Gaussian selection demo is useful for understanding the mechanism, but it does not replace the required clustering-inference computation.

### Evaluation criteria

| Criterion | % |
|---|---:|
| Model, selected target, and conditioning argument | 30% |
| Faithful implementation and selection audit | 25% |
| Simulation design, uncertainty, and sensitivity | 30% |
| Reproducibility and communication | 15% |

## Paper discussion and reading note

Choose a primary paper in consultation with the instructor for a taught offering, or use the [reading guide](readings.md) for independent study. Prepare a two-page note and a short discussion organised around six questions:

1. What is observed, and what is the target?
2. Which assumptions identify the target?
3. What does the main theorem guarantee, and for what procedure?
4. Which implementation choices matter to the theorem?
5. What is the most convincing experiment, and what comparison is missing?
6. What would you change or investigate next?

The proposed 10% discussion component evaluates accuracy, explanation of assumptions, critical analysis, and engagement with other participants. A concise derivation or a carefully chosen counterexample often contributes more than a long summary of every section.
