# 1. Statistical targets and observation mechanisms

:::{admonition} Core idea - The target comes before the algorithm
:class: course-core

A dataset can support several questions, but each question needs its own loss, population, and evaluation unit. Decide whether the output is a class decision, a probability, or an inferential statement before judging the method. A small training error answers none of these questions on its own.
:::

*Topics of Statistical Learning · Week 1*

## The question before the algorithm

A statistical learning problem starts with a target and an observation model. Before fitting a classifier, ask what will be predicted, for whom it will be predicted, which information is observed, and how errors will be evaluated. These questions become essential when a positive label is recorded selectively, two people supply a comparison instead of a score, or the data determine which hypothesis will be tested.

This chapter establishes notation used throughout the course. By the end, you should be able to distinguish population risk from training error, explain why an independent test sample is useful, identify the sampling unit, and describe information that an observed dataset does not contain.

## Population risk and the prediction target

Let $(X,Y)\sim P$ describe a future observation from the target population. A prediction rule $g$ maps features to an action or score, and a loss $\ell(g(X),Y)$ describes its consequence. The **population risk** is

$$
R_P(g)=E_P\{\ell(g(X),Y)\}.
$$

Both the distribution and the loss are part of the problem. For binary classification, zero–one loss evaluates class decisions. Log loss and Brier loss evaluate probability predictions. For regression, squared error targets a conditional mean; absolute error targets a conditional median. A method that is best for one loss need not be best for another.

For squared error, write $m(x)=E(Y\mid X=x)$. Conditioning on $X$ gives

$$
R_P(g)=E_P\{(g(X)-m(X))^2\}+E_P\{\operatorname{Var}(Y\mid X)\}.
$$

The cross term vanishes because $E\{Y-m(X)\mid X\}=0$. The first term is prediction error relative to the optimal conditional mean. The second is irreducible under the available features and this loss. Adding a useful feature can change both the target function and the remaining conditional variance.

:::{admonition} Worked example 1.1 — A class decision and a probability prediction
:class: course-example

Suppose $P(X=0)=0.6$, $P(X=1)=0.4$, and

$$
P(Y=1\mid X=0)=0.2,\qquad P(Y=1\mid X=1)=0.8.
$$

Under equal-cost zero–one loss, the optimal decisions are $g(0)=0$ and $g(1)=1$. Their risk is $0.6(0.2)+0.4(0.2)=0.20$. Always predicting zero has risk $P(Y=1)=0.44$.

For Brier loss $(q-Y)^2$, the optimal probability scores are instead $q(0)=0.2$ and $q(1)=0.8$. Their risk is $E\{\eta(X)(1-\eta(X))\}=0.16$, where $\eta(x)=P(Y=1\mid X=x)$. Using the hard class decisions as probability scores gives Brier risk $0.20$. A good classification rule therefore need not provide the most informative probabilities.
:::

## Learning a rule and evaluating it

Given independent training observations $D_n=\{(X_i,Y_i)\}_{i=1}^n$, the empirical risk is

$$
\widehat R_n(g)=\frac1n\sum_{i=1}^n\ell(g(X_i),Y_i).
$$

An empirical risk minimizer chooses $\widehat g\in\arg\min_{g\in\mathcal G}\widehat R_n(g)$, possibly with a regularization penalty. A fixed rule's empirical risk is unbiased for its population risk under the stated sampling model. That statement cannot simply be transferred to a rule selected using those same observations.

For exact empirical risk minimization over a fixed class, assuming the expectations exist,

$$
E\{\widehat R_n(\widehat g)\}
\leq \inf_{g\in\mathcal G}R_P(g)
\leq E\{R_P(\widehat g)\}.
$$

The first inequality follows by comparing the empirical minimizer with each fixed candidate before taking expectations. It describes average optimism; it does not assert that training error is smaller in every realization.

An independent test sample from $P$ provides a simpler evaluation statement:

$$
E\{\widehat R_{\mathrm{test}}(\widehat g)\mid D_n\}=R_P(\widehat g).
$$

Conditional on training, the fitted rule is fixed and the test observations remain fresh. If the test results are repeatedly used to tune the method, this interpretation changes. A practical experiment separates fitting, tuning, and final assessment, and records which decisions use each sample.

## Which population and which sampling unit?

An observation may be a person, document, prompt, comparison, or time point. Rows need not be independent units. Several ratings of one prompt can share difficulty; several comparisons can share an annotator. Evaluation and resampling should respect the scientific sampling unit and the target of generalization.

A change in the target population also changes risk. A class imbalance is not merely a display choice: it determines how class-specific errors are weighted. Report whether the target is the observed population, a future population, or a deliberately balanced population.

:::{admonition} Worked example 1.2 — The same classifier in a different population
:class: course-example

A classifier has sensitivity $0.8$ and specificity $0.9$. In a population with positive prevalence $\pi=0.2$, its zero–one risk is

$$
0.2(1-0.8)+0.8(1-0.9)=0.12.
$$

If the prevalence becomes $0.6$ while the class-conditional feature distributions remain the same, the same classifier has risk $0.6(0.2)+0.4(0.1)=0.16$. Its balanced error is $\{0.2+0.1\}/2=0.15$ in either population.

The change does not require the classifier to have deteriorated within either class. It reflects a different target distribution. Under the stated common-class-conditional assumption, source observations could be weighted by $0.6/0.2=3$ for positives and $0.4/0.8=0.5$ for negatives to evaluate the new population risk. Without that assumption, these two weights are not sufficient.
:::

## Identification, estimation, and guarantees

A target is **identified** when the observed-data distribution determines it uniquely within the assumed model. Estimation asks how to recover an identified quantity from finitely many observations. Numerical optimization then asks how well a computational procedure solves the chosen estimation problem. More data or a better optimizer cannot repair a target that the observation model leaves unidentified.

For example, an unlabelled feature distribution alone does not generally determine which observations belong to a scientifically named positive class. PU and UU learning add information through sampling assumptions, class proportions, or selected labels. We will state this information before deriving an empirical objective.

The course also uses several different guarantees:

| Question | Example target or criterion |
|---|---|
| How well will a rule predict? | Population risk or excess risk |
| How well does an estimated score recover a parameter? | Bias, variance, or interval coverage |
| How costly were sequential decisions? | Regret relative to a stated comparator |
| Is a selected hypothesis tested validly? | Type I error under the appropriate selection event |

A low prediction error does not establish a valid selected-hypothesis test. Conversely, a correctly calibrated test need not imply useful prediction. We will keep the target, its assumptions, and the scope of its guarantee together.

## A decision can change when the cost changes

Suppose a false positive costs $c_{\mathrm{FP}}$ and a false negative costs $c_{\mathrm{FN}}$, with correct decisions costing zero. At a feature value with positive probability $\eta(x)$, predicting positive has conditional cost $c_{\mathrm{FP}}\{1-\eta(x)\}$; predicting negative has cost $c_{\mathrm{FN}}\eta(x)$. The optimal threshold is therefore

$$\eta(x)>\frac{c_{\mathrm{FP}}}{c_{\mathrm{FP}}+c_{\mathrm{FN}}}.$$

If missed positives cost four times as much as false alarms, the threshold is $0.2$, not $0.5$. A case with probability $0.3$ is then classified positive even though negative is the more likely class. This is an illustrative decision calculation, not a new fitted-model result.

:::{admonition} Analysis check - align the score with the decision
:class: course-method

Write down the target population, prediction unit, loss, and available information at prediction time. Evaluate the declared cost as well as any probability score. A method can improve a probability forecast without changing every class decision, or improve a cost-sensitive decision while reducing ordinary accuracy.
:::

## Laboratory — Training error and future prediction risk

The standalone experiment fits polynomials of degrees 1, 2, and 8 to noisy quadratic data. It repeats the entire training procedure and evaluates conditional prediction risk on a fixed, fine midpoint grid. The evaluation uses the known simulation mean and adds the noise variance; it is an approximation to the integral over the specified uniform feature distribution. The true mean is available here only because this is a controlled simulation.

Inspect average training error, target risk, and Monte Carlo standard errors. The quadratic model matches the mean specification; the flexible model may fit noise. The experiment does not claim that one degree is universally optimal.

[Download lab](../_static/labs/01-statistical-targets.R)

:::{literalinclude} ../../labs/01-statistical-targets.R
:language: r
:::

### Executed laboratory output

:::{include} ../_generated/01-statistical-targets-output.md
:::

## Exercises

1. Suppose a false positive costs $c_{\mathrm{FP}}$ and a false negative costs $c_{\mathrm{FN}}$, both positive. Derive the optimal decision threshold for $\eta(x)=P(Y=1\mid X=x)$.
2. Explain why the test-risk equality above can fail when several candidate models are compared using the test sample and the best-looking one is reported.
3. Modify the laboratory to increase the training sample size. Which parts of the squared-error decomposition should change? Which part should not?
4. Each of 100 prompts has 20 ratings. What additional information is needed before treating 2,000 rows as 2,000 independent observations for uncertainty assessment?

:::{admonition} Worked solutions
:class: dropdown

**1.** Predicting one has conditional expected loss $c_{\mathrm{FP}}(1-\eta)$, while predicting zero has loss $c_{\mathrm{FN}}\eta$. Predict one when $\eta>c_{\mathrm{FP}}/(c_{\mathrm{FP}}+c_{\mathrm{FN}})$; either decision is optimal at equality.

**2.** The selected model now depends on test outcomes. Conditioning only on the original training data no longer makes the reported rule independent of the test sample. A separate final sample, or an evaluation procedure that accounts for this selection, is needed for the original interpretation.

**3.** More training observations can reduce error in estimating the mean function. They do not change the data-generating conditional variance. A misspecified linear model retains approximation error even with arbitrarily many observations; the experiment's irreducible variance remains $0.7^2$.

**4.** We need the dependence structure, how annotators are shared, and whether the target is a new rating, prompt, annotator, or some combination. Prompt-level dependence can invalidate row-wise uncertainty estimates; crossed prompt–annotator dependence may require more than a one-way resampling scheme.
:::

## Primary readings

- [Shalev-Shwartz and Ben-David, *Understanding Machine Learning: From Theory to Algorithms*](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/): selected sections on learning models, empirical risk, and generalization.
- [Bartlett, Jordan, and McAuliffe (2006), “Convexity, Classification, and Risk Bounds”](https://doi.org/10.1198/016214505000000907): an advanced connection between surrogate losses and classification risk.

**Next:** Chapter 2 makes the observation mechanism explicit when a recorded label is incomplete, ambiguous, or corrupted.
