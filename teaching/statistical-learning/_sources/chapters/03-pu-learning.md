# 3. Learning from positives and unlabelled observations

:::{admonition} Core idea - Subtract contamination, not observations
:class: course-core

An unlabelled sample from the full feature marginal contains positive and negative cases. The PU identity removes the positive contribution to the negative-class loss **in expectation**. It does not identify which individual unlabelled rows are negative, and it depends on representative positives and the specified prior.
:::

*Topics of Statistical Learning · Week 3*

## What is missing?

In positive–unlabelled learning, some observations are known to belong to a positive class, while an unlabelled collection contains both classes. Examples include confirmed relevant documents and a broader document collection, or confirmed examples of a phenomenon together with background observations. An unlabelled observation has an unknown class, not a known negative class.

The main task is to express a target classification risk using distributions from which data are actually available. The derivation is short, but its sampling assumptions are consequential. We first use an explicit two-sample model, then distinguish it from a single dataset in which only some positive labels are revealed.

## The two-sample model

Let $Y\in\{+1,-1\}$, let $\pi=P(Y=+1)\in(0,1)$ be the target class prior, and write

$$
P_+=P(X\mid Y=+1),\qquad P_-=P(X\mid Y=-1).
$$

The target feature marginal is

$$
P_U=\pi P_+ +(1-\pi)P_-.
$$

The introductory sampling scheme provides independent observations $X^+_1,\ldots,X^+_{n_+}\sim P_+$ and $X^U_1,\ldots,X^U_{n_U}\sim P_U$. These are representative samples from the specified positive distribution and the full target feature marginal. For now, $\pi$ is known.

The ratio $n_+/(n_++n_U)$ is controlled by data collection and is not an estimator of the target prevalence in this two-sample design. We may deliberately collect many confirmed positives while the target population contains relatively few positives.

## Deriving the PU risk identity

For a score $g$ and loss $\ell$, define $\ell_\pm(X)=\ell(g(X),\pm1)$. The ordinary population risk is

$$
R(g)=\pi E_+\ell_+ +(1-\pi)E_-\ell_-.
$$

The unavailable negative expectation can be removed by applying the mixture relation to $\ell_-$:

$$
E_U\ell_-=\pi E_+\ell_-+(1-\pi)E_-\ell_-.
$$

Substituting gives

$$
\boxed{R(g)=\pi E_+\ell_+ + E_U\ell_- -\pi E_+\ell_- .}
$$

Each expectation on the right can be estimated from the observed samples. The subtraction removes the positive-class contribution that would otherwise be counted when unlabelled observations are temporarily evaluated as negative.

The identity applies to a loss with finite relevant expectations. It is not restricted to zero–one loss. Its empirical version is

$$
\widehat R_{\mathrm{PU}}(g)
=\frac{\pi}{n_+}\sum_i\ell(g(X_i^+),+1)
+\frac1{n_U}\sum_j\ell(g(X_j^U),-1)
-\frac{\pi}{n_+}\sum_i\ell(g(X_i^+),-1).
$$

For a fixed $g$ and the stated known-prior sampling model, this estimator is unbiased. A model fitted by minimizing it still needs generalization analysis or independent evaluation. The fitted model is not fixed with respect to its training sample.

:::{admonition} Worked example 3.1 — Removing positive contamination
:class: course-example

Suppose $\pi=0.3$, sensitivity is $0.8$, and the false-positive rate is $0.1$. Under zero–one loss,

$$
E_+\ell_+=0.2,\qquad E_+\ell_-=0.8,
$$

while $E_U\ell_-=P\{g(X)=+1\}=0.3(0.8)+0.7(0.1)=0.31$. The PU identity yields

$$
R(g)=0.3(0.2)+0.31-0.3(0.8)=0.13.
$$

The direct fully labelled calculation is $0.3(0.2)+0.7(0.1)=0.13$. The observed marginal predicted-positive rate $0.31$ includes true positives as well as false positives. It is not the negative-class error contribution; subtracting $0.24$ leaves the required contribution $0.07$.
:::

## A logistic objective that can be implemented directly

Take $g_\beta(x)=\beta_0+x^\mathsf T\beta_1$ and logistic losses

$$
\ell(g,+1)=\log(1+e^{-g}),\qquad
\ell(g,-1)=\log(1+e^g).
$$

Since $\log(1+e^{-g})-\log(1+e^g)=-g$, the empirical PU risk simplifies to

$$
\widehat R_{\mathrm{PU}}(\beta)
=\frac1{n_U}\sum_j\log(1+e^{g_\beta(X_j^U)})
-\frac{\pi}{n_+}\sum_i g_\beta(X_i^+).
$$

For this linear score, the objective is convex. A positive quadratic penalty can stabilize estimation and ensure a finite minimizer; the penalty is a chosen regularizer rather than an unbiased component of the target risk. More flexible score classes introduce different computational and statistical issues. Chapter 4 examines the negative empirical risks that can arise from the subtraction.

Implement the softplus function stably as $\max(z,0)+\log(1+e^{-|z|})$, rather than forming a potentially overflowing $e^z$.

## A different design: labels revealed within one sample

Now suppose observations are sampled from the target population and $S=1$ indicates that a positive label is recorded. We assume $P(S=1\mid Y=-1,X)=0$. Under **selected completely at random**, or SCAR,

$$
P(S=1\mid Y=+1,X=x)=c
$$

for a constant $c>0$. Then $P(S=1\mid X=x)=cP(Y=+1\mid X=x)$, and the labelled positives have distribution $P_+$. However, the observations with $S=0$ are not distributed as the full feature marginal.

:::{admonition} Worked example 3.2 — Unlabelled rows are not always the marginal sample
:class: course-example

Let $\pi=0.3$ and $c=0.4$. The probability of an observed positive label is $0.12$. Among observations without a recorded label,

$$
P(Y=+1\mid S=0)=\frac{\pi(1-c)}{1-\pi c}
=\frac{0.18}{0.88}\approx0.2045.
$$

Using the classifier from Example 3.1, the predicted-positive rate in this subset is $0.2045(0.8)+0.7955(0.1)\approx0.2432$. If we incorrectly substitute that rate for $E_U\ell_-$, the PU expression becomes about $0.0632$, rather than the target risk $0.13$.

All feature vectors in the original population sample can represent the marginal distribution. The labelled-positive and full-feature averages then share observations, so their dependence matters for variance calculations. Alternatively, use a sampling model and risk expression designed specifically for the observed-label partition.
:::

## What the assumptions do not supply

SCAR makes the recorded positives representative of the positive class. It does not by itself identify an unknown $c$ or $\pi$. In the observed posterior $P(S=1\mid X)=c\eta(X)$, different pairs $(c,\eta)$ can produce the same product without additional restrictions. Likewise, a positive distribution and its mixture with an unrestricted negative distribution need not uniquely determine an unknown mixture proportion.

Known prevalence, a separately collected validation sample, or additional identifiable structure can supply information. Any class-prior estimator introduced later must state its own assumptions. Treating a supplied prior as known is appropriate for an introductory simulation, provided the simulation also examines sensitivity to an incorrect value.

## Plan the positive and unlabelled sample sizes

For a fixed rule and known $\pi$, independent positive and marginal samples give

$$\operatorname{Var}(\widehat R_{\mathrm{PU}})
=\frac{\pi^2}{n_+}\operatorname{Var}_+(\ell_+-\ell_-)
+\frac{1}{n_U}\operatorname{Var}_U(\ell_-).$$

The same positive observations appear in both positive-class averages, so their covariance is already included in $\operatorname{Var}_+(\ell_+-\ell_-)$. Treating these two averages as independent generally gives the wrong uncertainty calculation. Once the second term is small, collecting more unlabelled cases does little to reduce uncertainty dominated by the positive sample.

:::{admonition} Analysis check - distinguish the two sampling designs
:class: course-method

Use this variance expression for independent positive and full-marginal samples and a fixed rule. In a single partially labelled population sample, averages may share observations. For a fitted rule, training, tuning, and uncertainty need additional analysis; the displayed calculation is not a fitted-model confidence interval.
:::

## Laboratory — A linear PU classifier

The lab fits a regularized logistic PU objective to independent positive and marginal samples. It compares the resulting rule with a naive positive-versus-unlabelled fit, a fully labelled oracle benchmark, and the known Bayes score for the simulation. All evaluation uses a new labelled test sample. The final section verifies the SCAR subset calculation using a separate population sample.

[Download lab](../_static/labs/03-pu-learning.R)

:::{literalinclude} ../../labs/03-pu-learning.R
:language: r
:::

### Executed laboratory output

:::{include} ../_generated/03-pu-learning-output.md
:::

## Exercises

1. Derive the PU identity for squared probability loss as well as for zero–one loss. Which step changes?
2. In the two-sample design, why does increasing $n_+$ while holding the target population fixed not increase $\pi$?
3. Show the logistic simplification above and derive the gradient for a linear score with design vector $z(x)=(1,x^\mathsf T)^\mathsf T$.
4. In a single SCAR sample, derive $P(Y=+1\mid S=0)$ and state when it equals $\pi$.

:::{admonition} Worked solutions
:class: dropdown

**1.** The mixture identity holds for any integrable function, so no algebraic step changes. Substitute the chosen loss for $\ell_-$ and then for the full risk. The interpretation and optimization properties of the resulting objective do change with the loss.

**2.** The investigator controls the numbers sampled from $P_+$ and $P_U$. The target prevalence belongs to the population represented by $P_U$, not to the artificially assembled training collection.

**3.** The difference of the two logistic losses is $-g$. With quadratic penalty $\lambda\|\beta\|^2$, the gradient is $n_U^{-1}\sum_j\operatorname{logit}^{-1}(z_j^\mathsf T\beta)z_j-\pi n_+^{-1}\sum_i z_i+2\lambda\beta$.

**4.** Bayes' rule gives $\pi(1-c)/(1-\pi c)$. For $0<\pi<1$, equality with $\pi$ requires $c=0$, in which case no labels are observed. The degenerate all-negative or all-positive populations also require separate treatment. For a useful nondegenerate SCAR sample with $c>0$, the unlabelled subset has a smaller positive proportion.
:::

## Primary readings

- [Elkan and Noto (2008), “Learning Classifiers from Only Positive and Unlabeled Data”](https://cseweb.ucsd.edu/~elkan/posonly.pdf): the observed-label mechanism and SCAR relationship.
- [du Plessis, Niu, and Sugiyama (2015), “Convex Formulation for Learning from Positive and Unlabeled Data”](https://proceedings.mlr.press/v37/plessis15.html): risk rewriting and convex formulations.
- [Kiryo et al. (2017), “Positive-Unlabeled Learning with Non-Negative Risk Estimator”](https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html): the empirical-risk formulation and its practical limitations.
