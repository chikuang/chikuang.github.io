# 4. PU estimation, overfitting, and sensitivity

*Topics of Statistical Learning · Week 4*

## An unbiased objective can still be difficult to minimize

The PU identity from Chapter 3 expresses risk as the sum of a positive-class contribution and a difference of two expectations. Population non-negativity does not prevent the empirical difference from becoming negative. A flexible score can exploit this difference by assigning very large positive scores to the observed positives without behaving similarly on future positives.

This chapter separates three questions: whether a risk estimator is unbiased for a fixed rule, whether minimizing it produces a useful fitted rule, and whether the sampling assumptions represent the application. These are related questions, but none is a substitute for the others.

## The non-negative risk correction

Write the empirical risk as $\widehat A(g)+\widehat B(g)$, where

$$
\widehat A(g)=\pi\widehat E_+\ell_+,
\qquad
\widehat B(g)=\widehat E_U\ell_- -\pi\widehat E_+\ell_-.
$$

For a non-negative loss, the population quantity $B(g)=(1-\pi)E_-\ell_-$ is non-negative. The **non-negative PU** estimator replaces its empirical version by

$$
\widehat R_{\mathrm{nnPU}}(g)
=\widehat A(g)+\max\{0,\widehat B(g)\}.
$$

The correction applies to the estimated negative-class contribution, not simply to the total empirical risk. It prevents that contribution from rewarding the learner for becoming arbitrarily negative. The regularizer, score class, and numerical optimizer remain separate choices.

:::{admonition} Worked example 4.1 — Where the correction acts
:class: course-example

Use zero–one loss and a supplied prior $\pi=0.4$. Suppose the positive sample has error rate $0.10$, so the same observations evaluated as negative have loss $0.90$. Suppose $30\%$ of the marginal unlabelled sample is predicted positive. Then

$$
\widehat A=0.4(0.10)=0.04,\qquad
\widehat B=0.30-0.4(0.90)=-0.06.
$$

The unbiased empirical expression is $-0.02$, whereas the non-negative expression is $0.04$. Clipping the total expression at zero would instead give zero, a different estimator. None of these numbers reveals the unknown true risk of this particular fitted classifier without further assessment.
:::

## Bias, variance, and consistency are different properties

For a fixed rule under the known-prior model,

$$
\widehat R_{\mathrm{nnPU}}-\widehat R_{\mathrm{PU}}
=(-\widehat B)_+\geq0.
$$

Consequently, its expectation is at least the true risk whenever the original estimator is unbiased and the expectations exist. The correction generally introduces finite-sample upward bias. It can nevertheless improve mean squared error or the quality of the learned model. Such improvements depend on conditions; non-negative correction does not dominate every alternative in every finite sample.

For a fixed $g$, laws of large numbers for the relevant empirical averages and continuity of $b\mapsto\max(0,b)$ imply consistency when $B(g)\geq0$. A statement about a learned $\widehat g$ also needs control over the function class and optimization. Pointwise consistency for each fixed rule does not automatically justify minimization over an unrestricted set of rules.

Under independent positive and marginal samples, the variance of the unbiased fixed-rule estimator is

$$
\operatorname{Var}(\widehat R_{\mathrm{PU}})
=\frac{\pi^2}{n_+}\operatorname{Var}_+(\ell_+-\ell_-)
+\frac1{n_U}\operatorname{Var}_U(\ell_-).
$$

The two positive-class averages use the same observations. Keeping their covariance through $\ell_+-\ell_-$ is essential. This formula changes if the samples overlap or the prior is estimated. For the non-negative estimator, the maximum creates an additional nonlinearity, so a normal approximation should not be assumed without checking its justification.

## Sensitivity to the supplied class prior

Suppose the target has true prior $\pi$, but the population correction uses $\widetilde\pi$. For a fixed rule and unchanged positive and marginal distributions,

$$
R_{\widetilde\pi}^{\mathrm{correction}}(g)-R(g)
=(\widetilde\pi-\pi)E_+(\ell_+-\ell_-).
$$

This is a misspecified correction for the same target distribution. It should not be confused with deliberately changing the target population's prevalence while also using the appropriate target distribution.

:::{admonition} Worked example 4.2 — An incorrect prior can make the risk look smaller
:class: course-example

A classifier has sensitivity and specificity both equal to $0.8$, and the target prior is $0.3$. Its true risk is $0.20$. The relevant quantities are $E_+\ell_+=0.2$, $E_+\ell_-=0.8$, and $E_U\ell_-=0.38$.

The correct expression is $0.3(0.2)+0.38-0.3(0.8)=0.20$. Substituting $\widetilde\pi=0.45$ gives $0.11$. The discrepancy is

$$
(0.45-0.30)(0.20-0.80)=-0.09.
$$

The negative-class contribution is still positive in this example, so non-negative correction does not fix the prior error. A numerical safeguard and a correct observation model solve different problems.
:::

## Feature-dependent positive selection

In a single partially labelled population sample, let

$$
e(x)=P(S=1\mid Y=+1,X=x),
\qquad P(S=1\mid Y=-1,X=x)=0.
$$

If $e(x)$ varies, the observed positives need not represent $P_+$. When the propensity is known and positive wherever positive observations can occur, a useful identity is

$$
R(g)=E\left[\ell_-(X)+\frac{S}{e(X)}\{\ell_+(X)-\ell_-(X)\}\right].
$$

At any zero-propensity point with zero positive-class probability, define $S/e(X)=0$; positive-support positivity rules out zero propensity where the positive class can occur. To verify the identity, condition on $(X,Y)$. The expectation of $S/e(X)$ is one for positives and zero for negatives. The baseline negative loss is therefore replaced by the positive loss exactly when the latent class is positive.

Small propensities can amplify variability. An unknown propensity is not automatically identified from PU observations: $P(S=1\mid X)=e(X)P(Y=+1\mid X)$ can admit multiple factorizations. The feature-dependent-selection literature supplies additional assumptions and estimation methods, not a general exemption from identification questions. Regions with zero positive labelling probability require further information or restrictions.

## Designing a useful evaluation

A simulation should distinguish the learner's information from the evaluator's information. The learner receives the positive and marginal samples and a stated prior. The evaluator can use latent classes on a new sample because the simulation generated them. A fully labelled training method is an oracle benchmark, not an available PU competitor.

Vary one consequential assumption at a time: positive sample size, prior accuracy, positive-selection bias, or score flexibility. Report prediction error and an appropriate score loss, not only the optimized training objective. Repeat the whole fitting procedure to assess variability across training datasets. With genuine PU validation data, use a justified risk estimator and its uncertainty rather than silently treating unknown labels as negatives.

## R laboratory — Fixed-rule bias and fitted-rule behaviour

Part A repeatedly estimates the logistic risk of one fixed Bayes score. It reports bias, standard deviation, mean squared error, and how often the negative contribution is clipped. Part B fits flexible radial-basis scores with the unbiased and non-negative objectives, using the same parameter bounds, regularizer, and three deterministic starts. It reports optimizer status and independent test performance.

The finite parameter bounds and multiple starts are implementation choices. They do not certify a global minimum. The fixed-rule experiment and the fitted-rule comparison answer different questions and should be interpreted separately.

[Download R lab](../_static/labs/04-pu-practice.R)

:::{literalinclude} ../../labs/04-pu-practice.R
:language: r
:::

### Executed laboratory output

:::{include} ../_generated/04-pu-practice-output.md
:::

## Exercises

1. Prove the finite-sample upward-bias statement for a fixed rule. Does it imply worse mean squared error?
2. Derive the variance formula above. What goes wrong if the two positive-sample terms are incorrectly treated as independent?
3. Under logistic loss, simplify the prior-misspecification term. Which feature of the positive score distribution controls its magnitude?
4. Modify Part B to increase the positive sample size. Report both methods' independent test losses without claiming that one must always win.

:::{admonition} Worked solutions
:class: dropdown

**1.** The difference is $(-\widehat B)_+$ pointwise. Taking expectations and using $E\widehat R_{\mathrm{PU}}=R$ gives the result. Mean squared error equals variance plus squared bias, so an increase in bias can be offset by a reduction in variance. Dominance requires additional analysis.

**2.** Combine the positive terms into $\pi\widehat E_+(\ell_+-\ell_-)$ and use independence across the two samples. Separating their variances without covariance generally gives the wrong answer because the same positive observations appear twice.

**3.** Logistic losses satisfy $\ell_+-\ell_-=-g(X)$, so the discrepancy is $-(\widetilde\pi-\pi)E_+g(X)$. A large average positive-class score can make the correction sensitive to the supplied prior. This calculation holds for a fixed rule; refitting under a different prior also changes the rule.

**4.** Larger $n_+$ can reduce the variability of the subtraction, but the observed comparison also depends on the marginal sample, score class, regularization, and numerical optimization. Use repeated training samples and report the comparison with Monte Carlo uncertainty.
:::

## Primary readings

- [Kiryo et al. (2017), “Positive-Unlabeled Learning with Non-Negative Risk Estimator”](https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html): estimator, theory, and flexible-model experiments.
- [Bekker, Robberechts, and Davis (2019), “Beyond the Selected Completely At Random Assumption for Learning from Positive and Unlabeled Data”](https://arxiv.org/abs/1809.03207): feature-dependent labelling and the assumptions needed to learn its mechanism.
- [du Plessis, Niu, and Sugiyama (2015), “Convex Formulation for Learning from Positive and Unlabeled Data”](https://proceedings.mlr.press/v37/plessis15.html): the underlying risk-correction formulation.
