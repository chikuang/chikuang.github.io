# 6. Pairwise preference learning

**Question.** What can we learn when a person can compare two alternatives but cannot supply an absolute response?

[Download R lab](../_static/labs/06-pairwise-preferences.R)

A clinician may prefer one explanation to another, a reader may rank two retrieved passages, and a customer may choose between products. These observations are informative, but their meaning differs from a numerical measurement of quality. This chapter develops a statistical model for the comparisons, identifies its estimable quantities, and asks how the comparison schedule affects precision.

By the end, you should be able to derive a Bradley–Terry likelihood, diagnose two distinct failures of estimation, and connect preference learning to optimal experimental design.

## 6.1 · Specify the observation and target

There are $K$ items. For a scheduled comparison of items $i$ and $j$, let $Y=1$ mean that $i$ is preferred. The basic model is

$$
p_{ij}=\Pr(Y=1\mid i,j)=\frac{e^{\theta_i}}{e^{\theta_i}+e^{\theta_j}}
=\sigma(\theta_i-\theta_j),\qquad \sigma(u)=\frac{1}{1+e^{-u}}.
$$ (eq-bt-model)

Possible targets are a strength contrast $\theta_i-\theta_j$, a pairwise probability $p_{ij}$, or an ordering of the strengths. A confidence interval for a contrast does not automatically provide a simultaneous confidence statement about the entire ranking.

The introductory analysis assumes a fixed comparison schedule, constant strengths, binary outcomes without ties, and conditionally independent comparisons. These are modelling choices to check. Repeated judges, shared prompts, order effects, or changing preferences can violate the simple sampling model; Chapter 7 addresses some of these complications.

## 6.2 · Derive the likelihood

Suppose $w_{ij}$ of $n_{ij}$ independent comparisons favour item $i$. Write $x_{ij}=e_i-e_j$. Ignoring factors independent of the parameters, the log likelihood is

$$
\ell(\theta)=\sum_{i<j}\left[w_{ij}x_{ij}^{\mathsf T}\theta
-n_{ij}\log\{1+\exp(x_{ij}^{\mathsf T}\theta)\}\right].
$$

This is a binomial logistic regression with a structured design matrix. The score and negative Hessian are

$$
U(\theta)=\sum_{i<j}(w_{ij}-n_{ij}p_{ij})x_{ij},\qquad
I(\theta)=\sum_{i<j}n_{ij}p_{ij}(1-p_{ij})x_{ij}x_{ij}^{\mathsf T}.
$$ (eq-bt-information)

Adding the same constant to all strengths leaves every probability unchanged. Impose $\theta_K=0$ or $\sum_i\theta_i=0$ before fitting. This normalization fixes an arbitrary origin; it does not create information that the experiment failed to collect.

:::{admonition} Example 6.1 · A preference probability and its uncertainty
:class: course-example

**Setup.** In an illustrative fixed comparison, alternative A wins 12 times and B wins 8 times.

**Work it through.** The estimated log-strength difference is $\widehat\delta=\log(12/8)=0.405$. Its approximate standard error is $\sqrt{1/12+1/8}=0.456$. A nominal 95% Wald interval is approximately $[-0.489,1.300]$. Transforming the endpoints with the logistic function gives approximately $[0.380,0.786]$ for the win probability.

**What this shows.** A point estimate of 0.60 does not establish a reliable preference for A. The interval concerns repeated independent comparisons under the model; it is neither a probability that A is objectively correct nor a simultaneous ranking guarantee.
:::

## 6.3 · Identification is different from a finite estimate

Construct an **undirected comparison graph** with an edge whenever $n_{ij}>0$. If it is connected, all strength differences are identified in the model, subject to the common additive constant. If two components never meet, their relative location remains unidentified. No optimizer can recover an unsupported cross-component contrast from the likelihood alone.

Now construct a **directed win graph**, with an arrow from $i$ to $j$ when $i$ has beaten $j$. A finite, unique unpenalized maximum-likelihood estimate after normalization requires this graph to be strongly connected: every item can reach every other along directed wins. An undefeated group can cause separation even when the undirected comparison graph is connected.

For example, observing only A beating B and B beating C gives a connected schedule but supports arbitrarily increasing strength differences. Penalization or a proper prior can stabilize estimation, but the resulting answer depends on that additional assumption. Report it explicitly rather than treating a large finite optimizer output as a successful unpenalized fit.

## 6.4 · Why the comparison schedule matters

The information matrix in Equation {eq}`eq-bt-information` is a weighted graph Laplacian. Each edge contributes information about one difference. For a contrast $a^{\mathsf T}\theta$ with $a^{\mathsf T}\mathbf 1=0$, a local large-sample variance approximation is $a^{\mathsf T}I(\theta)^+a$, where $I^+$ is the generalized inverse on the identifiable contrast space.

This gives a concrete design problem: allocate a fixed comparison budget over pairs to minimize variance for an important contrast, average variance over several contrasts, or another specified information criterion. Local information depends on the unknown strengths, so a locally optimal allocation uses nominal values or a pilot. A robust or sequential design must account for that uncertainty.

:::{admonition} Example 6.2 · Equal budget, unequal precision
:class: course-example

**Setup.** Three equally strong items have $p_{ij}=1/2$. We have 60 comparisons and want $\theta_1-\theta_3$.

**Work it through.** Allocating 20 comparisons to each of the three pairs gives edge information $20/4=5$ and contrast variance $2/(3\times5)=2/15$. Allocating 30 each to pairs $(1,2)$ and $(2,3)$ gives two edge informations of $7.5$; the endpoint difference adds their variances, yielding $1/7.5+1/7.5=4/15$.

**What this shows.** Both schedules identify the target, but the chain has twice the local variance. These numbers use the equal-strength model and a local information approximation. They do not prove that balanced comparisons are universally optimal for every target and parameter value.
:::

## 6.5 · Execute the comparison experiment

The base-R lab compares complete, chain, and disconnected schedules. It reports design ranks, local precision for an endpoint contrast, and a fixed-design simulation. Only connected schedules enter the estimation comparison. Flagged fits are counted; coverage summaries use the usable fits and therefore must be read alongside those counts.

Change one element at a time: the total budget, strength separation, or graph. Before running, predict which contrast should become harder to estimate. The likelihood implementation uses `glm.fit`; the full R source is included so the comparison matrix and normalization remain visible.

:::{literalinclude} ../../labs/06-pairwise-preferences.R
:language: r
:::

:::{include} ../_generated/06-pairwise-preferences-output.md
:::

## 6.6 · Exercises

1. Show that replacing $\theta$ by $\theta+c\mathbf1$ leaves the likelihood unchanged. Explain why a disconnected graph introduces more than one arbitrary offset.
2. Derive the score and information matrix. Which term makes a nearly deterministic comparison less informative locally about a small change in its log odds?
3. Construct a connected comparison schedule with no finite unpenalized MLE. Explain what a ridge penalty changes and what it does not establish about the scientific target.
4. Modify the experiment to target $\theta_2-\theta_3$. Compare the complete and chain designs at the same budget. Report estimation error, fit flags, and Monte Carlo uncertainty; do not choose seeds to obtain a preferred ranking.

:::{admonition} Worked solutions
:class: dropdown

**1.** Every design row has entries summing to zero. Within a disconnected component, adding its own constant preserves all observed probabilities. Cross-component contrasts consequently change without changing the likelihood.

**2.** Differentiate $w\eta-n\log(1+e^\eta)$ to obtain $w-n\sigma(\eta)$ and curvature $n\sigma(\eta)\{1-\sigma(\eta)\}$. The latter becomes small as a probability approaches zero or one. Nearly certain wins provide little local information per comparison about the underlying log-strength contrast.

**3.** A beats B in every observed comparison and B beats C in every observed comparison. The win graph has no return path. A positive quadratic penalty selects finite strengths and an origin if applied appropriately, but this is a penalized model, not new evidence about the unconstrained contrast.

**4.** Replace the contrast vector and extract the matching fitted coefficient difference and covariance expression. The answer depends on the graph location and generating strengths. A legitimate result can favour either design for a particular target; comparisons should keep budget and simulation protocol fixed.
:::

## Reading and transition

Read [Bradley and Terry (1952)](https://doi.org/10.1093/biomet/39.3-4.324) for the original model and [Turner and Firth (2012)](https://www.jstatsoft.org/v48/i09/) for its GLM representation and R implementation. [Yan (2014)](https://arxiv.org/abs/1411.1168) discusses failure of the strong-connection condition. Next, allow comparisons to depend on context and on who supplied the preference.
