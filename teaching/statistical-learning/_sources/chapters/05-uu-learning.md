# 5. Learning from two unlabelled mixtures

:::{admonition} Core idea - Different mixtures provide an invertible contrast
:class: course-core

The information comes from known, different mixing proportions and shared class distributions. A nonzero prior gap makes the population inverse available; a small gap can still make finite-sample estimates unusably noisy. The target prior is a separate choice from both sampling priors.
:::

*Topics of Statistical Learning · Week 5*

## Where the supervision comes from

Unlabelled–unlabelled learning uses two collections without individual class labels. It is not ordinary clustering: the aim is to predict a specified positive or negative class, and aggregate information about the two collections supplies supervision. The basic model assumes that their class proportions are known and different, while their class-conditional feature distributions are shared.

The central calculation is a two-by-two linear inversion. It reveals both why learning is possible and why nearly equal mixture proportions produce unstable risk estimates. This chapter treats the prior information as known so the identification argument is transparent; estimating it requires additional assumptions and analysis.

## The two-mixture model and target population

Let $P_+$ and $P_-$ be the positive- and negative-class feature distributions. We observe independent samples from

$$
Q_1=aP_+ +(1-a)P_-,\qquad
Q_2=bP_+ +(1-b)P_-,
$$

where $a,b\in[0,1]$ are known and $a\ne b$. Usually both are strictly between zero and one, so both collections are genuinely unlabelled mixtures. The target population has a separately specified positive prior $\pi$, giving risk

$$
R_\pi(g)=\pi E_+\ell_+ +(1-\pi)E_-\ell_-.
$$

The target prior need not equal $a$ or $b$. Setting $\pi=1/2$ deliberately evaluates balanced risk. It does not establish that either observed collection, or the application population, is balanced.

The assumption of common $P_+$ and $P_-$ is essential. Different feature distributions in the two collections must arise from their class proportions under this model. A change in measurement instruments or in the positive-class population can violate that relationship even when the proportions are known accurately.

## Inverting the mixtures

For any measurable set, the two mixture equations form a linear system with determinant $a-b$. Its inverse yields the distribution identities

$$
P_+=\frac{(1-b)Q_1-(1-a)Q_2}{a-b},\qquad
P_-=\frac{aQ_2-bQ_1}{a-b}.
$$

These are identities between probability measures under the assumed model, although their right sides contain signed coefficients. Their population values remain non-negative measures. Empirical versions need not have every property of a probability distribution in a finite sample.

For an integrable function $h$, the same equations hold with distributions replaced by expectations. Apply the first to $\ell_+$ and the second to $\ell_-$:

$$
\begin{aligned}
R_\pi(g)
={}&\frac{\pi}{a-b}\left\{(1-b)E_{Q_1}\ell_+-(1-a)E_{Q_2}\ell_+\right\}\\
&+\frac{1-\pi}{a-b}\left\{aE_{Q_2}\ell_- -bE_{Q_1}\ell_-\right\}.
\end{aligned}
$$

Replacing these expectations with sample averages gives an unbiased estimator for a fixed rule under the known-prior model. As in PU learning, fitting a rule and selecting its tuning parameters require further control of generalization and optimization. Unbiasedness for a fixed rule is not a universal guarantee for every learning algorithm.

:::{admonition} Worked example 5.1 — Recovering class-specific error from aggregate information
:class: course-example

Let $a=0.8$, $b=0.2$, and let a classifier predict positive on a feature region $A$. Suppose $Q_1(A)=0.69$ and $Q_2(A)=0.36$. Inversion gives

$$
P_+(A)=\frac{0.8(0.69)-0.2(0.36)}{0.6}=0.80,
$$

$$
P_-(A)=\frac{0.8(0.36)-0.2(0.69)}{0.6}=0.25.
$$

The classifier therefore has false-negative rate $0.20$ and false-positive rate $0.25$. At target prior $\pi=0.4$, its error is $0.4(0.20)+0.6(0.25)=0.23$. Its balanced error is $0.225$.

Neither observed collection was labelled at the individual level. The different, known mixing proportions made the two class-specific probabilities recoverable. Predicting which collection an observation came from is a related but different classification problem.
:::

## Why small prior separation is a statistical problem

Let $h_1$ and $h_2$ collect the coefficients of the losses in the two expectations:

$$
h_1(X)=\frac{\pi(1-b)\ell_+(X)-(1-\pi)b\ell_-(X)}{a-b},
$$

$$
h_2(X)=\frac{(1-\pi)a\ell_-(X)-\pi(1-a)\ell_+(X)}{a-b}.
$$

For independent sample sizes $n_1,n_2$ and a fixed rule,

$$
\operatorname{Var}(\widehat R_\pi)
=\frac{\operatorname{Var}_{Q_1}(h_1)}{n_1}
+\frac{\operatorname{Var}_{Q_2}(h_2)}{n_2}.
$$

The coefficients contain $1/(a-b)$. In typical nondegenerate examples, variance is amplified at the scale $1/(a-b)^2$. Particular losses and rules can create cancellations, so the exact variance is the appropriate quantity to inspect.

For balanced zero–one risk of a threshold classifier predicting positive when $X>t$, write $F_k=P_{Q_k}(X\leq t)$. The estimator simplifies to

$$
\widehat R_{1/2}=\frac12+\frac{\widehat F_1-\widehat F_2}{2(a-b)},
$$

and its variance is

$$
\frac{F_1(1-F_1)}{4n_1(a-b)^2}
+\frac{F_2(1-F_2)}{4n_2(a-b)^2}.
$$

:::{admonition} Worked example 5.2 — An unbiased estimate outside the probability range
:class: course-example

Let $P_+=N(1,1)$, $P_-=N(-1,1)$, $t=0$, and $\pi=1/2$. The true classification risk is $\Phi(-1)\approx0.1587$. Take $n_1=n_2=500$ and fix $a=0.7$.

When $b=0.3$, the fixed-rule risk estimator has standard deviation about $0.038$. When $b=0.69$, the standard deviation is about $1.52$. Both estimators remain unbiased under the model, but the second often falls below zero or above one.

This is not a floating-point error. The two collections contain little information for separating their latent class contributions. Clipping the result to $[0,1]$ changes the estimator and generally introduces bias. At $a=b$, the displayed inverse is undefined and the basic two-mixture model no longer identifies the class-specific distributions without additional restrictions.
:::

## Identification failures and links to PU learning

If $a=b$, both observed distributions give the same mixture equation. An unrestricted pair $(P_+,P_-)$ cannot generally be recovered from that single equation. If the proportions themselves are unknown, even their difference and class orientation need not be identifiable from the observed distributions alone. Extra assumptions can change this conclusion, but they must be stated explicitly.

If class-conditionals differ between collections, the algebra is solving the wrong system. A larger sample can then concentrate the risk estimate around an incorrect limit. Sensitivity experiments should examine this structural failure as well as finite-sample variability.

PU learning is a useful special case: set $a=1$ so $Q_1=P_+$ and set $b=\pi$ so $Q_2$ is the target feature marginal. Substitution reduces the UU expression to $\pi E_+\ell_+ + E_U\ell_- -\pi E_+\ell_-$. This unification concerns observation distributions and risk identities; it does not make a set of ordinary clustering labels into positive-class supervision.

## Allocate a budget according to variance contributions

Write the two variance contributions for a fixed rule as $v_1/n_1+v_2/n_2$, where $v_j=\operatorname{Var}_{Q_j}(h_j)$ as defined above. With equal sampling costs, a fixed total $N=n_1+n_2$, and positive $v_1,v_2$, differentiating with respect to $n_1$ gives the continuous optimum

$$\frac{n_1}{n_2}=\sqrt{\frac{v_1}{v_2}}.$$

For illustrative values $v_1=4v_2$, the allocation ratio is two to one. This is a design calculation for specified variance contributions, not a claim that equal sample sizes are always inefficient. The contributions depend on the priors, loss, and rule; a pilot-based allocation must account for uncertainty in those inputs.

:::{admonition} Boundary - sample size cannot replace identification
:class: course-boundary

A larger budget can reduce variance when $a\ne b$. At $a=b$, the two-mixture inverse is unavailable under the unrestricted shared-component model. Collect different information or state additional identifying restrictions before fitting.
:::

## Laboratory — Prior separation and assumption failure

The lab holds one threshold classifier fixed and repeatedly estimates its balanced target risk. It compares four values of $b$ while keeping $a=0.7$, reports empirical and exact standard deviations, and records the fraction of estimates outside $[0,1]$. A final experiment shifts the positive-class mean in the second collection and compares the misspecified correction's analytic limit with its simulation average.

Known latent classes are used only inside the generator. The risk estimator receives features and supplied proportions, not individual class labels. The analytic reference risk is available to the evaluator because the simulation model is specified.

[Download lab](../_static/labs/05-uu-learning.R)

:::{literalinclude} ../../labs/05-uu-learning.R
:language: r
:::

### Executed laboratory output

:::{include} ../_generated/05-uu-learning-output.md
:::

## Exercises

1. Verify both measure inversions by substitution. Why is the condition $a\ne b$ necessary for this argument?
2. Substitute $a=1$ and $b=\pi$ into the UU risk identity and recover the PU identity.
3. Derive the balanced zero–one estimator and its variance. Why must the two sample sizes appear separately?
4. If the separation $|a-b|$ is approximately halved while other variance factors remain similar, how should both sample sizes change to maintain approximately the same precision?
5. What is the difference between fixing an unstable finite-sample estimate and fixing a violated common-class-conditional assumption?

:::{admonition} Worked solutions
:class: dropdown

**1.** The coefficient of $P_+$ in the first numerator is $a(1-b)-b(1-a)=a-b$, and the coefficient of $P_-$ is zero. The second identity is analogous. At equal priors the matrix is singular and this cancellation does not provide an inverse.

**2.** The first bracket becomes $\pi E_+\ell_+$. The second becomes $E_U\ell_- -\pi E_+\ell_-$ because $a-b=1-\pi$. Their sum is the PU expression.

**3.** For a threshold rule, $\ell_+=\mathbf1(X\leq t)$ and $\ell_-=1-\ell_+$. At $\pi=1/2$, collecting constants gives the stated estimator. Independent Bernoulli sample means have variances $F_k(1-F_k)/n_k$; the two independent samples may have different sizes and different event probabilities.

**4.** Increase both sizes by approximately a factor of four. This compensates for the squared inverse separation, provided the remaining variance terms are comparable. It does not repair exact equality of the priors.

**5.** More observations or appropriate regularization can improve a noisy but correctly specified estimator. Structural misspecification changes what the expression estimates. Addressing it requires an observation model, target, or data source that matches the application, not simply more repetitions of the same biased procedure.
:::

## Primary readings

- [Lu et al. (2019), “On the Minimal Supervision for Training Any Binary Classifier from Only Unlabeled Data”](https://arxiv.org/abs/1808.10585): the two-mixture risk formulation and its learning theory.
- [Lu et al. (2020), “Mitigating Overfitting in Supervised Classification from Two Unlabeled Datasets: A Consistent Risk Correction Approach”](https://proceedings.mlr.press/v108/lu20c.html): correction of empirical risks in flexible UU learning.
- [Lu et al. (2021), “Binary Classification from Multiple Unlabeled Datasets via Surrogate Set Classification”](https://proceedings.mlr.press/v139/lu21c.html): an optional extension beyond two collections.

**Connection to the next block:** Preference data replace an absolute class label with a comparison. We will again ask which target the observation mechanism identifies before fitting a model.
