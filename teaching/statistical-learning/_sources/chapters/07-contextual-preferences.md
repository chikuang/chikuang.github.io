# 7. Contextual preferences, judges, and reward learning

**Question.** Whose preference is being learned, in which context, and with what uncertainty?

[Download R lab](../_static/labs/07-contextual-preferences.R)

A single strength per item can be inadequate when the same answer is helpful for one question and inappropriate for another. Contextual preference learning models comparisons conditional on information about the task and the alternatives. Its statistical challenge includes both the predictive model and the mechanism that generates the comparisons.

This chapter extends paired comparisons, handles repeated judges carefully, and derives a finite-action connection to direct preference optimization. It does not require language-model training.

## 7.1 · A contextual comparison model

Let $x$ denote a context and $a,b$ two alternatives. Define their feature difference

$$
d(x,a,b)=\phi(x,a)-\phi(x,b).
$$

A linear utility model gives

$$
\Pr(a\succ b\mid x,a,b)=\sigma\{d(x,a,b)^{\mathsf T}\beta\}.
$$ (eq-contextual-preference)

The target might be this conditional probability, a coefficient contrast, or the decision rule selecting the alternative with largest predicted utility. These targets require different evaluations. Pairwise prediction accuracy alone does not establish a correct global ranking or factual reliability.

For observed labels $Y_r$ and differences $d_r$, the negative log likelihood is

$$
\sum_r\left[\log\{1+\exp(d_r^{\mathsf T}\beta)\}
-Y_r d_r^{\mathsf T}\beta\right].
$$

This is ordinary logistic loss on feature differences. An item-independent utility intercept cancels, and covariates that never vary within a comparison provide no contrast information. Identification requires sufficient variation in the observed differences; separation remains possible.

## 7.2 · Presentation and sampling matter

Suppose $h=1$ when alternative $a$ is displayed first and $h=-1$ when it is displayed second. A model with an order effect is

$$
\operatorname{logit}\Pr(a\succ b\mid x,a,b,h)
=d(x,a,b)^{\mathsf T}\beta+\delta h.
$$

Randomizing presentation helps separate an order effect from content features. It does not make all preference labels objective or remove systematic judge disagreement. Similarly, if comparisons are collected only for close alternatives, the training comparison distribution differs from a deployment distribution containing many easy pairs. State the evaluation population and ensure that the held-out sample represents it, or justify a weighting adjustment.

:::{admonition} Example 7.1 · Relevance and conciseness can disagree
:class: course-example

**Setup.** A has one more unit of relevance and two fewer units of conciseness than B. The hypothetical coefficients are $\beta=(1.2,0.3)$.

**Work it through.** The utility contrast is $1.2-2(0.3)=0.6$, so the predicted preference probability is $\sigma(0.6)=0.646$. If the order effect is $\delta=0.4$, displaying A first changes the model probability to $\sigma(1.0)=0.731$; displaying it second gives $\sigma(0.2)=0.550$.

**What this shows.** A reported preference rate can reflect both content and presentation. A preferred answer can still contain a factual error: the outcome here measures a judge's comparison, not an independently verified truth label.
:::

## 7.3 · Repeated judges are not independent rows

If judge $j$ evaluates several pairs, their responses can share a latent preference, expertise level, or order bias. One possible conditional model is

$$
\operatorname{logit}\Pr(Y_{jr}=1\mid d_{jr},h_{jr},b_j)
=d_{jr}^{\mathsf T}\beta+(\delta+b_j)h_{jr}.
$$

Here $b_j$ represents judge-specific order sensitivity. Integrating out $b_j$ generally does not preserve the same logistic regression coefficients. A pooled logistic fit can instead be understood as a working marginal model, with a population projection target determined by its expected loss. Distinguish this target from the conditional generating coefficient.

For independent judges with repeated observations, a cluster sandwich estimator uses summed score contributions $U_j$:

$$
\widehat V=\widehat H^{-1}
\left(\sum_j U_jU_j^{\mathsf T}\right)\widehat H^{-1}.
$$

Its justification needs enough independent clusters and suitable regularity. It is not an exact small-sample guarantee. A judge bootstrap resamples whole judges, preserving their within-judge observations. Resampling individual comparison rows would destroy that structure.

:::{admonition} Example 7.2 · Eight hundred rows need not mean eight hundred independent measurements
:class: course-example

**Setup.** Forty judges each supply twenty responses. For a simple scalar mean, suppose responses within each judge have equal variance and common correlation $\rho=0.15$.

**Work it through.** The variance inflation relative to independent rows is $1+(20-1)\rho=3.85$. The corresponding effective sample size for this particular mean calculation is $800/3.85\approx208$.

**What this shows.** The number of independent judges matters. This effective-sample-size formula is an illustrative exchangeable-mean calculation, not a universal replacement sample size for every logistic coefficient or crossed judge–prompt design.
:::

When both prompts and judges repeat, dependence can be crossed. A one-way judge bootstrap may then be insufficient. The independent unit for validation and resampling must match the target population. For adaptive pair selection or stopping, inference must additionally be justified under that sequential sampling scheme; ordinary independent-row standard errors are not automatically valid.

## 7.4 · From preference likelihood to a policy

For a fixed context, let $\pi_0(a)>0$ be a reference distribution over finitely many actions. Given rewards $r(a)$, consider

$$
\max_\pi\left\{\sum_a\pi(a)r(a)
-\tau\sum_a\pi(a)\log\frac{\pi(a)}{\pi_0(a)}\right\},\quad \tau>0.
$$

A Lagrange multiplier for $\sum_a\pi(a)=1$ gives

$$
\pi^*(a)=\frac{\pi_0(a)e^{r(a)/\tau}}{Z},\qquad
r(a)=\tau\log\frac{\pi^*(a)}{\pi_0(a)}+\tau\log Z.
$$

Under a Bradley–Terry reward comparison, the common $\log Z$ cancels. The preference probability becomes a logistic function of the difference between two policy/reference log ratios. Replacing $\pi^*$ with a fitted policy and minimizing the resulting negative log probability gives the central DPO objective.

The identity depends on the reward-comparison model, positive reference support, and the KL-regularized optimization setup. It does not establish that the labels are unbiased, that a restricted fitted policy reaches the optimum, or that preference alignment guarantees factual correctness. This can be an offline preference-learning problem: collecting a fixed comparison dataset does not itself make the analysis online learning.

## 7.5 · Execute the judge and policy experiment

The R lab simulates independent judges with repeated responses and heterogeneous order effects. It fits a working logistic model and compares independent-row, judge-sandwich, and judge-bootstrap standard errors. The conditional simulation coefficients are deliberately not treated as the exact targets of the marginal working model. The final calculation verifies the finite-action policy identity numerically.

:::{literalinclude} ../../labs/07-contextual-preferences.R
:language: r
:::

:::{include} ../_generated/07-contextual-preferences-output.md
:::

## 7.6 · Exercises

1. Explain why adding a constant feature to every alternative does not identify an absolute utility intercept.
2. Change the standard deviation of the judge-specific order effects to zero and then increase it. Which standard errors change most? Interpret the result without assuming that a working coefficient equals the conditional generating coefficient.
3. Derive the KL-regularized optimizer and explain why the normalizing constant cancels in a comparison at the same context.
4. Design a train/test split for a dataset containing repeated judges and repeated prompts. State whether your target is performance on new prompts, new judges, or both; identify which dependence remains.

:::{admonition} Worked solutions
:class: dropdown

**1.** A common feature subtracts to zero in $\phi(x,a)-\phi(x,b)$. The observations contain information about relative utility only. A normalization or prior cannot turn this into observed absolute utility.

**2.** Order-effect uncertainty is especially sensitive because the simulation places a shared random effect on that slope. Changes in other coefficients depend on the design. The three estimators answer an uncertainty question for the fitted working model; none corrects its interpretation into a conditional mixed-model coefficient automatically.

**3.** Differentiating with respect to $\pi(a)$ gives $r(a)-\tau\{\log(\pi(a)/\pi_0(a))+1\}+\lambda=0$. Exponentiate and normalize. The same context has the same $Z$, so subtracting rewards removes it. Comparing policies across contexts does not permit cancellation of unrelated normalizers in the same way.

**4.** A new-prompt target requires held-out prompts; a new-judge target requires held-out judges. To evaluate both, reserve both groups and ensure no shared observations leak through either index. For uncertainty about a crossed population, justify a suitable multiway method rather than automatically resampling rows.
:::

## Reading and transition

[Turner and Firth (2012)](https://www.jstatsoft.org/v48/i09/) develops structured paired-comparison models and order effects. [Rafailov et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract.html) gives the DPO formulation. Next, change the learning protocol: decisions must be made before each new outcome is available.
