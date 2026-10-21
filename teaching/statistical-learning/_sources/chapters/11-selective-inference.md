# 11. Selective inference: asking a question after seeing the data

:::{admonition} Core idea - Selection changes the reference experiment
:class: course-core

When a statistic helped choose the question, its ordinary null distribution no longer describes the selected experiment. Conditional inference compares the observation with alternative data that would have produced the specified selection, after handling nuisance information.
:::

## From a selected question to a conditional reference

```{figure} ../_static/diagrams/selection-pipeline.*
:alt: Hierarchical and path-conditioned k-means procedures preserve different events. After nuisance conditioning, the admissible radii determine the truncated null distribution under the specified model.
:class: course-flow
:width: 100%

Hierarchical and path-conditioned k-means procedures preserve different events. After nuisance conditioning, the admissible radii determine the truncated null distribution under the specified model.
```

## Learning objectives

By the end of this chapter, you should be able to:

1. distinguish a fixed inferential target from a target selected using observations;
2. write a conditional Type I error requirement for a selected hypothesis;
3. derive and compute an exact pivot for the largest absolute Gaussian observation;
4. explain when sample splitting creates the independence an analysis needs;
5. identify additional selection steps that a proposed guarantee has not covered.

The preceding chapters allowed the learner to choose a predictor, a comparison, or an action. We now allow the analyst to choose the *question*. This changes the reference distribution used for inference. A large estimate may be large partly because it won a search.

## 11.1 A selected parameter is still a parameter

Suppose that

$$
Z_j\sim N(\theta_j,1),\qquad j=1,\ldots,m,
$$

independently. Each $\theta_j$ is a fixed unknown number. An analyst inspects all observations and reports

$$
J=\mathop{\mathrm{argmax}}_j |Z_j|.
$$

Ties have probability zero under this continuous model; the lab nevertheless uses a deterministic tie rule. The reported target is $\theta_J$. Its index is random, but the list of possible parameters was defined before observing the data. Saying which parameter was selected is part of specifying the inferential procedure.

For a fixed coordinate, $2\{1-\Phi(|Z_j|)\}$ is a valid two-sided p-value for $H_{0j}:\theta_j=0$. Applying that formula to the winner does not retain the same calibration among reported hypotheses. Under the global null, the probability that the winning naïve p-value is below $\alpha$ is

$$
1-(1-\alpha)^m.
$$

For $m=20$ and $\alpha=0.05$, this is about $0.642$. Increasing the number of opportunities to find an extreme observation changes the distribution of the reported statistic even though each original measurement remains well calibrated.

## 11.2 What probability should be controlled?

One selective requirement is, for each eligible $j$ and every parameter vector satisfying $\theta_j=0$,

$$
\Pr_{\theta}\{p_j\leq\alpha\mid J=j\}\leq\alpha.
$$

Other coordinates need not have zero means. This is a statement about errors among repetitions in which the procedure asks question $j$. It differs from familywise error control across a collection of simultaneous rejections. A method can be designed for either goal, but the goal must be stated.

Conditioning more finely can simplify computation. We will also condition on the sign of the selected observation and on the unselected observations. If the resulting p-value is uniform under each such conditional distribution, averaging over the additional information preserves the desired selective calibration. Conditioning discards information, so computational convenience can have a power cost.

Fithian, Sun, and Taylor formulate this connection between selection, conditional error, and information loss in [Optimal Inference After Model Selection](https://arxiv.org/abs/1410.2597).

## 11.3 Deriving an exact selected-Gaussian pivot

Fix a selected index $j$, the observed unselected vector $z_{-j}$, and the selected sign $s\in\{-1,1\}$. Define

$$
c=\max_{k\neq j}|z_k|,\qquad T=sZ_j.
$$

Before selection, $T\sim N(s\theta_j,1)$ and is independent of $Z_{-j}$. The selection event with sign $s$ is simply $T\geq c$. Consequently,

$$
f_{T\mid J=j,S=s,Z_{-j}}(t)
=\frac{\phi(t-s\theta_j)}
 {1-\Phi(c-s\theta_j)}\,\mathbf 1(t\geq c).
$$

Under $H_{0j}$, an upper-tail conditional p-value is

$$
p_{\mathrm{sel}}
=\frac{1-\Phi(|z_j|)}{1-\Phi(c)}.
$$

This upper tail tests extremeness in the direction selected by the data, while conditioning on that direction. It is not a claim that the direction was specified in advance. The probability integral transform gives a uniform p-value under the conditional null. The conditioning threshold $c$ is therefore an essential part of the calculation.

For numerical stability, evaluate the ratio by subtracting log survival probabilities and exponentiating. Direct subtraction of a normal CDF from one loses precision for extreme observations.

:::{admonition} Example 11.1 — A large observation with a strong competitor
:class: course-example

Let $z=(2.4,-1.7,0.5)$. Coordinate 1 is selected, its sign is positive, and $c=1.7$. The ordinary two-sided p-value is approximately $0.0164$. The selective value is

$$
\frac{1-\Phi(2.4)}{1-\Phi(1.7)}
\approx 0.184.
$$

The observation is unusual for an unselected standard normal, but much less unusual among positive observations known to exceed $1.7$. This calculation does not establish that $\theta_1=0$. It explains why the original reference distribution overstates the evidence against that hypothesis.
:::

:::{admonition} Example 11.2 — A global-null test answers a different question
:class: course-example

Under $\theta_1=\cdots=\theta_m=0$, the maximum $M=\max_j|Z_j|$ has CDF

$$
\Pr(M\leq t)=\{2\Phi(t)-1\}^{m},\qquad t\geq0.
$$

Thus $1-\{2\Phi(M)-1\}^{m}$ is an exact global-null p-value. It tests whether *all* coordinates have mean zero. The conditional p-value above instead targets the selected coordinate and permits other means to be nonzero. Substituting one for the other changes the null hypothesis.
:::

## 11.4 From a one-dimensional truncation to polyhedral selection

For Gaussian data $Y\sim N(\mu,\Sigma)$, consider a linear contrast $\eta^\top Y$ and a selection event $AY\leq b$. Define

$$
T=\eta^\top Y,\qquad
v=\frac{\Sigma\eta}{\eta^\top\Sigma\eta},\qquad
R=Y-vT.
$$

Then $\operatorname{Cov}(R,T)=0$, so Gaussianity makes $R$ and $T$ independent. Once $R=r$ is fixed, each selection inequality becomes a bound on the scalar $T$:

$$
(Av)_iT\leq b_i-(Ar)_i.
$$

Intersecting these bounds gives a truncation interval, possibly with zero-probability or vacuous cases. Conditional on $R$ and selection, $T$ follows its original Gaussian law restricted to that interval. This is the central geometric idea behind the polyhedral approach of [Lee, Sun, Sun, and Taylor (2016)](https://doi.org/10.1214/15-AOS1371).

The assumptions matter: this derivation uses Gaussianity, known covariance, and an affine description of the selection event. An arbitrary clustering or tuning pipeline need not produce one polyhedron. The next chapters use different selection geometry.

## 11.5 What splitting does and does not provide

If an independent discovery dataset chooses a fixed hypothesis and an untouched inference dataset supports a valid test of that hypothesis, ordinary inference can be conditionally valid given discovery. The price is reduced information at each stage.

Independence must concern the *inferential information*, not just a convenient division of row numbers. If held-out observations are assigned to clusters using the same features subsequently tested, their group membership still depends on the test observations. Similarly, selecting a threshold after viewing validation errors is another selection operation. A complete analysis records the question, the information that selected it, and the information used to answer it.

## Translate the conditioning event into a rejection threshold

In the selected-Gaussian example, condition on the other coordinates and on a positive selected sign. Let $c$ be the largest absolute competing coordinate. Under the selected coordinate's zero-mean null, the remaining observation is a standard normal truncated to $(c,\infty)$. For $z>c$,

$$p_{\mathrm{sel}}=\frac{1-\Phi(z)}{1-\Phi(c)}.$$

A level-$\alpha$ rejection therefore requires

$$z\geq\Phi^{-1}\!\left(1-\alpha\{1-\Phi(c)\}\right).$$

At $c=2$ and $\alpha=0.05$, this threshold is about $3.05$. Being just larger than the competitor is not unusual within the selected experiment, even when the ordinary Gaussian tail is small. This is a calculation for the stated Gaussian selection model, not a universal adjustment for arbitrary searches.

:::{admonition} Analysis check - describe the repeated experiment
:class: course-method

List the selected target, the selection event, the nuisance information conditioned on, and the remaining random statistic. Then state whether the error guarantee is conditional or unconditional. A p-value without that reference experiment is difficult to interpret.
:::

## 11.6 Reproducible Laboratory

The script compares naïve and selective rejection rates under the global null and checks selected null coordinates when one other coordinate has a nonzero mean. It also reports Monte Carlo standard errors. An empirical rejection rate near $0.05$ is useful validation of this implementation; the exact guarantee comes from the derivation, not the simulation.

[Download lab](../_static/labs/11-selective-inference.R)

:::{literalinclude} ../../labs/11-selective-inference.R
:language: r
:caption: Exact inference for the largest absolute independent Gaussian observation
:::

### Computed output

:::{include} ../_generated/11-selective-inference-output.md
:::

## Exercises

**Exercise 11.1.** Derive $1-(1-\alpha)^m$ for the naïve selected test under the global null. What happens as $m$ grows?

**Exercise 11.2.** Prove that the conditional p-value is uniform when $\theta_j=0$, conditional on $Z_{-j}$, $J=j$, and the selected sign. Explain why the other means may be nonzero.

**Exercise 11.3.** Derive the lower and upper truncation bounds for $AY\leq b$ after fixing $R=r$. Treat rows with $(Av)_i=0$.

**Exercise 11.4.** A researcher selects the largest effect, reports its ordinary 95% interval, and applies a Bonferroni correction to a separate list of p-values. Has the interval acquired conditional 95% coverage? State the missing argument.

## Solutions

:::{admonition} Solution to Exercise 11.1
:class: dropdown

Each ordinary two-sided p-value is independent uniform under the global null. The selected p-value is their minimum. Its probability of exceeding $\alpha$ is $(1-\alpha)^m$, so its rejection probability is the complement. For fixed $\alpha>0$, this probability tends to one.
:::

:::{admonition} Solution to Exercise 11.2
:class: dropdown

Given the conditioning information, $T$ has standard-normal density restricted to $[c,\infty)$. Its conditional survival function is $\bar\Phi(t)/\bar\Phi(c)$. Applying that survival function to a continuous random variable with this law gives a uniform random variable. Independence makes the conditional distribution of $Z_j$ unaffected by the observed other coordinates before truncation; their unknown means therefore do not enter this pivot.
:::

:::{admonition} Solution to Exercise 11.3
:class: dropdown

Write $a_i=(Av)_i$ and $d_i=b_i-(Ar)_i$. Positive $a_i$ produce upper bounds $d_i/a_i$; negative $a_i$ produce lower bounds $d_i/a_i$. Their intersection is

$$
\left[
\max_{i:a_i<0}d_i/a_i,\;
\min_{i:a_i>0}d_i/a_i
\right],
$$

with infinite endpoints when a set of bounds is empty. For $a_i=0$, require $0\leq d_i$; otherwise the conditional selection event is impossible.
:::

:::{admonition} Solution to Exercise 11.4
:class: dropdown

No. Altering the p-values of another procedure does not alter the reported interval or prove its conditional coverage. A simultaneous confidence procedure could justify a different familywise coverage statement, while a selective interval requires inversion of tests or another argument under the stated conditioning event. The researcher must identify the desired coverage probability and construct an interval satisfying it.
:::

## Reading and next step

- **Core:** Fithian, W., Sun, D., and Taylor, J. *Optimal Inference After Model Selection*. [arXiv:1410.2597](https://arxiv.org/abs/1410.2597), 2014, revised 2017. Read the introduction and the definition of selective error.
- **Core excerpts:** Lee, J. D., Sun, D. L., Sun, Y., and Taylor, J. E. (2016). *Exact post-selection inference, with application to the lasso*. **Annals of Statistics 44**, 907–927. [Published paper](https://doi.org/10.1214/15-AOS1371), especially the polyhedral argument.

Next we replace the selected coordinate with a pair of groups produced by hierarchical clustering. The same requirement—account for how the question was chosen—remains, but the statistic and selection event change.
