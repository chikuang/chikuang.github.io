# 14. Research synthesis: assumptions, validation, and independent information

:::{admonition} Core idea - Match the evidence to the guarantee
:class: course-core

Reproducible computation shows that an analysis can be repeated. Statistical validity also requires the correct target and assumptions. A useful extension changes an assumption deliberately and measures which guarantee survives, rather than treating one successful run as universal evidence.
:::

## Separate discovery and inference information

```{figure} ../_static/diagrams/independent-information.*
:alt: For the equal split under the known-variance Gaussian model, X + E and X - E are independent under joint repeated sampling and each has twice the original variance. Independence is not conditional on the original X.
:class: course-flow
:width: 100%

For the equal split under the known-variance Gaussian model, X + E and X - E are independent under joint repeated sampling and each has twice the original variance. Independence is not conditional on the original X.
```

## Learning objectives

By the end of this chapter, you should be able to turn a course method into a focused research question, design a sensitivity study that targets its assumptions, separate several kinds of statistical guarantee, and implement Gaussian data thinning as an independent-information validation strategy.

A successful course project does not require a new theorem or training a large language model. It requires a clearly stated problem, a defensible method, a reproducible comparison, and an explanation of what the evidence establishes.

## 14.1 Start with the target and the observation mechanism

The course has considered several ways in which an ordinary supervised-learning problem changes:

| Setting | Available information | Main question to resolve |
|---|---|---|
| Positive–unlabelled learning | Some positive examples and an unlabelled sample | Which labelling mechanism and class prior identify risk? |
| Unlabelled–unlabelled learning | Two mixtures with different class proportions | Are the shared components identifiable, and is inversion stable? |
| Preference learning | Comparisons, possibly repeated within prompts or people | What preference parameter is identified by this comparison design? |
| Online learning | Sequentially revealed losses or rewards | Which comparator and feedback model define regret? |
| Selective inference | A hypothesis chosen using observations | Which selection event and sampling model justify error control? |

The same report template works across these settings: **target, observed data, assumptions, estimator or decision rule, guarantee, and failure mode**. Their guarantees remain different. Small excess prediction risk, small regret, a calibrated confidence interval, and a valid selected-hypothesis test cannot be ranked by a single shared metric.

## 14.2 A focused extension: independent information for the same units

In post-clustering inference, row splitting can leave the group assignment dependent on the feature being tested. An alternative is to split each observation into statistically independent information components while keeping the observation's identity.

[Neufeld, Dharamshi, Gao, and Witten (2024)](https://jmlr.org/papers/v25/23-0446.html) develop data thinning for convolution-closed distributions. We study the Gaussian case because its assumptions and derivation are transparent.

Let

$$
X\sim N_n(\mu,\sigma^2I_n),\qquad
E\sim N_n(0,\sigma^2I_n),\qquad X\perp E,
$$

with known $\sigma$. For a discovery-information fraction $\tau\in(0,1)$, form

$$
A=\tau X+\sqrt{\tau(1-\tau)}E,\qquad
B=(1-\tau)X-\sqrt{\tau(1-\tau)}E.
$$

These components sum to $X$. Their means are $\tau\mu$ and $(1-\tau)\mu$, and their variances are $\tau\sigma^2I_n$ and $(1-\tau)\sigma^2I_n$. Their covariance is

$$
\tau(1-\tau)\sigma^2I_n
-\tau(1-\tau)\sigma^2I_n=0.
$$

Because $(A,B)$ is jointly Gaussian, zero covariance establishes independence.

Normalize the components to preserve the original means:

$$
X^{D}=\frac A\tau
=X+\sqrt{\frac{1-\tau}{\tau}}E,\qquad
X^{I}=\frac B{1-\tau}
=X-\sqrt{\frac{\tau}{1-\tau}}E.
$$

Then $X^D$ and $X^I$ are independent, both have mean $\mu$, and their variances are $\sigma^2/\tau$ and $\sigma^2/(1-\tau)$. Larger $\tau$ improves discovery precision while making inference noisier. Independent normal noise is added featurewise for the spherical multivariate model.

:::{admonition} Example 14.1 — An equal information split
:class: course-example

With $\tau=1/2$, the normalized components are $X^D=X+E$ and $X^I=X-E$. Both have variance $2\sigma^2I_n$, their covariance is zero, and their average is $X$.

They are independent under the joint repeated-sampling model for $(X,E)$. They are **not** independent conditional on the original $X$, because their sum is then fixed. The validation argument conditions on discovery information, not on every variable generated during the construction.
:::

## 14.3 Inference conditional on the discovered groups

Cluster the rows of $X^D$ and obtain groups $C_1,C_2$. Conditional on all discovery information, these index sets are fixed and $X^I$ retains its Gaussian law. For one feature, the target is

$$
\delta_C=\frac1{|C_2|}\sum_{i\in C_2}\mu_i
-\frac1{|C_1|}\sum_{i\in C_1}\mu_i,
$$

and the estimator uses the corresponding means of $X^I$. Its conditional variance is

$$
v_C=\frac{\sigma^2}{1-\tau}
\left(\frac1{|C_1|}+\frac1{|C_2|}\right).
$$

Therefore $\widehat\delta_C\pm z_{0.975}\sqrt{v_C}$ has conditional 95% coverage for this selected target under the stated model. The discovery algorithm can be complicated because its information is independent of the inference component. Any further decision based on $X^I$ requires its own accounting.

The target still depends on the discovered groups. A good confidence procedure does not ensure those groups match a scientifically meaningful population partition. This is why the laboratory reports grouping agreement and interval coverage separately.

## 14.4 A sensitivity study must challenge an assumption

The independence construction uses the true noise scale. Suppose instead that the injected noise has variance $s^2I_n$ while the original observation variance remains $\sigma^2I_n$. The normalized components have covariance

$$
\operatorname{Cov}(X^D,X^I)=(\sigma^2-s^2)I_n.
$$

The cancellation fails when $s\neq\sigma$. Substituting an estimate into both the randomization and the interval formula therefore requires further theory; the known-variance derivation does not automatically apply.

:::{admonition} Example 14.2 — Reproducibility is not a validity guarantee
:class: course-example

Suppose the true standard deviation is one, but the procedure uses $s=0.7$. Then $\operatorname{Cov}(X^D,X^I)=0.51I_n$. The discovery and inference components remain dependent. Using the same random seed on every computer will reproduce that dependence exactly.

A useful project reruns the null experiment at $s/\sigma\in\{0.7,1,1.3\}$ and reports the change in false rejection, including Monte Carlo uncertainty. It treats the misspecified cases as stress tests, not as cases covered by the exact theorem.
:::

This is a more informative extension than changing an arbitrary tuning parameter without explaining its role. In UU learning, a comparable study changes the gap between mixture priors; in preference learning, it changes comparison-graph connectivity or within-prompt dependence; in online learning, it changes stationarity or the feedback available to the learner.

## 14.5 Design a small but interpretable experiment

Use one main question and a short sequence of experiments:

1. **Reproduction:** implement or reproduce a central result under its stated assumptions.
2. **Controlled change:** vary one assumption or resource constraint and keep the rest fixed.
3. **Matched comparison:** use the same simulated datasets or random-number streams when comparing methods.
4. **Uncertainty:** report Monte Carlo standard errors or uncertainty across independent repetitions.
5. **Interpretation:** explain which part of the theoretical argument survives the change.

For a rejection rate $\widehat p$ from $B$ independent repetitions, the approximate Monte Carlo standard error is $\{\widehat p(1-\widehat p)/B\}^{1/2}$. With $B=400$ and a true rate near $0.05$, this is about $0.011$. A difference of one percentage point cannot be interpreted confidently from that experiment alone.

When comparing two procedures on the same replicates, use the replicate-wise difference rather than treating two estimated rates as independent. Record computational failures separately. Removing failed runs without disclosure changes the evaluated procedure and can hide instability.

## 14.6 What the final project should contain

The proposal should identify the target, a small set of assumptions, one baseline, and one planned modification. The reproduction checkpoint should show that the baseline works in a setting where its behaviour is understood. The final report should contain the mathematical setup, methods, experiment design, primary results, limitations, and instructions to regenerate the results.

Reusing an assignment implementation is appropriate when the report identifies what is reused and adds a new question or comparison. For example, extend the PU assignment by deliberately violating constant positive-labelling probability, or extend the clustering assignment by varying the discovery-information fraction. Comparing increasingly complicated methods without a fixed target is not a substitute for a research question.

Maintain a source record for data, a fixed seed strategy, software versions, the exact analysis script, and a description of the independent sampling unit. Repeated ratings from one prompt or annotator must not be counted as independent observations merely because they occupy separate rows.

## Information reserved for inference has a measurable cost

Conditional on a given pair of discovered groups, the standard error of the normalized inference contrast is inflated by $1/\sqrt{1-\tau}$ relative to using an unselected full observation with the same fixed groups. For $\tau=0.2,0.5,0.8$, the factors are approximately $1.12,1.41,2.24$.

More discovery information can improve the grouping, while less remaining inference information widens a conditional interval. Actual groups can change with $\tau$, so these factors are not an unconditional power comparison between different discovered partitions. Choose the split through a declared design, not by repeatedly examining the eventual inferential result.

:::{admonition} Boundary - protect the reserved information
:class: course-boundary

Once the inference component has influenced feature choice, group selection, or tuning, the original independence argument no longer describes the full workflow. Fix those choices using discovery information, and use inference information only for the stated assessment or account for the additional selection.
:::

## 14.7 Reproducible Laboratory

The base-R script studies Gaussian thinning at three discovery fractions. It reports null rejection, selected-target interval coverage, interval length, estimation error, and a separate co-membership agreement measure under a simulated mean structure. It then challenges the known-noise assumption.

The intervals target the means of the discovered index sets. The script computes those targets from the simulated $\mu_i$ rather than substituting the original generating group centres. All numerical results are produced by the downloadable script.

[Download lab](../_static/labs/14-research-synthesis.R)

:::{literalinclude} ../../labs/14-research-synthesis.R
:language: r
:caption: Gaussian information splitting, selected-target coverage, and scale sensitivity
:::

### Computed output

:::{include} ../_generated/14-research-synthesis-output.md
:::

## Exercises

**Exercise 14.1.** Derive the means, variances, and covariance of $A$ and $B$. Which step requires Gaussianity to conclude independence?

**Exercise 14.2.** With fixed discovered groups, derive the variance of $\widehat\delta_C$. What happens to inference precision as $\tau$ approaches one?

**Exercise 14.3.** Explain why coverage must be evaluated against $\delta_C$ rather than the difference between the two generating population centres when recovery is imperfect.

**Exercise 14.4.** Choose one earlier course module and write a five-sentence project specification: target, observation mechanism, baseline, assumption to challenge, and main evaluation measure.

## Solutions

:::{admonition} Solution to Exercise 14.1
:class: dropdown

Linearity gives the stated means. Independence of $X$ and $E$ gives $\operatorname{Var}(A)=\tau^2\sigma^2I+\tau(1-\tau)\sigma^2I=\tau\sigma^2I$, with the analogous expression for $B$. The cross-covariance terms cancel. Zero covariance alone is insufficient in general; joint Gaussianity converts it into independence.
:::

:::{admonition} Solution to Exercise 14.2
:class: dropdown

Each inference observation has variance $\sigma^2/(1-\tau)$ and the two group averages involve disjoint independent rows. Adding their variances gives $v_C$. As $\tau$ approaches one, this variance diverges for fixed group sizes. Better discovery can therefore be accompanied by much wider intervals.
:::

:::{admonition} Solution to Exercise 14.3
:class: dropdown

The test and interval use the estimated index sets. If they mix generating populations, the corresponding averages of $\mu_i$ are mixtures of the generating centres, and their difference can be much smaller. Scoring the interval against another target confounds inference error with clustering recovery and does not evaluate the stated coverage guarantee.
:::

:::{admonition} Solution to Exercise 14.4
:class: dropdown

One possible specification is: “Estimate binary-classification risk from two unlabelled mixtures. Observe independent mixture samples with known, distinct priors and shared class-conditionals. Reproduce the unbiased UU risk estimator. Shrink the prior gap while holding sample sizes and component distributions fixed. Evaluate empirical bias, variance, and downstream excess risk, reporting uncertainty across repetitions.” A complete project would then specify the loss, target prior, model class, and finite-sample implementation.
:::

## Reading and course conclusion

- Neufeld, A., Dharamshi, A., Gao, L. L., and Witten, D. (2024). *Data Thinning for Convolution-Closed Distributions*. **JMLR 25**(57), 1–35. [Paper](https://jmlr.org/papers/v25/23-0446.html) and [author code](https://github.com/anna-neufeld/datathin_paper).

The recurring statistical task is to identify what the observations reveal under stated assumptions, then construct a learning or inferential procedure that respects that information. The final presentation should make that chain visible from the research question to the reported result.
