# 12. Selective inference after hierarchical clustering

## Learning objectives

After this chapter, you should be able to define a selected-group mean contrast, explain why ordinary cluster comparisons can fail, derive the one-dimensional radial perturbation underlying a selective test, and distinguish exact known-variance inference from a plug-in analysis.

Clustering constructs labels from observations. If those labels are then treated as pre-existing scientific groups, an ordinary test ignores the construction step. A visually separated partition is especially easy to produce because separation is an objective of the clustering algorithm itself.

## 12.1 State the model before interpreting a cluster

Consider $n$ independent observations of $q$ features:

$$
X_i\sim N_q(\mu_i,\sigma^2 I_q),\qquad
i=1,\ldots,n,\quad \sigma>0\text{ known}.
$$

The means $\mu_i$ are fixed unknown vectors and may differ across observations. This is a Gaussian mean model, not an assumed finite-mixture model. Let $\mathcal C(X)$ denote the partition returned by a specified clustering algorithm with a fixed number of clusters $K$.

For two non-overlapping index sets $C_1,C_2$ appearing in that partition, define

$$
\bar\mu_C=\frac1{|C|}\sum_{i\in C}\mu_i,\qquad
H_0^{C_1,C_2}:\bar\mu_{C_1}=\bar\mu_{C_2}.
$$

The null concerns the underlying means of the observations in these particular groups. It does not say that the entire population has one cluster, that $K$ is correct, or that the groups represent two biological types. Those are different scientific questions requiring other models or evidence.

The framework and exact hierarchical-clustering procedure were developed by [Gao, Bien, and Witten (2024)](https://doi.org/10.1080/01621459.2022.2116331). The article appeared online in 2022; the journal volume is dated 2024.

## 12.2 Why the ordinary reference distribution fails

For fixed groups, let

$$
\nu_i=\frac{\mathbf1(i\in C_1)}{|C_1|}
      -\frac{\mathbf1(i\in C_2)}{|C_2|},
\qquad
D=X^\top\nu=\bar X_{C_1}-\bar X_{C_2}.
$$

Then $\|\nu\|^2=|C_1|^{-1}+|C_2|^{-1}$ and, under the fixed-group null,

$$
\frac{\|D\|^2}{\sigma^2\|\nu\|^2}\sim\chi_q^2.
$$

After clustering, $\nu$ depends on $X$. Substituting the selected $\nu$ into the fixed-group formula generally changes its null distribution. This is not solved by declaring the selected labels fixed *after* looking at the data: the repeated-sampling procedure still selects them.

:::{admonition} Example 12.1 — A significant difference created by the partition
:class: course-example

Suppose two estimated groups each contain ten observations, $\sigma=1$, and their observed mean difference is $(1.2,0)^\top$. A fixed-group calculation gives

$$
W=\frac{1.2^2}{1/10+1/10}=7.2,\qquad
\Pr(\chi_2^2\geq7.2)=e^{-3.6}\approx0.0273.
$$

This number does not account for the fact that the groups were chosen to separate the observations. Group sizes and means alone are insufficient to compute the selective p-value: the clustering algorithm and the geometry of the full dataset determine which alternative datasets would produce the same groups.
:::

## 12.3 Reduce the problem to one varying distance

Write the observed contrast length as $r=\|x^\top\nu\|$ and its direction as $u=x^\top\nu/r$. Under the continuous model, $r=0$ has probability zero. Define

$$
P_\nu^\perp=I_n-\frac{\nu\nu^\top}{\|\nu\|^2}.
$$

The data decompose into a component orthogonal to the contrast and a component along it:

$$
X=P_\nu^\perp X+\frac{\nu}{\|\nu\|^2}(X^\top\nu)^\top.
$$

Condition on $P_\nu^\perp X=P_\nu^\perp x$ and on the direction of $X^\top\nu$ being $u$. The remaining scalar is the contrast length. A dataset with candidate length $\phi\geq0$ is

$$
x'(\phi)=P_\nu^\perp x+\frac{\nu}{\|\nu\|^2}\phi u^\top
=x+\frac{\nu}{\|\nu\|^2}(\phi-r)u^\top.
$$

Under the null, before imposing the clustering event, this length follows

$$
\Phi_{\mathrm{rad}}\sim \sigma\|\nu\|\chi_q.
$$

The notation $\Phi_{\mathrm{rad}}$ denotes a random radius here, not the normal CDF. Gaussian independence separates the orthogonal component, and spherical symmetry under the null separates radius from direction.

:::{admonition} Example 12.2 — What the perturbation actually moves
:class: course-example

If both groups have size ten, $\|\nu\|^2=0.2$. Increasing the contrast length by $\Delta$ moves every observation in $C_1$ by $+\Delta u/2$ and every observation in $C_2$ by $-\Delta u/2$. Observations outside these groups remain unchanged.

Within-group pairwise differences remain unchanged. The selected groups are pulled apart or pushed together while preserving the nuisance information on which we conditioned. With unequal sizes, the two shifts have unequal magnitudes so that the overall weighted mean is still preserved.
:::

## 12.4 The truncation set remembers the algorithm

Define

$$
\mathcal S(x)=
\{\phi\geq0:C_1,C_2\in\mathcal C(x'(\phi))\}.
$$

A selective p-value is the conditional upper tail

$$
p_{\mathrm{sel}}
=\frac{
\int_{\mathcal S(x)\cap[r,\infty)}
 f_{\sigma\|\nu\|\chi_q}(t)\,dt}
{\int_{\mathcal S(x)}
 f_{\sigma\|\nu\|\chi_q}(t)\,dt}.
$$

The denominator conditions on observing those groups. The additional conditioning above removes nuisance parameters. Computing $\mathcal S$ is the algorithmic challenge: as $\phi$ varies, different merges can occur in the dendrogram.

For several hierarchical linkages with squared Euclidean distance, the authors provide an efficient exact characterization. The selected set can contain multiple intervals. A coarse grid of radii is useful for visualization, but it is not an exact replacement for that characterization; narrow intervals or tail intervals can be missed.

For teaching, use average linkage and squared Euclidean distance exactly as in the [authors' tutorial](https://lucylgao.github.io/clusterpval/HierCluster.html). Changing the metric while retaining a formula derived for another metric is not a harmless implementation choice.

## 12.5 What must be fixed or accounted for?

The first laboratory fixes $K=2$, leaving a single pair. It also fixes the distance, linkage, features, and noise scale. If an analyst searches across values of $K$, chooses features because they make the separation look stronger, or selects a pair after inspecting its test statistic, the stated conditioning event may no longer describe the full procedure.

The package can estimate the noise scale when it is not supplied. That practical default does not convert an unknown-variance problem into the finite-sample known-variance theorem. Our simulation supplies $\sigma=1$. An application should distinguish an exact guarantee under known covariance from an approximation or asymptotic argument involving estimated covariance.

Likewise, testing every pair introduces a multiple-testing problem. A valid selected-pair p-value alone does not assert familywise or false-discovery-rate control for an arbitrary collection of pairwise tests.

## 12.6 Why ordinary row splitting can still fail

Imagine clustering training observations, training a classifier to reproduce those labels, assigning held-out observations, and testing the held-out features between assigned groups. The assignment uses each held-out observation's own features. Those same features then determine both the group and the test statistic. The null for the selected held-out groups remains data-dependent.

Independent replicate measurements of the same units can provide a different route: discover groups from one measurement and compare their underlying means with an independent measurement, under a suitable joint model. Distribution-specific data thinning is another approach discussed in Chapter 14. Neither construction follows merely from calling some rows a test set.

## 12.7 Reproducible R laboratory

The required base-R experiment contrasts a fixed partition with average-linkage clustering under one Gaussian cloud. It demonstrates the calibration failure directly. When the authors' <code>clusterpval</code> package is installed, the same script additionally executes its exact known-variance procedure and records how many runs succeeded. It reports explicit status if that extension is unavailable; it never labels a base-R naïve test as selective inference.

[Download R lab](../_static/labs/12-hierarchical-clustering.R)

:::{literalinclude} ../../labs/12-hierarchical-clustering.R
:language: r
:caption: Fixed versus learned groups, with an optional authors' exact selective procedure
:::

### Computed output

:::{include} ../_generated/12-hierarchical-clustering-output.md
:::

## Exercises

**Exercise 12.1.** Derive $\operatorname{Var}(D)$ for fixed groups and obtain the fixed-group chi-square statistic.

**Exercise 12.2.** For $|C_1|=a$ and $|C_2|=b$, derive the two shifts induced by increasing $\phi$ by $\Delta$.

**Exercise 12.3.** An analyst tries five linkages and reports the smallest selective p-value computed separately for each. Explain why the individual procedure's guarantee does not justify that report.

**Exercise 12.4.** In a power simulation, estimated groups sometimes mix two generating populations. Which mean contrast is being tested? Why should recovery probability and rejection probability both be reported?

## Solutions

:::{admonition} Solution to Exercise 12.1
:class: dropdown

Independence gives $\operatorname{Var}(X^\top\nu)=\sum_i\nu_i^2\sigma^2I_q=\sigma^2(1/a+1/b)I_q$. Under the null the contrast mean is zero. Dividing by its common standard deviation gives $q$ independent standard normals, whose squared sum is $\chi_q^2$. The derivation assumes $\nu$ is fixed; selection is precisely what invalidates using it unchanged for learned groups.
:::

:::{admonition} Solution to Exercise 12.2
:class: dropdown

The multiplier is $\nu_i/(1/a+1/b)$. It equals $b/(a+b)$ in group 1 and $-a/(a+b)$ in group 2. Thus the shifts are $b\Delta u/(a+b)$ and $-a\Delta u/(a+b)$. Their weighted sum is zero, and their difference is $\Delta u$.
:::

:::{admonition} Solution to Exercise 12.3
:class: dropdown

Selecting the linkage by its p-value adds a data-dependent search. Each p-value was calibrated for its specified selection event, not for winning a competition across five analyses. The analyst must pre-specify the procedure or justify a method accounting for the additional search and the desired error criterion. Simply reporting the minimum retains neither the original conditional guarantee nor an automatic multiplicity guarantee.
:::

:::{admonition} Solution to Exercise 12.4
:class: dropdown

The target is the difference between averages of the underlying $\mu_i$ over the *estimated* index sets. That can differ from the difference between the two generating population centres. Recovery probability describes whether the intended groups were obtained; conditional power given recovery describes detection after that event. A single unconditional rejection rate mixes these questions and can obscure a poor clustering procedure.
:::

## Reading and next step

- **Core:** Gao, L. L., Bien, J., and Witten, D. (2024). *Selective Inference for Hierarchical Clustering*. **JASA 119**(545), 332–342. [Open manuscript](https://arxiv.org/abs/2012.02936).
- **Implementation:** [clusterpval technical details](https://lucylgao.github.io/clusterpval/technical.html) and [hierarchical tutorial](https://lucylgao.github.io/clusterpval/HierCluster.html).

The next chapter replaces a sequence of hierarchical merges with the iterative assignment history of Lloyd's k-means algorithm.
