# 13. Selective inference after k-means

## Learning objectives

This chapter develops three skills: translating an iterative learning algorithm into a selection event, distinguishing a selected mean contrast from clustering stability, and evaluating an implementation whose inferential guarantee depends on the algorithmic history.

We retain the known-variance Gaussian model and selected-group mean target from Chapter 12. The new difficulty is computational. A hierarchical procedure selects a sequence of merges. Lloyd's k-means procedure alternates assignments and centre updates. The event describing its output therefore has a different structure.

## 13.1 Lloyd's algorithm is part of the statistical procedure

Given initial centres $c_1^{(0)},\ldots,c_K^{(0)}$, Lloyd's algorithm alternates

$$
g_i^{(t)}
=\mathop{\mathrm{argmin}}_{k}
\|x_i-c_k^{(t-1)}\|^2
$$

and

$$
c_k^{(t)}
=\frac{1}{|\{i:g_i^{(t)}=k\}|}
\sum_{i:g_i^{(t)}=k}x_i.
$$

Tie handling, empty-cluster handling, initialization, the iteration limit, and stopping rules complete this description. Each assignment step and centre update cannot increase the within-cluster sum of squares when the algorithm is well defined. Monotonic improvement does not guarantee a global optimum.

Different software can implement different algorithms under the name k-means. Base R's default is not Lloyd's algorithm. Even when Lloyd is requested, an inferential implementation may need information that ordinary fitting software discards. The [KmeansInference tutorial](https://yiqunchen.github.io/KmeansInference/articles/Tutorials.html) explains why its fitting function retains the intermediate assignments.

:::{admonition} Example 13.1 — The final partition does not describe the full path
:class: course-example

Take one-dimensional observations $(-3,-1,1,4)$ and initialize centres at $-3$ and $-1$. The first assignment puts $-3$ alone and the other three observations together. Updating gives centres $-3$ and $4/3$. The next assignment gives groups $\{-3,-1\}$ and $\{1,4\}$, whose updated centres are $-2$ and $2.5$.

Starting instead at $-3$ and $4$ reaches the same final groups immediately. These runs share a final partition but not an assignment history. A test that conditions on the full path must describe the path actually used.
:::

## 13.2 Why the fitting criterion rewards the naïve test statistic

For $K=2$, the sum-of-squares decomposition is particularly informative. Let $n_1,n_2$ be cluster sizes, $\bar x$ the overall mean, and $D=\bar x_{C_1}-\bar x_{C_2}$. Then

$$
\operatorname{TSS}
=\operatorname{WCSS}+\operatorname{BSS},\qquad
\operatorname{BSS}=\frac{n_1n_2}{n_1+n_2}\|D\|^2.
$$

The ordinary known-variance Wald statistic for fixed groups is

$$
W=\frac{\|D\|^2}{\sigma^2(1/n_1+1/n_2)}
=\frac{\operatorname{BSS}}{\sigma^2}.
$$

For a fixed dataset, minimizing within-cluster variation therefore increases precisely the between-group statistic that a naïve test treats as unselected. The problem is built into the optimization objective; it is not only a peculiarity of one software implementation.

:::{admonition} Example 13.2 — Optimization and significance share a statistic
:class: course-example

Suppose $q=2$, $\sigma=1$, the total sum of squares is 80, and a two-cluster solution has within-cluster sum of squares 48. Then $\operatorname{BSS}=32$ and $W=32$. A fixed-group chi-square calculation would give $e^{-16}\approx1.13\times10^{-7}$.

These illustrative values describe the calculation, not an executed simulation result. Searching for a small WCSS also searches for a large $W$. A selective analysis must account for that search before interpreting the small reference tail as evidence.
:::

## 13.3 Conditioning on the assignment history

[Chen and Witten (2023)](https://www.jmlr.org/papers/v24/22-0371.html) develop an efficiently computable selective test by conditioning on the intermediate assignments of k-means. The target remains

$$
H_0^{C_1,C_2}:\bar\mu_{C_1}=\bar\mu_{C_2}
$$

for the selected groups, under the stated Gaussian model. Their method is specific to the algorithm and conditioning event; a hierarchical-clustering formula cannot simply be reused.

Use the same contrast vector $\nu$, observed direction $u$, and radius perturbation as before:

$$
x'(\phi)=P_\nu^\perp x+\frac{\nu}{\|\nu\|^2}\phi u^\top.
$$

Now require the algorithm to reproduce its observed assignment history as $\phi$ varies. Along this path each observation is affine in $\phi$:

$$
x_i'(\phi)=a_i+b_i\phi.
$$

Once a previous assignment is fixed, each updated centre is an average of such observations and is also affine in $\phi$. The condition that observation $i$ chooses centre $k$ instead of centre $\ell$ is

$$
\|a_i+b_i\phi-c_k(\phi)\|^2
\leq
\|a_i+b_i\phi-c_\ell(\phi)\|^2.
$$

Both sides are quadratic in $\phi$. Intersecting the constraints across observations, candidate centres, and iterations yields an admissible set of radii. Integrating the appropriate scaled-chi density over this set gives the selective upper tail.

This derivation explains the computational opportunity without claiming that the original selection event is polyhedral in the full data matrix. Quadratic inequalities along a one-dimensional path can be manageable even when a general multidimensional description is complicated.

## 13.4 The choices outside the path also matter

The first demonstration fixes $K$, the features, the noise scale, and the initialization rule. The package's fitting and inference routines use the same seed and iteration limit. A selected run from many random restarts is a different procedure: the comparison between restarts is another selection step.

Similarly, choosing $K$ from a silhouette plot, selecting coordinates after inspecting separation, or learning a transformation and treating it as known can move beyond the stated guarantee. This does not mean those actions are forbidden. It means that a theorem for one specified pipeline does not automatically cover an expanded pipeline.

Conditional validity also has a cost. Conditioning on more of the path can discard information. A procedure with a finer conditioning event may be easier to compute but have lower power than a feasible procedure conditioning less. There is no universal rule that additional conditioning is free.

## 13.5 Three different assessments of a clustering

| Question | Example quantity | Interpretation |
|---|---|---|
| Does the algorithm produce a similar partition under perturbation? | Pairwise co-membership agreement across runs | Algorithmic or sampling stability |
| Did it recover a stipulated grouping? | Adjusted Rand index against known simulation labels | Recovery relative to that reference |
| Is a selected mean contrast inconsistent with its null? | Selective p-value | Evidence under a specified sampling model and selection event |

None of these quantities substitutes for the others. A stable partition of a continuous cloud need not reveal a mixture of populations. An unstable algorithm can still be paired with a correctly calibrated selected-hypothesis test. Recovery metrics require a reference grouping, while an inferential null must identify the relevant population parameter.

For a simulation under alternatives, report how often the intended groups are recovered and what rejection probability is obtained conditional on that recovery, or otherwise describe the actual selected contrasts. This prevents a favourable rejection rate from hiding a changing target.

## 13.6 Reproducible R laboratory

The base-R component implements a transparent two-centre Lloyd trace and verifies that its objective does not increase. It compares initializations using a label-invariant co-membership measure and demonstrates naïve null rejection after learning groups.

If <code>KmeansInference</code> is installed, an additional block executes the authors' fitting-dependent selective procedure, with fixed arguments, and reports paired naïve and selective results for that same procedure. The script distinguishes this extension from the base-R diagnostic. Initialization agreement is not assumed between two separately written implementations.

[Download R lab](../_static/labs/13-kmeans-inference.R)

:::{literalinclude} ../../labs/13-kmeans-inference.R
:language: r
:caption: Lloyd's assignment path, naïve calibration, and the authors' selective method
:::

### Computed output

:::{include} ../_generated/13-kmeans-inference-output.md
:::

## Exercises

**Exercise 13.1.** Prove the two-cluster identity $\operatorname{BSS}=n_1n_2\|D\|^2/(n_1+n_2)$.

**Exercise 13.2.** Expand one assignment inequality along $x'(\phi)$ and show that it is a quadratic inequality. Why can its feasible set be disconnected?

**Exercise 13.3.** A software function returns a final partition but no iteration history. Is that enough to implement the path-conditioned procedure described here? What extra information is required?

**Exercise 13.4.** Construct a research report outline that presents stability, recovery, and selective testing without treating one as proof of another. Specify the null-model part of the experiment.

## Solutions

:::{admonition} Solution to Exercise 13.1
:class: dropdown

Since $\bar x=(n_1\bar x_1+n_2\bar x_2)/n$, we have $\bar x_1-\bar x=(n_2/n)D$ and $\bar x_2-\bar x=-(n_1/n)D$. Therefore

$$
n_1\|\bar x_1-\bar x\|^2+n_2\|\bar x_2-\bar x\|^2
=\frac{n_1n_2^2+n_2n_1^2}{n^2}\|D\|^2
=\frac{n_1n_2}{n}\|D\|^2.
$$
:::

:::{admonition} Solution to Exercise 13.2
:class: dropdown

Write the two difference vectors as $r_k+s_k\phi$ and $r_\ell+s_\ell\phi$. Subtracting their squared norms gives

$$
(\|s_k\|^2-\|s_\ell\|^2)\phi^2
+2(r_k^\top s_k-r_\ell^\top s_\ell)\phi
+\|r_k\|^2-\|r_\ell\|^2\leq0.
$$

A downward-opening quadratic can be nonpositive outside its two roots, producing two rays. Intersections of multiple such conditions with $\phi\geq0$ can produce multiple intervals. A single truncation interval cannot be assumed.
:::

:::{admonition} Solution to Exercise 13.3
:class: dropdown

No. The procedure must recover the initial-centre rule, intermediate assignments, and relevant stopping and tie rules so that it can express the same selection event. Re-running a documented deterministic algorithm from preserved initialization information may recover the history; a bare final partition generally cannot identify it.
:::

:::{admonition} Solution to Exercise 13.4
:class: dropdown

First assess null calibration with independent $N_q(0,\sigma^2I_q)$ rows and a known noise scale. Report selective rejection frequency with Monte Carlo uncertainty and all failures. Next introduce a documented mean structure, report recovery of its reference groups, and distinguish conditional power from unconditional rejection. Finally vary initialization or perturbations and show a separate co-membership stability measure. Keep the three headings and interpretations separate.
:::

## Reading and next step

- **Core:** Chen, Y. T., and Witten, D. M. (2023). *Selective inference for k-means clustering*. **JMLR 24**(152), 1–41. [Paper and code](https://www.jmlr.org/papers/v24/22-0371.html).
- **Implementation:** [KmeansInference](https://yiqunchen.github.io/KmeansInference/) and its [software tutorials](https://yiqunchen.github.io/KmeansInference/articles/Tutorials.html).

The final chapter turns the course's methods into a reproducible research workflow and develops an independent-information comparison using Gaussian data thinning.
