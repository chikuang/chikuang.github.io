# Computational workflow and reproducibility

The required experiments use **base R** and run on an ordinary laptop. The hierarchical-clustering and k-means chapters also show how to call the authors' specialist packages; those additional calculations run when the packages are installed. The page output states whether those optional calculations actually ran.

## 1. Run a worked experiment

Install R from [the R Project](https://www.r-project.org/) or use an existing installation. RStudio is convenient but optional. Download a chapter's `.R` file, open a fresh R session in that folder, and run:

```r
source("03-pu-learning.R", echo = FALSE)
sessionInfo()
```

From a terminal, the same calculation is:

```bash
Rscript --vanilla 03-pu-learning.R
```

Each script sets its own seed and prints its results. The outputs embedded in this book come from running those scripts, rather than from manually typed illustrative results. Small floating-point differences between platforms are possible. The [lab bundle](_static/downloads/statistical-learning-labs.zip) contains all scripts, recorded outputs, and an execution manifest with the R version and file hashes.

If a script honours `COURSE_FIGURE_DIR`, it can also save its figure:

```r
dir.create("figures", showWarnings = FALSE)
Sys.setenv(COURSE_FIGURE_DIR = normalizePath("figures"))
source("11-selective-inference.R")
```

## 2. Predict, reproduce, change, interpret

For every lab, use the same four-step workflow.

1. **Predict:** state the expected direction of a result and the assumption supporting that prediction.
2. **Reproduce:** run the worked script unchanged and check the printed diagnostic.
3. **Change:** alter one data-generating feature, comparator, or selection rule. Keep the remaining parts fixed.
4. **Interpret:** identify whether the change affected identification, estimation error, decision performance, or inferential validity.

Keep a worked example separate from your independent solution. Passing a numerical check establishes only what that check measures. A transfer experiment and a statistical explanation remain part of the exercise.

## 3. Keep the target outside the learner

In a simulation, the evaluator knows the true class labels, class prior, arm means, or null hypothesis. The learner receives only the information allowed by the observation model. For example, UCB may see the reward from the arm it selects; its action rule cannot read the true arm means used later to calculate pseudo-regret.

For a PU experiment, generate the labelled positive sample and the unlabelled sample under the stated mechanism. Use a separately generated labelled test sample for evaluation. Do not treat missing labels as negative labels without recognising that this changes the target.

## 4. Quantify simulation error

If a rejection rate is estimated from independent repetitions, a basic Monte Carlo standard error is:

$$
\widehat{\operatorname{se}}(\widehat r)
=\sqrt{\frac{\widehat r(1-\widehat r)}{B}}.
$$

At a true rejection rate near 0.05 and $B=500$, this is about 0.0097. A result of 0.06 is therefore not strong evidence of a small systematic calibration defect by itself. State the repetition count and include an interval or an uncertainty bar. When comparing two procedures, using the same simulated dataset for both gives a paired comparison; analyse the paired differences.

## 5. Make a small reproducible report

Use Quarto, R Markdown, or another transparent report format. The website itself is built with Sphinx; students may use R-oriented tools for their reports. A minimal Quarto source is available as [report-template.qmd](_static/downloads/report-template.qmd).

Your report should identify:

- the observed data, target, and assumptions;
- the method and a meaningful baseline;
- the simulation or data source and all preprocessing;
- training and evaluation separation;
- results with uncertainty and one failure or sensitivity analysis;
- the command needed to regenerate the report and figures.

Record the software session with `sessionInfo()`. For a larger project, an `renv` lockfile can record R package versions. Do not include private datasets or credentials in a public repository.

## 6. Optional clustering-inference packages

Read the authors' installation instructions and package tutorials before using them. These procedures encode specific algorithms and selection events.

- [`clusterpval`](https://lucylgao.github.io/clusterpval/): hierarchical clustering, with a [technical explanation](https://lucylgao.github.io/clusterpval/technical.html).
- [`KmeansInference`](https://yiqunchen.github.io/KmeansInference/): selected contrasts after a documented k-means procedure, with [worked tutorials](https://yiqunchen.github.io/KmeansInference/articles/Tutorials.html).

An ordinary `hclust` or `kmeans` result does not automatically satisfy another procedure's conditioning requirements. Specify the distance, linkage or update rule, initialisation, number of clusters, and rule for selecting a pair to test.

## Readiness check

Before starting the course, verify that you can simulate a normal mixture, fit a logistic regression, write a function with explicit inputs, summarise repeated simulations, and derive a conditional expectation by partitioning on a binary label.

Try this check without looking at the answer: simulate $Y\sim\operatorname{Bernoulli}(0.3)$ and $X\mid Y\sim N(2Y-1,1)$. Estimate the error of the threshold rule $\widehat Y=1\{X>0\}$. How does the theoretical error depend on the class prior?

:::{admonition} Readiness-check answer
:class: dropdown

The two class-specific errors are both $\Phi(-1)$, so the population error is $0.3\Phi(-1)+0.7\Phi(-1)=\Phi(-1)\approx0.1587$. The prior cancels because this particular model and threshold are symmetric. Moving the threshold, changing the variances, or changing the separation can remove that cancellation.
:::
