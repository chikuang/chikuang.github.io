# Interactive demonstrations

Change a setting, predict its effect, and then examine the result. These small browser experiments expose a mechanism; the chapters develop the assumptions and proofs. Every experiment uses simulated data and a visible random seed. The displays run locally in your browser.

## 1. Recovering risk from incomplete labels

Let $X\mid Y=+1\sim N(\delta/2,1)$ and $X\mid Y=-1\sim N(-\delta/2,1)$. For the threshold classifier $g_t(x)=1\{x>t\}$, the true misclassification risk is

$$
R(t)=\pi\Phi(t-\delta/2)+(1-\pi)\{1-\Phi(t+\delta/2)\}.
$$

The true curve is available because this is a simulation. A PU learner receives a representative positive sample and an independent sample from the full feature marginal. A UU learner receives two unlabelled mixtures with shared class-conditionals and known priors $a$ and $b$.

Try a wrong supplied prior in PU mode. In UU mode, move the two mixture priors together. At $a=b$, the two-mixture inversion is unidentified. An empirical corrected risk can lie outside $[0,1]$ even though a population misclassification probability cannot.

:::{raw} html
<section class="course-demo" id="risk-demo" aria-label="Incomplete-label risk experiment">
<div class="demo-controls">
<label for="risk-kind">Observation model<select id="risk-kind"><option value="pu">Positive–unlabelled (PU)</option><option value="uu">Two unlabelled mixtures (UU)</option></select></label>
<label for="risk-delta">Class separation δ <output id="risk-delta-value">2</output><input id="risk-delta" type="range" min="0" max="4" step="0.2" value="2"></label>
<label for="risk-pi">Target prior π <output id="risk-pi-value">0.4</output><input id="risk-pi" type="range" min="0.1" max="0.9" step="0.05" value="0.4"></label>
<label for="risk-assumed" id="risk-assumed-control">Supplied PU prior <output id="risk-assumed-value">0.4</output><input id="risk-assumed" type="range" min="0.1" max="0.9" step="0.05" value="0.4"></label>
<label for="risk-a" class="risk-uu-control" hidden>Mixture prior a <output id="risk-a-value">0.8</output><input id="risk-a" type="range" min="0.05" max="0.95" step="0.05" value="0.8"></label>
<label for="risk-b" class="risk-uu-control" hidden>Mixture prior b <output id="risk-b-value">0.2</output><input id="risk-b" type="range" min="0.05" max="0.95" step="0.05" value="0.2"></label>
<label for="risk-n">Observations per sample<select id="risk-n"><option>100</option><option selected>400</option><option>1600</option></select></label>
<label for="risk-seed">Random seed<input id="risk-seed" type="number" min="1" max="999999" step="1" value="42"></label>
</div>
<button id="risk-resample" type="button" class="demo-button">New sample</button>
<div id="risk-chart" class="demo-chart"></div>
<p id="risk-status" class="demo-status" aria-live="polite"></p>
<p class="demo-caption">Black: true risk. Pink: empirical corrected risk. Gold in PU mode: non-negative correction. Each curve evaluates the same fixed threshold grid; choosing its minimum introduces an additional selection step.</p>
</section>
<noscript><p>Enable JavaScript for the interactive controls, or run the downloadable R labs in Chapters 3–5.</p></noscript>
:::

The PU display uses $\widehat\pi\widehat E_+\ell_+ +\widehat E_U\ell_- -\widehat\pi\widehat E_+\ell_-$. Its non-negative version truncates only the estimated negative-class contribution. The UU display uses the mixture inversion from [Chapter 5](chapters/05-uu-learning.md), with the specified target prior. See [Chapters 3–4](chapters/03-pu-learning.md) for PU assumptions and finite-sample qualifications.

## 2. Exploration with one-arm feedback

Three stationary Bernoulli arms have means $(0.30,0.45,\mu_3)$. Each round reveals only the reward from the selected arm. Compare round-robin uniform exploration with UCB, which first pulls each arm once and then maximises

$$
\widehat\mu_k(t-1)+\sqrt{2\log(t)/N_k(t-1)}.
$$

The plot shows cumulative **pseudo-regret** $\sum_{t=1}^T(\mu^\star-\mu_{A_t})$. Only the simulator uses the true means. Both procedures use the same pre-generated reward stream for each arm, indexed by that arm's pull count, to support a paired comparison.

:::{raw} html
<section class="course-demo" id="bandit-demo" aria-label="Bernoulli bandit experiment">
<div class="demo-controls">
<label for="bandit-mu">Third-arm mean <output id="bandit-mu-value">0.65</output><input id="bandit-mu" type="range" min="0.1" max="0.9" step="0.05" value="0.65"></label>
<label for="bandit-horizon">Rounds<select id="bandit-horizon"><option>100</option><option selected>500</option><option>2000</option></select></label>
<label for="bandit-seed">Random seed<input id="bandit-seed" type="number" min="1" max="999999" step="1" value="81"></label>
</div>
<button id="bandit-resample" type="button" class="demo-button">New reward streams</button>
<div id="bandit-chart" class="demo-chart"></div>
<div id="bandit-status" class="demo-status" aria-live="polite"></div>
<p class="demo-caption">Black: uniform exploration. Pink: UCB. One trajectory does not establish an expected-regret guarantee. UCB need not win on every finite sample or every short horizon.</p>
</section>
:::

This is a reward-bandit demonstration. A duelling-bandit learner observes a comparison outcome instead of an arm reward; it needs an explicit winner criterion and a corresponding performance definition. See [Chapter 9](chapters/09-bandits.md) and [Chapter 10](chapters/10-duelling-bandits.md).

## 3. Selection changes the reference distribution

Consider independent $Z_1,\ldots,Z_p\sim N(0,1)$ and select $J=\arg\max_j|Z_j|$. Conditional on the remaining coordinates, the selected statistic must satisfy $|Z_J|\ge m$, where $m=\max_{k\ne J}|Z_k|$. The selected two-sided p-value is

$$
p_{\mathrm{sel}}=\frac{1-\Phi(|Z_J|)}{1-\Phi(m)},
\qquad p_{\mathrm{naive}}=2\{1-\Phi(|Z_J|)\}.
$$

For $p=1$, take $m=0$ and the two p-values agree. Under this independent all-null model, selecting the smallest naive p-value produces rejection probability $1-(1-\alpha)^p$. The repeated simulation reports a Monte Carlo standard error alongside each rejection rate.

:::{raw} html
<section class="course-demo" id="selection-demo" aria-label="Elementary Gaussian selection experiment">
<div class="demo-controls">
<label for="selection-p">Candidate statistics p <output id="selection-p-value">10</output><input id="selection-p" type="range" min="1" max="30" step="1" value="10"></label>
<label for="selection-alpha">Test level α<select id="selection-alpha"><option>0.01</option><option selected>0.05</option><option>0.10</option></select></label>
<label for="selection-seed">Random seed<input id="selection-seed" type="number" min="1" max="999999" step="1" value="23"></label>
</div>
<button id="selection-resample" type="button" class="demo-button">New selected statistic</button>
<div id="selection-chart" class="demo-chart"></div>
<div id="selection-status" class="demo-status" aria-live="polite"></div>
<p class="demo-caption">This is an elementary Gaussian selection example, not a clustering-significance calculator. Hierarchical clustering and k-means require their own algorithm-specific selection events and conditioning.</p>
</section>
:::

The display numerically evaluates the normal tail and repeats the null experiment 3,000 times. [Chapter 11](chapters/11-selective-inference.md) develops the conditional calculation. [Chapters 12–13](chapters/12-hierarchical-clustering.md) explain what changes for selected cluster contrasts.
