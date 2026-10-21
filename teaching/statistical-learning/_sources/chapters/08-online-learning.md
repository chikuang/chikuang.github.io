# 8. Online learning and regret

:::{admonition} Core idea - Regret is defined by the protocol and comparator
:class: course-core

An online decision must use only past information. Its guarantee compares accumulated loss with a specified benchmark under a specified feedback model. Changing the information revealed after a round or allowing the comparator to change decisions changes the problem.
:::

**Question.** How should a learner act when each decision must precede the next observation?

[Download lab](../_static/labs/08-online-learning.R)

Batch learning chooses a rule after observing a training sample. Online learning specifies a sequence: act, observe feedback, update, and act again. The feedback might contain a full loss function or only a partial observation. This chapter begins with full information, where the mathematical distinction between prediction and adaptation is easiest to see.

The central target is cumulative loss relative to a declared comparator. Small regret is a precise guarantee, but its meaning depends on what that comparator is allowed to do.

## The information available at each round

```{figure} ../_static/diagrams/online-protocol.*
:alt: The action precedes the feedback. The full-information and bandit settings differ in what is revealed after acting; a valid update must respect that boundary.
:class: course-flow
:width: 100%

The action precedes the feedback. The full-information and bandit settings differ in what is revealed after acting; a valid update must respect that boundary.
```

## 8.1 · Write the protocol before the algorithm

At round $t$, the learner chooses $w_t\in\mathcal W$ using information available before the new loss. A loss function $\ell_t$ is revealed, the learner incurs $\ell_t(w_t)$, and it updates its decision. In the full-information setting it can use the observed loss function or an appropriate gradient. In prediction with experts, it sees every expert's loss after acting.

For a fixed decision set, define

$$
\operatorname{Regret}_T=
\sum_{t=1}^T\ell_t(w_t)
-\min_{w\in\mathcal W}\sum_{t=1}^T\ell_t(w).
$$ (eq-online-regret)

The comparator is the best **single fixed decision in hindsight**. This is not necessarily the best decision at each time. Regret can be negative, because the learner changes its decision while the comparator cannot. The useful goal is an upper bound growing more slowly than $T$.

The sequence need not be iid for the deterministic convex regret argument below. For randomized partial-feedback methods, specify what the environment can observe before choosing outcomes; an adversary that reacts to the current random action can change the guarantee.

## 8.2 · Learn from experts with Hedge

Suppose $K$ experts supply advice and their losses satisfy $L_{tk}\in[0,1]$. Hedge maintains weights

$$
p_{tk}=\frac{\exp\{-\eta\sum_{s<t}L_{sk}\}}
{\sum_j\exp\{-\eta\sum_{s<t}L_{sj}\}}.
$$

The learner's mixture loss is $\sum_kp_{tk}L_{tk}$. It is the conditional expected loss when an expert is sampled using $p_t$. With a convex loss of a numerical prediction, combining predictions can sometimes have loss no larger than the mixture loss, by Jensen's inequality; that is an additional property of the prediction setup.

Full-information feedback is essential to this update: all expert losses are observed after the round. The exponential-weights analysis yields a bound of the form $\log K/\eta+\eta T/8$ for bounded losses and the mixture-loss convention. Balancing the terms gives order $\sqrt{T\log K}$ regret.

:::{admonition} Example 8.1 · Good adaptation can still lose on the next round
:class: course-example

**Setup.** Two experts incur losses $(0,1)$ on the first round and $(1,0)$ on the second. Use uniform initial weights and $\eta=\log3$.

**Work it through.** Round-one mixture loss is $1/2$. The updated weights are $(3/4,1/4)$, so round-two mixture loss is $3/4$. Total loss is $1.25$; either fixed expert loses exactly 1, so regret is $0.25$.

**What this shows.** The weight update uses past evidence and cannot know that the next outcome reverses. A cumulative guarantee does not assert that every update improves the next prediction.
:::

## 8.3 · Derive projected online gradient descent

Let $\mathcal W$ be closed and convex with Euclidean diameter at most $D$. Suppose each $\ell_t$ is convex and a subgradient $g_t$ at $w_t$ has norm at most $G$. Projected OGD is

$$
w_{t+1}=\Pi_{\mathcal W}(w_t-\eta g_t).
$$

For any fixed comparator $u\in\mathcal W$, convexity gives $\ell_t(w_t)-\ell_t(u)\le g_t^{\mathsf T}(w_t-u)$. The projection inequality gives

$$
\|w_{t+1}-u\|^2\le\|w_t-u\|^2
-2\eta g_t^{\mathsf T}(w_t-u)+\eta^2\|g_t\|^2.
$$

Rearranging and summing cancels the intermediate squared distances. Consequently,

$$
\operatorname{Regret}_T\le\frac{D^2}{2\eta}
+\frac{\eta G^2T}{2}.
$$ (eq-ogd-bound)

For a known horizon, $\eta=D/(G\sqrt T)$ yields $DG\sqrt T$. The proof makes the assumptions visible: bounded domain, bounded subgradients, convex losses, and updates that occur after the current decision. Smoothness is not required for this bound.

:::{admonition} Example 8.2 · A guarantee with an explicit comparator
:class: course-example

**Setup.** Decisions lie in $[-1,1]$ and losses are linear, $\ell_t(w)=g_tw$ with $|g_t|\le1$. Take $T=100$.

**Work it through.** Here $D=2$, $G=1$, and the constant step size is $0.2$. Equation {eq}`eq-ogd-bound` gives regret at most 20. The best fixed decision has total loss $-|\sum_tg_t|$, while a separate best decision at every time would have loss $-\sum_t|g_t|$.

**What this shows.** The guarantee concerns the first comparator. Replacing it with the second changes the problem; the original proof does not establish small regret against an unrestricted sequence of decisions.
:::

## 8.4 · Regret is not automatically statistical risk

Under iid observations $Z_t$, let $\ell_t(w)=\ell(w,Z_t)$ and $R(w)=E\ell(w,Z)$. If $w_t$ depends only on preceding observations, it is independent of the current iid $Z_t$. For convex loss, the average iterate $\bar w_T=T^{-1}\sum_tw_t$ then satisfies an online-to-batch inequality of the form

$$
E\{R(\bar w_T)-R(w^*)\}\le
\frac{E\operatorname{Regret}_T}{T},
$$

where $w^*$ minimizes population risk and the expectations exist. Independence and convexity justify this conversion. An arbitrary dependent or drifting sequence does not permit the same argument merely because an online algorithm was used.

Likewise, a fixed-comparator regret bound does not establish calibrated confidence intervals, validity under arbitrary optional stopping, or successful tracking of unrestricted change. Dynamic or adaptive regret changes the comparator class and generally needs additional structure or a different guarantee.

## Zero static regret can coexist with a large tracking error

Consider an even number $T$ of rounds with two experts whose losses alternate between $(0,1)$ and $(1,0)$. A uniform mixture has loss $T/2$, and each single fixed expert also has loss $T/2$. Its static regret is exactly zero.

A comparator allowed to switch freely chooses the zero-loss expert each round and incurs zero loss. Relative to that comparator, the same learner loses $T/2$. Thus a static-regret guarantee does not promise to track every change in the best action. Dynamic comparisons require restrictions on the comparator or environment to obtain useful guarantees.

:::{admonition} Analysis check - complete the protocol
:class: course-method

Specify what is known before acting, what feedback arrives afterward, and which decisions the comparator may make. In the laboratory, distinguish a mixture's expected loss from the realized loss of a sampled expert; they are different random quantities even when their expectations agree.
:::

## 8.5 · Execute the full-information experiment

The lab supplies all experts' losses after each decision. One expert has the smallest expected loss in the first half, and another in the second. It compares uniform mixing, ordinary Hedge, and a policy told the exact change boundary. The last is an oracle illustration, explicitly using information unavailable to a learner that must detect changes.

The table reports loss relative to both a fixed expert and a two-segment comparator. The OGD example separately checks the deterministic inequality on a realized sequence of linear losses. These examples distinguish a full-information calculation from the bandit experiment in the next chapter.

:::{literalinclude} ../../labs/08-online-learning.R
:language: r
:::

:::{include} ../_generated/08-online-learning-output.md
:::

## 8.6 · Exercises

1. Reproduce Example 8.1 with $\eta=0$. Explain why the result does not show that learning is useless.
2. Complete the projection-and-telescoping proof. What breaks if the gradient has no finite bound on the decision set?
3. Run the experiment without changing the experts' loss probabilities halfway through. Report both comparators and explain when they coincide or differ in a finite sample.
4. A researcher sorts a static dataset into an arbitrary order and calls it a time series. What online-learning questions can still be studied, and which claims about temporal distribution shift are unsupported?

:::{admonition} Worked solutions
:class: dropdown

**1.** With zero learning rate, the weights stay uniform and total loss is 1, matching the best fixed expert in this particular two-round sequence. Other sequences reward adaptation. A two-round counterexample to monotone improvement is not a universal comparison of algorithms.

**2.** Bound the first squared distance by $D^2$, discard the nonnegative final squared distance, and bound $\sum_t\|g_t\|^2$ by $TG^2$. Without a gradient bound, that sum can be large or require a different data-dependent analysis.

**3.** Even under stationary expected losses, random outcomes can make the best realized expert differ across halves. The two-segment comparator is at least as flexible and has loss no larger than the best fixed expert. Report this distinction rather than interpreting every difference as evidence of population drift.

**4.** One can study sequential prediction under the chosen presentation protocol, computation, and regret relative to declared comparators. The imposed order does not supply evidence that the original population evolved over real time. A temporal claim requires a defensible timestamp or longitudinal data-generating interpretation.
:::

## Reading and transition

Read Sections 1.1–1.3 and 3.1 of [Hazan, *Introduction to Online Convex Optimization*, second edition](https://arxiv.org/abs/1909.05207). The text also develops alternative comparator notions. Next, remove full feedback: choosing an action determines which reward is observed.
