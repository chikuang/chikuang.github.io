# The statistical lens

For each new idea, ask: **What are the observations? What is random? What is the target? What is being optimized? What assumptions justify the evaluation?** The connections below are learning aids; their limits matter as much as their similarities.

## A translation guide

| GenAI concept | Statistical starting point | What needs care |
|:--|:--|:--|
| Language modeling | Conditional density estimation and the chain rule | Token likelihood measures text prediction, not the truth of a claim. |
| Cross-entropy training | Maximum likelihood and empirical risk minimization | Token observations within documents are dependent; split and evaluate at an appropriate unit. |
| Embeddings | Learned feature maps and low-dimensional representations | Coordinates are learned representations, not uniquely identified scientific factors. |
| Attention | Query-dependent weighted averages, reminiscent of kernel smoothing | Queries, keys, and values are learned; attention weights are not automatically causal importance or posterior probabilities. |
| Transformer | A composition of learned feature transformations | Attention is one component alongside positional information, nonlinear layers, and residual connections. |
| Temperature | A transformation of a categorical sampling distribution | More or less randomness does not by itself improve calibration or factual correctness. |
| Variational autoencoder | Latent-variable modeling and variational inference | The encoder approximates a posterior under the fitted model; approximation and model misspecification both matter. |
| Diffusion | Denoising, score estimation, and stochastic processes | A learned reverse process approximates a distribution; it does not recover a unique original observation from noise. |
| LoRA | A low-rank restriction on parameter updates | This is a constraint on an update matrix, not necessarily on the entire fitted model. |
| Preference optimization | Paired-comparison models and regularized decision making | Annotator preferences and reward proxies are not objective truth. |
| RAG | Retrieval followed by prediction conditional on selected evidence | Retrieved documents are selected, potentially incomplete or wrong, and usually not random samples from a posterior. |
| Agent | Sequential decisions with actions, observations, and a stopping rule | The language model is only one part of the system; tools, environment, and accumulated errors also matter. |
| LLM evaluation | Measurement, experimental design, and uncertainty quantification | Repeated answers to one prompt do not create independent test tasks. |

## Three equations to keep in view

### A language model is a distribution over sequences

For tokens $x_1,\ldots,x_T$,

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

Training commonly minimizes the sum of negative log conditional probabilities. Generation then samples from a conditional distribution, often modified by a decoding rule. Training and decoding answer different questions.

### Attention forms a data-dependent weighted average

For one attention head,

$$
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),
\qquad Z=AV,
$$

where softmax is applied row by row and $M$ can mask unavailable positions. With $Q\in\mathbb R^{n_q\times d_k}$, $K\in\mathbb R^{n_k\times d_k}$, and $V\in\mathbb R^{n_k\times d_v}$, the output is $Z\in\mathbb R^{n_q\times d_v}$. Each row of $A$ sums to one over allowed positions. This invites a comparison to a normalized smoother, while the learned representations and end-to-end training make it different from a fixed kernel estimator. See [Vaswani et al.](https://arxiv.org/abs/1706.03762).

### Evaluate a system on a specified task distribution

$$
R(\pi)=\mathbb E_{\tau\sim\mathcal D}\!\left[L(\pi,\tau)\right].
$$

Here $\pi$ is the full system, $\tau$ a task, and $L$ a loss that may include failure, cost, and latency. Specify the task population and sampling design before reporting a performance estimate. For stochastic systems, distinguish between-task variation from variation across repeated runs of the same task.

## Questions for every paper or experiment

1. What is the estimand or target behavior, and what is actually measured?
2. What are the training, validation, and test units? Where could leakage enter?
3. Which quantities are learned, fixed, sampled, or selected adaptively?
4. What simple baseline would tell us whether the additional complexity helps?
5. How would distribution shift, dependence, or misspecification change the conclusion?
6. Which claims are empirically supported, and which remain useful analogies?
