
# Course guide

**STAT XXXX Generative AI**

This sequence assumes a strong statistics background and begins at an introductory level in deep learning and GenAI. Modules are learning units, not calendar weeks. Work through them at a comfortable pace, returning to the mathematics when it clarifies an experiment. All 14 chapters now include developed lessons, executable **Python** labs, exercises, and worked solutions. Core experiments progress from NumPy to PyTorch and then transparent retrieval and tool-use policies. They run on a CPU without model downloads or API credentials. Appendix A supplies the Python/PyTorch prerequisites. Appendix B contains an optional executable local-LLM track with published real-model traces; model downloads are separate from the offline core and normal book builds.

## Two complementary study routes

For a first pass focused on attention and agents, read Chapters 1–6, then 10–14. Read Appendix A alongside the first six chapters and begin live notebook 1 after Chapter 6, then return to the later live notebooks after their matching chapters. Appendix B retains the comparative experiments. Chapters 7–9 supply the wider generative-model and adaptation perspective; return to them for variational inference, score estimation, and preference learning. They remain part of the full course, but need not delay the first functioning LLM system.

Each chapter now separates questions from revealable answers and includes a coding/experiment task with acceptance criteria. Start with the original lab, make one controlled change, and write a short explanation of the target, result, and limitation. The **Show all code** button provides a persistent study preference; outputs stay visible and answers remain separately controlled.

## Road map

| Module | Topic | Statistical connection | Evidence of learning |
|:--|:--|:--|:--|
| 1 | [Generative AI as probabilistic modeling](chapters/01-probability-to-genai.md) | Density estimation, likelihood, risk | Derive and sample a tiny language model |
| 2 | [Python and neural network foundations](chapters/02-python-neural-networks.md) | GLMs, optimization, regularization | Fit a small neural predictor |
| 3 | [Tokens and embeddings](chapters/03-tokens-embeddings.md) | Feature maps, dimension reduction | Compare two text representations |
| 4 | [Attention from first principles](chapters/04-attention.md) | Adaptive weighted averages | Implement and inspect attention |
| 5 | [The transformer](chapters/05-transformers.md) | Compositional representations | Trace a decoder block |
| 6 | [Training and using language models](chapters/06-language-models.md) | Conditional likelihood, sampling | Compare decoding rules |
| 7 | [Latent-variable generative models](chapters/07-latent-variable-models.md) | Variational inference, change of variables | Derive an ELBO and simulate |
| 8 | [Diffusion and score-based modeling](chapters/08-diffusion.md) | Denoising, stochastic processes | Run a toy noise/recovery experiment |
| 9 | [Adaptation and alignment](chapters/09-adaptation-alignment.md) | Low-rank constraints, paired comparisons | Compare adaptation strategies |
| 10 | [Retrieval-augmented generation](chapters/10-rag.md) | Conditional prediction, selection | Build and inspect a small RAG pipeline |
| 11 | [Evaluation and uncertainty](chapters/11-evaluation-uncertainty.md) | Experimental design, calibration | Produce an evaluation report |
| 12 | [Agents and tool use](chapters/12-agents.md) | Sequential decisions, stopping | Build a bounded single agent |
| 13 | [Reliable agent experiments](chapters/13-agent-reliability.md) | Dependence, ablations, reliability | Compare agent and workflow baselines |
| 14 | [A statistician's GenAI project](chapters/14-capstone.md) | Reproducible empirical research | Deliver a small teaching or research system |

## Part I · From statistical learning to attention


## A route with concrete stopping points

| Read | Python preparation | Demonstrate before continuing |
|:--|:--|:--|
| Chapter 1 | A.1–A.4 alongside the lesson | Explain one probability estimate and one generated draw. |
| Chapter 2 | A.5–A.7 and A.12 first | Trace one forward/loss/backward/update cycle. |
| Chapters 3–4 | A.1–A.4, especially axes | Tokenize a sentence and verify a masked weighted average. |
| Chapters 5–6 | A.8 and A.12 | Trace tensor dimensions and distinguish fitting from decoding. |
| Live lab 1 | A.11–A.12, then C.1 | Run one real model call and inspect its tokens. |
| Chapter 10 + live lab 2 | A.9 and A.11 | Replace the generator while preserving the RAG pipeline. |
| Chapter 11 | Question-level paired outcomes | Separate contract checks, semantic review and task success. |
| Chapter 12 + live lab 3 | The tool-contract migration table | Explain an error observation and the resulting next action. |
| Chapter 13 | The complete retry/cache table | Calculate a paired factorial contrast. |
| Chapter 14 + live lab 4 | Request, tool and report contracts | Complete and verify the natural-language simulation workflow. |

Chapters 7–9 are a parallel branch: latent variables and ELBO, diffusion as regression, and low-rank/preference adaptation. Return to them after Chapter 6 or after the system-building route. Their small numerical experiments explain mechanisms; they do not claim to train a production image generator or fine-tune a large instruction model.

[Appendix C](appendices/c-guided-live-labs.md) provides the four self-contained live notebooks. Each starts from a fresh kernel, loads the pinned local model, exposes the relevant intermediate objects, and saves actual calls. Appendix B retains the earlier comparative experiments and failures for analysis.

(module-01)=
### 1 · Generative AI as probabilistic modeling

**Question.** What does a generative model learn, and what does it mean to generate a new observation?

- **Learn:** distinguish joint, conditional, and discriminative models; use the chain rule to factor a text distribution; connect cross-entropy to negative log likelihood.
- **Statistical lens:** density estimation, empirical risk, smoothing, and held-out log loss.
- **Python practice:** estimate a tiny categorical or bigram model, sample sequences, and compare training and held-out loss. [Chapter 1](chapters/01-probability-to-genai.md) develops the full example, including a bigram language model.
- **Self-check:** explain why fluent sampled text need not be factually correct, and why zero-probability events cause problems for log loss.

(module-02)=
### 2 · Python and neural network foundations

**Question.** What changes when a regression model learns its own nonlinear features?

- **Learn:** arrays, shapes, indexing, broadcasting, functions, and reproducible random generators; then tensors, automatic differentiation, minibatches, and a PyTorch training loop.
- **Statistical lens:** linear/logistic regression, basis expansions, regularization, optimization, overfitting, and validation.
- **Python practice:** fit logistic regression and a small multilayer perceptron to the same simulated problem; record train/validation curves and compare against a simple baseline.
- **Self-check:** distinguish the model, loss, optimizer, and data split; check a derivative numerically before trusting autodiff.

(module-03)=
### 3 · Tokens, embeddings, and representations

**Question.** How does text become something a numerical model can use?

- **Learn:** tokenization, vocabulary, one-hot representations, embedding lookup, positional information, cosine similarity, and contextual representations.
- **Statistical lens:** design matrices, learned feature maps, matrix factorization, and identifiability.
- **Python practice:** tokenize a small course-authored corpus, learn character merges, and compare bag-of-words with SVD embedding similarity. Inspect examples where similarity is misleading.
- **Self-check:** explain why a token is not necessarily a word and why a two-dimensional embedding plot is not evidence of causal structure.

(module-04)=
### 4 · Attention from first principles

**Question.** How can a representation select and combine information relevant to a query?

- **Learn:** queries, keys, values, scaled dot products, row-wise softmax, self-attention, cross-attention, causal masks, and multiple heads.
- **Statistical lens:** a query-dependent normalized weighted average; compare with kernel smoothing without claiming exact equivalence.
- **Python practice:** calculate a three-token attention example by hand, implement it in NumPy, verify dimensions and row sums, and plot how weights change when keys change.
- **Self-check:** explain the role of the mask and the $\sqrt{d_k}$ scaling; describe why attention weights alone are not a faithful explanation of a prediction.
- **Reading:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762), especially the architecture and attention equations.

## Part II · Transformers and generative models

(module-05)=
### 5 · Assembling the transformer

**Question.** How does one attention calculation become a sequence model?

- **Learn:** residual connections, normalization, feed-forward layers, positional encodings, encoder/decoder distinctions, and a decoder-only forward pass.
- **Statistical lens:** a composition of fitted feature transformations with shared parameters and a conditional prediction head.
- **Python practice:** implement a tiny decoder block in PyTorch; track every tensor's dimensions and use a causal-mask check to show future tokens cannot influence earlier outputs.
- **Self-check:** draw the route from token IDs to next-token probabilities, and distinguish the original encoder-decoder transformer from a decoder-only language model.

(module-06)=
### 6 · Language model training, prompting, and decoding

**Question.** What is learned in the weights, and what changes when we provide a prompt?

- **Learn:** next-token prediction, teacher forcing, context windows, pretraining versus inference, in-context examples, greedy decoding, temperature, top-k, and top-p.
- **Statistical lens:** conditional likelihood, distribution shift, and transformations of a sampling distribution. In-context behavior can be studied statistically without assuming the model performs exact Bayesian inference.
- **Python practice:** train the same tiny transformer from Chapter 5 on a synthetic sentence grammar, hold out complete sentence combinations, and compare decoding choices on fixed prompts.
- **Self-check:** explain why perplexity is not a universal measure of usefulness and why prompting is not parameter estimation by gradient descent at inference time.

(module-07)=
### 7 · Latent variables, VAEs, GANs, and flows

**Question.** What alternatives to sequential factorization can represent a complex distribution?

- **Learn:** the encoder/decoder view of a VAE, the evidence lower bound (ELBO), reparameterization, and a conceptual comparison with adversarial learning and invertible flows.
- **Statistical lens:** latent-variable models, variational inference, density-ratio ideas, and change of variables. Compare objectives, sampling, and access to likelihoods.
- **Python practice:** derive the ELBO and implement a small VAE on synthetic two-dimensional data. Treat GANs and flows as comparison topics on the first pass.
- **Self-check:** separate the latent prior, approximate posterior, and observation model; explain why matching a few summary statistics cannot establish distributional equivalence.
- **Reading:** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

(module-08)=
### 8 · Diffusion and score-based generation

**Question.** How can learning to remove noise support generation from noise?

- **Learn:** a forward noising process, learned reverse transitions, denoising objectives, score functions, and conditional generation; preview image and multimodal applications.
- **Statistical lens:** stochastic processes, denoising estimation, and the score $\nabla_x\log p(x)$, which differs from a likelihood score with respect to model parameters.
- **Python practice:** simulate progressive Gaussian noising on a mixture distribution, then study a small learned denoiser and the effect of the noise schedule.
- **Self-check:** distinguish the known forward process from the fitted reverse process and separate training from sampling.
- **Reading:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Continuous-time formulations are an optional extension.

(module-09)=
### 9 · Fine-tuning, LoRA, and preference learning

**Question.** How do we adapt an existing model, and whose preferences are represented?

- **Learn:** supervised fine-tuning, parameter-efficient updates, LoRA, instruction data, preference data, the RLHF pipeline, and the basic DPO idea.
- **Statistical lens:** transfer learning, regularization, low-rank parameter updates, and Bradley–Terry-style paired comparisons. Keep model quality separate from preference-label quality.
- **Python practice:** fit a toy paired-comparison model and compare a full matrix update with a low-rank update. Small-model fine-tuning is optional after the conceptual exercise.
- **Self-check:** explain what changes under prompting, RAG, and fine-tuning; identify label noise and selection bias in preference data.
- **Readings:** [LoRA](https://arxiv.org/abs/2106.09685) and [DPO](https://arxiv.org/abs/2305.18290).

## Part III · Reliable systems and agents

(module-10)=
### 10 · Retrieval-augmented generation (RAG)

**Question.** How can a model answer using a specified collection of documents?

- **Learn:** document chunking, indexing, retrieval, reranking, context assembly, citation, and abstention when evidence is insufficient.
- **Statistical lens:** prediction conditional on selected evidence, retrieval error, measurement error, and the interaction between pipeline components.
- **Python practice:** build a small pipeline over public statistical notes. Inspect retrieved passages, compare with a keyword baseline, and test answerable and unanswerable questions separately.
- **Self-check:** identify whether an error arose in retrieval, evidence selection, or generation. A citation does not by itself show that its source supports the claim.
- **Reading:** [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).

(module-11)=
### 11 · Evaluation, calibration, and uncertainty

**Question.** What evidence would justify saying one system is better than another?

- **Learn:** task-specific metrics, held-out evaluation, data contamination, human annotation, LLM-as-a-judge limitations, calibration, abstention, and an introduction to conformal methods.
- **Statistical lens:** estimands, sampling units, paired comparisons, bootstrap uncertainty, clustered dependence, and distribution shift. Conformal coverage needs an explicit target and assumptions such as exchangeability; it is not an automatic guarantee of free-text truthfulness.
- **Python practice:** compare two fixed systems on the same held-out questions; bootstrap at the question level, inspect judge disagreement, and draw a risk-versus-coverage curve for abstention.
- **Self-check:** distinguish token probabilities, answer confidence, and factual correctness; state the population to which a reported interval applies.
- **Reading:** [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).

(module-12)=
### 12 · Agents: tools, state, and decisions

**Question.** What makes a system an agent, beyond producing a text response?

- **Learn:** a model–action–observation loop, tool schemas, state, memory, structured outputs, retries, and stopping rules; distinguish a fixed workflow from adaptive action selection.
- **Statistical lens:** sequential decision making, partial observation, expected utility, and the value of information. Learn the basic state/action/reward vocabulary without requiring a reinforcement learning course first.
- **Python practice:** build a single agent with two narrow tools, such as a calculator and document lookup. Validate arguments, log actions and observations, and enforce a small step budget.
- **Self-check:** explain when a fixed Python function or workflow is sufficient and why tool access does not imply reliable tool use.
- **Reading:** [ReAct](https://arxiv.org/abs/2210.03629); study observable actions and results without treating generated reasoning text as a faithful record of internal computation.

(module-13)=
### 13 · Agent reliability and experimental design

**Question.** Does an agent improve outcomes enough to justify its additional complexity?

- **Learn:** trajectory-level evaluation, failure taxonomies, repeated trials, cost/latency tradeoffs, memory ablations, prompt injection through retrieved text, and human approval for consequential actions.
- **Statistical lens:** dependence within a task trajectory, factorial comparisons, uncertainty in success rates, and adaptive selection bias.
- **Python practice:** compare a cached-answer baseline, a one-pass tool workflow, and a bounded retry policy on paired tasks; extend the protocol to an LLM policy only after establishing these offline baselines. Inject harmless tool failures; log recovery, loop counts, and stopping behavior. Treat external documents as data rather than tool instructions.
- **Self-check:** separate task failure, tool failure, unsupported answers, and budget exhaustion. Add multiple agents only as an optional ablation after establishing a single-agent baseline.

(module-14)=
### 14 · Capstone: a GenAI system for a statistician

Chapter 14 provides a complete simulation-assistant project with a validated request schema, coverage simulation, Monte Carlo intervals, a nominal baseline, and held-out scenarios. For an extension, choose a citation-grounded assistant for public statistical notes, a teaching assistant that explains simulation results, or a tool-using simulation planner with a constrained action space.

Produce a **reproducible Python notebook and a short MyST chapter** containing:

1. A target user, task population, loss, and explicit scope.
2. A simple baseline and one justified extension, such as RAG or tool use.
3. Separate development and held-out evaluation sets, with provenance.
4. Repeated-run results, task-level uncertainty, cost, and failure examples.
5. An ablation that tests what the added component contributes.
6. A plain-language explanation of what the evidence supports and where the system fails.

Success means being able to **explain, implement, evaluate, and teach** the system. A small, well-understood experiment is a complete capstone.

## Optional directions after the core

Multimodal models, advanced diffusion/SDE theory, scaling laws, interpretability, reasoning-model training, causal questions about evaluation, and multi-agent systems can each become a later topic. Select them to answer a concrete research question rather than trying to learn every framework at once.

Use the [independent assignments](assignments.md) after each worked example. The [common evaluation record](evaluation-record.md) is reused in Chapters 10–14.
