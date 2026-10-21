# STAT XXXX Generative AI

```{raw} html
<div class="course-hero">
<p class="eyebrow">Topic Course · Independent study</p>
<h2>From probability to attention.<br>From models to agents.</h2>
<p>A personal learning course by <a href="https://chikuang.github.io/">Chi-Kuang Yeh</a>, beginning with statistics and statistical machine learning, and learning modern generative AI from the ground up.</p>
</div>
```

## Begin with what you know

You understand probability, inference, linear algebra, optimization, and statistical machine learning. This course uses that background to explain generative AI, attention, transformers, and agents. **No prior Python or GenAI knowledge is assumed.** We introduce the programming language alongside the mathematics.

This is a self-paced **Topic Course** for independent study, with material that can later be adapted for teaching. There is no fixed semester or grading. The lessons are in English, using the terminology of the research literature.

```{admonition} Start with the complete first chapter
:class: tip
[Chapter 1: From probability models to generative AI](chapters/01-probability-to-genai.md) develops categorical estimation, log loss, smoothing, sampling, temperature, and a working bigram language model. It includes explained Python code, executed outputs, eight exercises, worked solutions, and a downloadable notebook.
```

## A course in three stages

| Stage | Chapters | What you will understand |
|:--|:--|:--|
| Model the data | 1–4 | Probability models, Python, embeddings, and attention as a context-dependent weighted average. |
| Understand generation | 5–10 | Transformers, language models, latent variables, diffusion, adaptation, and retrieval. |
| Evaluate the system | 11–14 | Uncertainty, agents, reliable experiments, and a small teaching or research project. |

Every chapter starts with a concrete question, develops a statistical formulation, and connects it to a small Python experiment. NumPy makes the mathematics visible; later chapters introduce PyTorch for neural networks.

Each chapter also includes a **pipeline or framework diagram** with the inputs, outputs, and checks at each stage. Read these alongside the code to distinguish fitting from prediction, model generation from evaluation, and an agent's proposals from tool execution. The [RAG chapter](chapters/10-rag.md) follows one question through indexing, retrieval, context selection, prompt construction, response validation, and evaluation.

## Study and experiment

1. [Set up Python and JupyterLab](python-workflow.md). You can also read the rendered lessons without installing anything.
2. Work through [Chapter 1](chapters/01-probability-to-genai.md), running cells from top to bottom.
3. Use the [statistical translation guide](statistical-lens.md) and [selected readings](readings.md) when terminology is unfamiliar.
4. Follow the [14-chapter course guide](syllabus.md), then complete the worked simulation-assistant project.

**Complete core course:** All 14 chapters contain developed lessons, mathematical explanations, executable Python experiments, exercises, worked solutions, and downloadable notebooks. The core chapter examples run on a laptop CPU without API keys or pretrained-model downloads. [Appendix A](appendices/a-python-pytorch.md) supplies a step-by-step Python/PyTorch bridge. [Appendix B](appendices/b-real-llm-systems.md) adds a real local instruction model, RAG, an LLM tool policy, and a natural-language simulation assistant; its optional live run requires a model download, while the book analyzes saved real-model traces. The [short Python warm-up](01-python-lab.md) remains available as a supplement.

The progression includes neural-network training, embeddings, masked attention, a tiny transformer, character generation, a VAE, diffusion, low-rank adaptation, preference learning, retrieval, uncertainty, tool-use policies, reliability experiments, and a worked statistical simulation assistant. Chapters 5–6 use the same tiny decoder for architecture, training, held-out scoring, and generation. Retrieval and agent chapters retain transparent baselines and link to the executed real-model comparisons in Appendix B. All chapters include programming tasks, acceptance checks, and separate worked solutions.

```{toctree}
:caption: Course guide
:maxdepth: 1
:hidden:

syllabus
python-workflow
statistical-lens
readings
```

```{toctree}
:caption: Chapters
:maxdepth: 1
:hidden:

chapters/01-probability-to-genai
chapters/02-python-neural-networks
chapters/03-tokens-embeddings
chapters/04-attention
chapters/05-transformers
chapters/06-language-models
chapters/07-latent-variable-models
chapters/08-diffusion
chapters/09-adaptation-alignment
chapters/10-rag
chapters/11-evaluation-uncertainty
chapters/12-agents
chapters/13-agent-reliability
chapters/14-capstone
```

```{toctree}
:caption: Appendices
:maxdepth: 1
:hidden:

appendices/a-python-pytorch
appendices/b-real-llm-systems
```

```{toctree}
:caption: Supplement
:maxdepth: 1
:hidden:

01-python-lab
```
