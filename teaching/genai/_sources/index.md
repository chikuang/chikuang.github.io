# Generative AI through a statistical lens

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

## Study and experiment

1. [Set up Python and JupyterLab](python-workflow.md). You can also read the rendered lessons without installing anything.
2. Work through [Chapter 1](chapters/01-probability-to-genai.md), running cells from top to bottom.
3. Use the [statistical translation guide](statistical-lens.md) and [selected readings](readings.md) when terminology is unfamiliar.
4. Follow the [14-chapter course outline](syllabus.md) as the remaining lessons develop.

**Current materials:** Chapter 1 is a complete lesson. Chapters 2–14 have individual pages with questions, learning objectives, statistical connections, and planned Python practice; they are explicitly marked as outlines. The [short Python warm-up](01-python-lab.md) remains available as a supplement.

```{raw} html
<p class="math-heritage">Black and yellow recall Waterloo; pink celebrates my Faculty of Mathematics background. The palette follows <a href="https://uwaterloo.ca/brand/how-express-our-brand/colour">Waterloo's published colour guide</a>. This is a personal course website.</p>
```

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
:caption: Supplement
:maxdepth: 1
:hidden:

01-python-lab
```
