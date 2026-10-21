# Topics of Statistical Learning

:::{raw} html
<section class="course-hero" aria-label="Course introduction">
<div class="eyebrow">Topic Course · Chi-Kuang Yeh · R</div>
<h2>Learn from imperfect information.<br>Know what the evidence supports.</h2>
<p>Incomplete labels, comparisons, sequential feedback, and discoveries made from the data itself. A graduate course on the assumptions that make statistical learning and inference possible.</p>
<div class="course-actions"><a class="course-start" href="chapters/01-statistical-targets.html">Start with Chapter 1 <span aria-hidden="true">→</span></a><a href="syllabus.html">Explore the course guide</a></div>
</section>
:::

This course develops **preference learning, online learning, positive–unlabelled (PU) and unlabelled–unlabelled (UU) learning, other forms of weak supervision, and selective inference after clustering**. The unifying habit is to specify the observation process, the statistical target, the assumptions that identify it, and the guarantee a method actually provides.

Each of the 14 chapters contains explanations, mathematical arguments, worked examples, a downloadable R experiment, exercises, and separate worked solutions. Read the example first, predict what the experiment will show, and then change an assumption. The [interactive demonstrations](demos.md) offer a quick way to see three of these mechanisms in action.

**Instructor:** [Chi-Kuang Yeh](https://chikuang.github.io/) · [cyeh@gsu.edu](mailto:cyeh@gsu.edu)  
**Level:** advanced master's / doctoral · **Format:** independent study or a proposed 15-week graduate elective  
**Computing:** R on an ordinary laptop; the required chapter experiments use base R.

## Four connected questions

| Learning route | Central question | Chapters |
|---|---|---|
| Incomplete and weak supervision | What can be learned when the labels are incomplete, noisy, or only available in aggregate? | [1–5](chapters/01-statistical-targets.md) |
| Preference learning | What does a comparison identify, and which comparisons should we collect? | [6–7](chapters/06-pairwise-preferences.md) |
| Online learning and bandits | How should a learner act when data arrive sequentially and feedback may be partial? | [8–10](chapters/08-online-learning.md) |
| Inference after selection | How can we make a valid statement about a structure selected using the same data? | [11–14](chapters/11-selective-inference.md) |

The last route studies inference after a data-dependent choice. It connects to the learning routes through the observation and selection mechanisms; it is a distinct inferential problem from learning with weak labels.

## Choose a starting point

- **Read in order:** begin with [statistical targets](chapters/01-statistical-targets.md), then follow the chapter links. The [course guide](syllabus.md) maps the book to 15 teaching weeks.
- **Focus on a research problem:** use the [reading guide](readings.md) to find the relevant model, theorem, and implementation. Revisit the first chapter whenever the target becomes unclear.
- **Learn through computation:** follow the [R workflow](r-workflow.md), run a chapter experiment, then attempt the exercises before opening the solutions.
- **Build a research project:** use the [assignments](assignments.md) for focused practice and the [project guide](project.md) for a reproduction plus a consequential extension.

## What you should be able to do

By the end, you should be able to derive PU and UU risk identities; distinguish several weak-label mechanisms; fit and critique a pairwise preference model; formulate a sequential decision problem with an explicit comparator; derive a basic regret bound; explain conditional validity after selection; and design a reproducible experiment that probes a method's assumptions.

You will need graduate probability, mathematical statistics, linear algebra, basic optimisation, and some R experience. [The workflow page](r-workflow.md) includes a short readiness check and a minimal reproducible-report template.

:::{note}
This is a developed topic-course resource. The accompanying graduate-course proposal uses **STAT 8XXX**, with the number and offering to be assigned. The 15-week route and assessment scheme are proposed teaching arrangements; meeting dates, registration details, and institution-specific policies will be set for an actual offering.
:::

## Course materials

[Download the complete course book](https://chikuang.github.io/teaching/statistical-learning/Topics-of-Statistical-Learning.pdf) · [Download all R labs](_static/downloads/statistical-learning-labs.zip) · [Course proposal](_static/downloads/Topics_of_Statistical_Learning_Proposal.pdf) · [Source and build instructions](https://github.com/chikuang/Teaching-SML)

The reading layout follows [Generative AI](https://chikuang.github.io/teaching/genai/): use the sidebar to navigate, the page contents to find a section, **Code** to open a program, and the theme control for light or dark reading. Outputs stay visible; exercise solutions open separately.

:::{toctree}
:hidden:
:caption: Course guide

syllabus
r-workflow
assignments
project
demos
readings
:::

:::{toctree}
:hidden:
:caption: I · Weak supervision

chapters/01-statistical-targets
chapters/02-weak-supervision
chapters/03-pu-learning
chapters/04-pu-practice
chapters/05-uu-learning
:::

:::{toctree}
:hidden:
:caption: II · Preference learning

chapters/06-pairwise-preferences
chapters/07-contextual-preferences
:::

:::{toctree}
:hidden:
:caption: III · Online learning

chapters/08-online-learning
chapters/09-bandits
chapters/10-duelling-bandits
:::

:::{toctree}
:hidden:
:caption: IV · Inference after selection

chapters/11-selective-inference
chapters/12-hierarchical-clustering
chapters/13-kmeans-inference
chapters/14-research-synthesis
:::
