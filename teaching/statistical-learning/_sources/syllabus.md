# Course guide and study route

**Topics of Statistical Learning** studies the statistical structure behind learning with incomplete information and inference after a data-dependent choice. The course combines mathematical derivations, small computational investigations, critical reading, and a focused research project.

## Course information

| Item | Proposed arrangement |
|---|---|
| Instructor | Chi-Kuang Yeh · [cyeh@gsu.edu](mailto:cyeh@gsu.edu) |
| Course number | STAT 8XXX — number to be assigned |
| Level and credits | Advanced master's / doctoral elective; 3 proposed credit hours |
| Teaching format | Lecture, seminar discussion, and guided computational investigations |
| Duration | 15 teaching weeks; also suitable for independent study |
| Preparation | Graduate probability and inference, linear algebra, basic optimisation, statistical learning, and programming experience |
| Independent preparation | Approximately 6–9 hours per week during a taught offering |
| Offering information | Term, CRN, room, meeting times, and office hours to be announced |

The public book is usable immediately. Course registration, dated deadlines, and institutional policies are supplied separately when an offering is confirmed.

## Learning outcomes

1. Specify the observed variables, latent variables, population target, and observation mechanism for a learning problem.
2. Derive PU and UU risk representations and explain the assumptions behind identification and stable estimation.
3. Distinguish noisy, complementary, partial, bag-level, and multiple-source labels, and recognise how their mechanisms differ.
4. Analyse identification, separation, comparison design, and dependence in preference models.
5. Formulate full-information and bandit problems, define an appropriate comparator, and derive a basic regret guarantee.
6. Distinguish ordinary, selective, and simultaneous inferential claims, including claims made after clustering.
7. Conduct and communicate a reproducible methodological investigation with a baseline, a sensitivity analysis, and uncertainty assessment.

## A 15-week teaching route

The chapters are learning units. The schedule allocates extra time to the assumptions and computations that benefit from discussion.

| Wk | Question and material | Reading focus | Suggested milestone |
|---|---|---|---|
| 1 | [Targets, risks, and observation mechanisms](chapters/01-statistical-targets.md) | The course's model–target–mechanism framework | Computational readiness check |
| 2 | [Forms of weak supervision](chapters/02-weak-supervision.md) | Natarajan et al.; optional Ishida and Ratner | Mechanism comparison |
| 3 | [Positive–unlabelled learning](chapters/03-pu-learning.md) | Elkan–Noto; PU risk derivation | Start Assignment 1 |
| 4 | [PU estimation and sensitivity](chapters/04-pu-practice.md) | Kiryo et al.; selected Bekker sections | Prior and selection experiment |
| 5 | [Two unlabelled mixtures](chapters/05-uu-learning.md) | Lu et al.; inversion and conditioning | Assignment 1 |
| 6 | [Pairwise preferences](chapters/06-pairwise-preferences.md) | Turner–Firth; identification and estimation | Project proposal |
| 7 | [Context and comparison design](chapters/07-contextual-preferences.md) | Preference model extensions; optional DPO | Start Assignment 2 |
| 8 | [Online learning and regret](chapters/08-online-learning.md) | Selected Hazan sections | Work through the OGD bound |
| 9 | [Bandit feedback](chapters/09-bandits.md) | Selected Lattimore–Szepesvári chapters | Evaluate exploration |
| 10 | [Duelling bandits](chapters/10-duelling-bandits.md) | Yue et al.; winner criteria | Assignment 2 |
| 11 | [Selection and conditional inference](chapters/11-selective-inference.md) | Lee et al.; truncated reference distributions | Reproduction checkpoint |
| 12 | [Inference after hierarchical clustering](chapters/12-hierarchical-clustering.md) | Gao–Bien–Witten | Start Assignment 3 |
| 13 | [Inference after k-means](chapters/13-kmeans-inference.md) | Chen–Witten; optional data thinning | Audit algorithm choices |
| 14 | [Research synthesis](chapters/14-research-synthesis.md) | Project-specific primary papers | Assignment 3; project clinic |
| 15 | Project presentations and discussion | Revisit the target and assumptions | Presentation; final report in the designated final period |

## Independent-study route

Work through one chapter at a time. For each chapter, write down the model and target before reading the derivation. Run the supplied experiment unchanged, reproduce its main result, then alter one parameter or assumption. Answer at least one derivation question and one experimental question before revealing a solution. Keep a short record of what the experiment can and cannot establish.

For a narrower reading group, Chapters 1–5 form a weak-supervision block; Chapters 6–10 form a preference and sequential-learning block; Chapters 11–13 form an inference-after-selection block. Chapter 1 supplies notation for all routes, and Chapter 14 provides a shared project framework.

## Proposed assessment for a taught offering

| Component | Weight | Timing |
|---|---:|---|
| Three integrated assignments | 45% total; 15% each | Weeks 5, 10, 14 |
| Paper discussion and a two-page reading note | 10% | Assigned during the term |
| Project proposal | 5% | Week 6 |
| Reproduction checkpoint | 10% | Week 11 |
| Final project report | 20% | Designated final-assessment period |
| Project presentation | 10% | Week 15 |

There is no proposed written midterm or final examination. A well-designed reproduction with one consequential extension is sufficient for the project. A new theorem or publication is not required. Details and rubrics are on the [assignment](assignments.md) and [project](project.md) pages.

## Working principles

**Reproducibility.** Record random seeds, software versions, preprocessing, data provenance, and an execution command. Separate training, method selection, and final evaluation. Report Monte Carlo uncertainty for simulated performance.

**Collaboration.** Discuss ideas and compare understanding. Identify collaborators and external sources, and write your own derivations, code, and interpretation. Any offering-specific collaboration policy takes precedence for assessed work.

**AI tools.** They may assist with coding, editing, or checking a line of reasoning when permitted by the offering. Record substantive use and verify the result. An apparently plausible derivation, citation, or executable program still needs independent checking. Do not put restricted or confidential data into an external service.

**Accessibility and policies.** Students should use the official accommodations and support channels at their institution. The final offering syllabus will include the applicable attendance, late-work, academic-integrity, accessibility, and support policies. This public study guide does not substitute for that document.
