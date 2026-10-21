# Independent assignments

Each chapter now has three distinct activities: an executed worked example in the lesson, a student notebook with an unfinished function and independent acceptance fixtures, and a transfer experiment on new inputs. The 14 student notebooks intentionally contain `NotImplementedError`; they are not failed book builds or completed solutions.

Read **A.11 before running any downloaded notebook**, and A.12 before Chapter 2 uses compact syntax. Keep the notebook anywhere inside the cloned repository. A standalone download outside that tree needs `GENAI_COURSE_ROOT` pointing to the clone. Restart the kernel and run all cells after implementing the function. The published chapter notebooks remain executed reference material.

| Chapter | Transfer focus | Download |
|:--|:--|:--|
| 1 | Fixed counts, not a lucky simulation | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/01-student.ipynb) |
| 2 | A new design matrix | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/02-student.ipynb) |
| 3 | Identity survives reordering | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/03-student.ipynb) |
| 4 | Unequal attention weights | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/04-student.ipynb) |
| 5 | Detect a deliberately leaky output | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/05-student.ipynb) |
| 6 | A stopping token is not grammaticality | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/06-student.ipynb) |
| 7 | Optional branch: detect mass between modes | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/07-student.ipynb) |
| 8 | Optional branch: an independent score oracle | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/08-student.ipynb) |
| 9 | Optional branch: count the update | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/09-student.ipynb) |
| 10 | A valid citation can support a false statement | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/10-student.ipynb) |
| 11 | Join by question, then subtract | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/11-student.ipynb) |
| 12 | Reject a plausible invented finish | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/12-student.ipynb) |
| 13 | Compute an interaction after matching tasks | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/13-student.ipynb) |
| 14 | Honor every requested setting | [Student notebook](https://chikuang.github.io/teaching/genai/_static/assignments/14-student.ipynb) |

The [acceptance fixtures](_static/assignments/acceptance-checks.py) use hand-computable targets and negative cases, independently of the [worked functions](_static/assignments/solutions.py). They are learning checks, not a hidden test set. A function can pass a few fixtures and still be wrong elsewhere: add your own counterexample. In particular, a constant zero gradient, uniform attention that ignores logits, and a constant causal-test pass are rejected.

## What counts as completion

A complete submission includes the implemented function, executed fixtures and a deliberately wrong control, all rows of the prescribed transfer experiment, and an interpretation naming the estimand and the limit of the evidence. Use the notebook's 3+2+3+2 rubric. Numerical agreement is distinct from support for a scientific claim; an unexpected comparison is acceptable when correctly recorded and interpreted.

For Chapters 10–14 use the [common evaluation record](evaluation-record.md). A missing review stays null. A valid JSON response with an unsupported claim is a failure. A refused unsupported capstone request can be correct behavior, but only if its intent was preserved. Chapters 7–9 are optional: their small independent numerical experiments are sufficient; training a large image or language model is not required.
