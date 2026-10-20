# Learn, teach, and build with Python

The whole website is built by a **Python toolchain: Sphinx + MyST-NB + sphinx-book-theme**. Python executes the lesson cells, makes the figures, and builds the static HTML pages. You can publish that HTML on GitHub Pages. The browser uses the theme's bundled CSS and JavaScript for navigation and search; authoring and building require neither Node.js nor Quarto.

## 1 · Choose how to study

- **Read first:** all examples and their outputs are visible on the website.
- **Experiment:** download the <a href="notebooks/01-probability-to-genai.ipynb" download>Chapter 1 notebook</a>, open it in JupyterLab, and change the code.
- **Write lessons:** edit the Markdown files under `docs/` and rebuild the website.

A *terminal* accepts commands that manage files and run programs. A *notebook* contains text and Python code cells. Enter the commands below in a terminal; enter the Python examples from the lessons in notebook code cells. The shell prompt is not part of a command.

## 2 · Create a Python environment

Use **Python 3.12** for the tested course environment. A virtual environment keeps this course's packages separate from other projects.

Download or clone [Teaching-GenAI on GitHub](https://github.com/chikuang/Teaching-GenAI), then open a terminal in that folder. If Git is installed:

```bash
git clone https://github.com/chikuang/Teaching-GenAI.git
cd Teaching-GenAI
```

On macOS or Linux, with Python 3.12 installed as `python3.12`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Here `-m` runs a Python module. `venv` creates the environment; activation makes `python` refer to that environment; `pip install` installs the packages listed in the file. Run `python --version` to check the interpreter. If PowerShell blocks activation, use `.venv\Scripts\python.exe` wherever the instructions say `python`; activation is a convenience, not a requirement.

`requirements.txt` gives compatible package ranges. `requirements-lock.txt` records the tested macOS/Python 3.12 environment; a platform-specific package in that snapshot may need adjustment on another operating system. The first chapter runs on a CPU with no API key or model download.

## 3 · Open a notebook

Save the downloaded notebook in the course folder, then run:

```bash
python -m jupyterlab
```

JupyterLab opens in your browser. Open the `.ipynb` file using its file browser. Select a code cell and press **Shift+Enter**. Run from the top: later cells refer to earlier objects. A number in brackets indicates a completed execution; an asterisk indicates a running cell.

To check reproducibility, choose **Kernel → Restart Kernel and Run All Cells**. The downloaded notebook already contains the website's executed outputs, but you can edit the cells and run them again. Saving a notebook preserves its code and outputs; it does not preserve every object in the running kernel's memory.

## 4 · Build the website

From the repository root, with the environment active:

```bash
python scripts/build.py
```

This command exports the lesson notebooks, executes their Python cells in fresh kernels, and asks Sphinx to create the HTML website. It stops on execution errors or Sphinx warnings. Successful output is written to `_build/html/`, including executed notebooks under `notebooks/`.

To preview the result:

```bash
python -m http.server 8767 --directory _build/html
```

Open `http://localhost:8767` in a browser. Keep the terminal running while you browse. Press **Control+C** to stop the server. After editing a lesson, run the build command again and refresh the browser. This is an explicit rebuild, not automatic live reload.

## 5 · Understand the source files

| Path | Purpose |
|:--|:--|
| `docs/index.md` | Homepage and chapter navigation. |
| `docs/chapters/` | Canonical lesson text and Python code cells. |
| `docs/conf.py` | Python configuration for Sphinx and notebook execution. |
| `docs/_static/course.css` | White backgrounds, readable typography, and yellow/pink accents. |
| `scripts/build.py` | Python entry point for execution, rendering, and notebook export. |
| `requirements.txt` | Python package dependencies. |
| `_build/html/` | Generated website; rebuild it rather than editing it. |

The lessons use **MyST Markdown**, which combines ordinary Markdown, mathematical notation, and executable notebook cells. A minimal executable lesson looks like this:

````markdown
---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# A small probability example

For counts $n_k$, estimate $p_k$ by $n_k / N$.

```{code-cell} python
import numpy as np
counts = np.array([6, 3, 1])
probabilities = counts / counts.sum()
print(probabilities)
```
````

The first block identifies a notebook and its kernel. The Markdown becomes prose; the `code-cell` block is executed by Python during the build. Add a new chapter to the `toctree` in `docs/index.md` so that it appears in the navigation. The build script discovers executable MyST chapters automatically and exports their notebooks.

Use [MyST-NB's authoring documentation](https://myst-nb.readthedocs.io/en/latest/authoring/text-notebooks.html) for notebook syntax and [Sphinx's documentation](https://www.sphinx-doc.org/en/master/) for site configuration. Chapter 1's notebook uses standard Markdown and LaTeX so its prose can also be read directly in JupyterLab.

## 6 · Publish a reviewed build

The public course lives at [chikuang.github.io/teaching/genai/](https://chikuang.github.io/teaching/genai/). Its source is in `chikuang/Teaching-GenAI`; rendered pages live in the separate `chikuang/chikuang.github.io` repository under `teaching/genai/`.

After building and previewing, copy the **contents** of `_build/html/` into that course directory, then commit and push the reviewed source and rendered pages to their respective repositories. Keep the course directory separate from the rest of the personal website. GitHub Pages serves the published HTML.

The course's GitHub Actions workflow independently executes and builds the lessons and saves an HTML artifact. It validates the source; publication to the personal website remains a separate step.
