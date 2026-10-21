# Reading guide

Start with the introductory resources. Read foundational papers after working through the corresponding small example. This is a selected learning path, not an exhaustive survey or a claim about the newest model.

## First-pass resources

| Resource | When to use it | Reading goal |
|:--|:--|:--|
| [Dive into Deep Learning](https://d2l.ai/) | Modules 2–5 | Work through arrays, autodiff, multilayer perceptrons, attention, and transformers; skip familiar mathematical preliminaries as needed. |
| [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Module 2 onward | Learn tensors, datasets, model definitions, gradients, and the training loop. |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Modules 3, 5, 6, and 9 | Connect tokenization and transformer concepts to practical model tooling. |
| [MyST-NB](https://myst-nb.readthedocs.io/en/latest/) and [Sphinx](https://www.sphinx-doc.org/) | Throughout | Write and build executable teaching notes with Python. |

## Foundational papers by module

- **Attention and transformers — Module 4–5:** Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762). First identify the inputs, outputs, attention equation, and block diagram; return to experiments later.
- **Latent-variable generation — Module 7:** Kingma and Welling (2013 preprint), [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). Connect the ELBO to variational inference and study the reparameterization step.
- **Adversarial generation — Module 7:** Goodfellow et al. (2014), [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). Contrast its discrimination objective with the VAE bound.
- **Diffusion — Module 8:** Ho, Jain, and Abbeel (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Track the forward and reverse processes and the denoising objective.
- **Efficient adaptation — Module 9:** Hu et al. (2021 preprint), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Identify which parameters are frozen and which low-rank matrices are learned.
- **Preference learning — Module 9:** Rafailov et al. (2023), [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). Study the preference likelihood and the role of the reference policy.
- **Retrieval — Module 10:** Lewis et al. (2020), [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). Identify the retriever, generator, and conditioning information; distinguish the original model from a simple modern retrieval pipeline.
- **Uncertainty — Module 11:** Angelopoulos and Bates (2021 preprint), [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511). Establish the prediction target, calibration set, and coverage assumptions before applying the ideas to generated outputs.
- **Agents — Module 12:** Yao et al. (2022 preprint; ICLR 2023), [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). Follow the interaction with tools and observations, and inspect how success is measured.

## A statistician's reading note

For each paper, write a short note answering: What is the statistical object? What is optimized? What assumptions are made? What is the comparison baseline? What are the experimental units? Which claim would I want to reproduce?

Links and source identities were checked when the course was expanded on October 20, 2026. Software documentation evolves; record the versions used for each lab.
