# 10 · Retrieval-augmented generation (RAG)

```{admonition} Chapter status: outline
:class: note
This is the planned scope of Chapter 10. The complete lesson, executable examples, and solutions will be developed later. [Chapter 1](01-probability-to-genai.md) is available now.
```

**Question.** How can a model answer using a specified collection of documents?

- **Learn:** document chunking, indexing, retrieval, reranking, context assembly, citation, and abstention when evidence is insufficient.
- **Statistical lens:** prediction conditional on selected evidence, retrieval error, measurement error, and the interaction between pipeline components.
- **Python practice:** build a small pipeline over public statistical notes. Inspect retrieved passages, compare with a keyword baseline, and test answerable and unanswerable questions separately.
- **Self-check:** identify whether an error arose in retrieval, evidence selection, or generation. A citation does not by itself show that its source supports the claim.
- **Reading:** [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
