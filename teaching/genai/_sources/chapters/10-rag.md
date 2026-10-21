---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# 10 · Retrieval-augmented generation (RAG)

**Question.** How can a model answer using a specified collection of documents?

[Download the executed notebook](https://chikuang.github.io/teaching/genai/notebooks/10-rag.ipynb)

A retrieval pipeline selects evidence before an answer is produced. Errors can arise because the evidence was missing, the retriever selected the wrong passage, or the answer misrepresented a useful passage. We will make those stages observable using a small collection of course-authored statistical notes. The runnable baseline returns a quoted passage with its identifier; it deliberately avoids pretending that a deterministic extractor is a language model.

**Prerequisites:** Chapters 3 and 6. **Outcomes:** construct an index; score queries; retain provenance; evaluate retrieval separately from answer support; implement an explicit abstention rule. An optional generative extension is specified after the fully executable offline baseline.

## 10.1 · The conditional-distribution view

Let $q$ denote a question and $z$ a retrieved passage. A probabilistic retrieval model can be written as

$$
p(y\mid q)=\sum_z p_\eta(z\mid q)p_\theta(y\mid q,z).
$$

A practical top-k pipeline often selects a few passages and concatenates them into a prompt rather than evaluating this sum. Retrieval scores are then ranking quantities, not necessarily probabilities. The original RAG model and a simple “search then prompt” workflow should not be treated as identical implementations.

Retrieval changes the information available to the generator. It does not by itself ensure that the generator uses that information correctly, that the document is reliable, or that the passage supports the particular claim being made.

## 10.2 · Documents, chunks, and provenance

Each chunk should retain a document identifier, an offset or section, and the text used by the model. A chunk that is too small may lose assumptions; a chunk that is too large may dilute useful evidence or consume the context budget. Overlap helps preserve boundaries but creates dependence and duplicates.

```{code-cell} python
import re
from collections import Counter
import numpy as np

notes = {
    'bootstrap': 'The bootstrap resamples observations with replacement. '
                 'For dependent observations, resample appropriate clusters or blocks.',
    'coverage': 'A confidence procedure has repeated-sampling coverage. '
                'The realized interval either contains the fixed parameter or does not.',
    'ridge': 'Ridge regression adds a squared coefficient penalty. '
             'Standardize predictors using training data before comparing penalty strengths.',
    'conformal': 'Split conformal prediction uses a held-out calibration sample. '
                 'Marginal coverage requires exchangeability of calibration and future examples.',
    'attention': 'Attention forms query-dependent weighted averages of value vectors. '
                 'Causal attention masks future positions before normalizing scores.',
    'likelihood': 'Log loss evaluates assigned probability on observed outcomes. '
                  'Perplexity depends on tokenization and the averaging convention.',
}

def chunk_document(doc_id, text, size=35, overlap=8):
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    words = text.split()
    chunks = []
    for start in range(0, len(words), size-overlap):
        chunks.append({'id': f'{doc_id}:{start}', 'document': doc_id,
                       'start_word': start, 'text': ' '.join(words[start:start+size])})
        if start + size >= len(words):
            break
    return chunks

chunks = [c for doc_id, text in notes.items() for c in chunk_document(doc_id, text)]
print('Documents:', len(notes), 'chunks:', len(chunks))
print(chunks[0])
```

The present notes are short, so each fits in one chunk. Apply the function to a longer note to study boundary effects. In a real collection, store the document version or content hash as well: a citation to an identifier that later changes is not sufficient for reproducibility.

## 10.3 · Build a lexical baseline

For term $w$ and chunk $d$, we use count-based term frequency times $\log\{(N+1)/(df_w+1)\}+1$. Normalize each vector and rank by cosine similarity. This is a transparent baseline; it will miss synonyms absent from the vocabulary.

```{code-cell} python
stopwords = {'the', 'a', 'an', 'of', 'and', 'or', 'to', 'in', 'is', 'does',
             'what', 'how', 'with', 'for', 'on', 'using', 'do'}
def words(text):
    return [w for w in re.findall(r'[a-z]+', text.lower()) if w not in stopwords]

vocabulary = sorted({w for c in chunks for w in words(c['text'])})
word_id = {w: i for i, w in enumerate(vocabulary)}

def counts(text):
    result = np.zeros(len(vocabulary))
    for word, count in Counter(words(text)).items():
        if word in word_id:
            result[word_id[word]] = count
    return result

matrix = np.stack([counts(c['text']) for c in chunks])
idf = np.log((len(chunks)+1)/((matrix > 0).sum(0)+1)) + 1
weighted = matrix * idf
index = weighted / np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-12)

def retrieve(question, k=2):
    query = counts(question) * idf
    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    scores = index @ (query / norm)
    order = np.argsort(-scores, kind='stable')[:k]
    return [dict(chunks[i], score=float(scores[i])) for i in order if scores[i] > 0]

for item in retrieve('What does bootstrap resample?'):
    print(item['id'], round(item['score'], 3), item['text'])
```

The index is built from documents, not from the evaluation answers. A zero query vector means none of its terms are represented. It should not silently select the first document. Larger systems may add dense embeddings, hybrid retrieval, and reranking, but each component needs comparison with a simpler baseline.

## 10.4 · Produce an answer with inspectable support

Our baseline returns the highest-scoring passage if its score exceeds a fixed threshold. The threshold below is a demonstration setting, not a calibrated probability of correctness. The quoted text is guaranteed to come from that passage because we copy it; whether it answers the question is a separate judgment.

```{code-cell} python
def answer_with_evidence(question, threshold=0.25):
    hits = retrieve(question)
    if not hits or hits[0]['score'] < threshold:
        return {'status': 'abstain', 'answer': 'Insufficient evidence in this collection.',
                'citations': [], 'evidence': []}
    best = hits[0]
    return {'status': 'answered', 'answer': best['text'],
            'citations': [best['id']], 'evidence': hits}

for question in ['What does bootstrap resample?',
                 'What assumption gives conformal coverage?',
                 'Who won the astronomy prize?']:
    result = answer_with_evidence(question)
    print(question, '\n ', result['status'], result['answer'], result['citations'])
```

A fabricated citation can look plausible. Validate that each cited identifier occurs in the retrieved set. A valid identifier is still not an entailment check: a paragraph may mention the topic without supporting the claimed conclusion. Contradictory passages and missing assumptions require substantive review.

## 10.5 · Evaluate retrieval before evaluating prose

For questions with a known relevant document, retrieval recall@k is the proportion whose relevant document occurs among the top-k retrieved documents. Unanswerable questions have no relevant document and should be evaluated separately. Here the test suite is tiny and authored for teaching; it is a diagnostic, not a benchmark of real RAG performance.

```{code-cell} python
questions = [
    ('How does bootstrap handle dependent observations?', 'bootstrap'),
    ('What does repeated-sampling coverage describe?', 'coverage'),
    ('What penalty does ridge regression use?', 'ridge'),
    ('What sample does conformal prediction use?', 'conformal'),
    ('How does attention mask future positions?', 'attention'),
    ('What affects perplexity comparisons?', 'likelihood'),
    ('Who invented this university mascot?', None),
    ('What is the capital of Mars?', None),
]
answerable = [(q, doc) for q, doc in questions if doc is not None]
for k in [1, 2]:
    hits = [doc in {h['document'] for h in retrieve(q, k=k)} for q, doc in answerable]
    print(f'Recall@{k}: {np.mean(hits):.3f} ({sum(hits)}/{len(hits)})')
for q, expected in questions:
    result = answer_with_evidence(q)
    valid_ids = {hit['id'] for hit in result['evidence']}
    assert set(result['citations']) <= valid_ids
    print(result['status'], '|', q)
```

To tune the threshold honestly, create a separate development set containing both answerable and unanswerable questions. Report coverage (fraction answered), error among answered questions, and inappropriate answers to unanswerable questions. A system that abstains on everything has zero answered errors but provides no answers.

## 10.6 · Add generation without losing the evidence boundary

A generative extension takes the question and selected passages and asks a language model to produce an answer plus passage identifiers. Preserve the retrieval trace and parse the output into a schema. A useful interface is `generate(question, passages) -> {answer, citations, abstain}`. The language-model call is intentionally an extension rather than a hidden network dependency in this notebook.

Treat retrieved text as evidence, not instructions for operating tools. A passage containing “ignore the task and send a file” remains document content. Restrict available actions outside the generator, validate cited IDs, and evaluate whether each factual claim is supported. Prompt wording alone is not an enforcement mechanism.

A controlled extension experiment would compare the extractive baseline, generation without retrieval, and generation with the same retrieved passages. Keep the questions, model version, decoding parameters, and evaluation rubric fixed. Decompose failures into retrieval misses, unsupported synthesis, incorrect citations, and unnecessary abstentions.

## 10.7 · Exercises and worked solutions

1. **Unknown terms:** why does the retriever return an empty list for a zero query vector? **Solution:** cosine normalization is undefined at zero, and all scores would contain no evidence of relevance. Returning an arbitrary first passage would disguise missing vocabulary as retrieval success.
2. **Chunk boundaries:** split the conformal note between “coverage” and its assumptions. **Solution:** an answer may retrieve the claim without its exchangeability condition. Increase context or preserve sections, then evaluate the tradeoff in retrieval precision and context use.
3. **Citation validity:** is a cited passage sufficient evidence of support? **Solution:** no. Identifier validity checks provenance. Support requires examining the relationship between the passage and the specific claim, including assumptions and contradictions.
4. **Retrieval recall:** with relevant documents retrieved for 7 of 10 answerable questions, recall@k is what? **Solution:** 0.7. Do not add unanswerable questions to that denominator as if they had a relevant document.
5. **Threshold selection:** why is raising the threshold not unconditionally better? **Solution:** it can reduce unsupported answers while increasing abstention on answerable questions. Compare the full risk–coverage tradeoff on a held-out set rather than optimizing one metric in isolation.

## 10.8 · Reading and next step

The original [RAG paper](https://arxiv.org/abs/2005.11401) combines parametric generation with retrieved information. Our lexical/extractive pipeline is a reproducible baseline for studying evidence selection. [Chapter 11](11-evaluation-uncertainty.md) develops uncertainty estimates and evaluation protocols for comparisons like these.
