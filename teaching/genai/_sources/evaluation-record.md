# One evaluation record across the course

[Download JSON](_static/evaluation/common-evaluation.json) · [Download CSV](_static/evaluation/common-evaluation.csv) · [Inspect editorial annotations](_static/evaluation/editorial-annotations.json) · [Inspect scripted counterexamples](_static/evaluation/semantic-fixtures.json)

The table contains **14 rows from the four current guided demonstrations**, including deterministic baselines. It does not contain 14 independent questions: a repeated condition on the same question is another row for the same task. The two RAG LLM calls are labeled `llm_inspection` and `llm`; use only the declared condition in a comparison. The first call remains incomplete under its concept rubric, and the fixed Agent baseline fails to fulfill the resolvable alias request. Neither failure is dropped.

The support/completeness labels are worked editorial annotations produced by the course revision assistant after inspecting the saved text and supplied evidence. They are **not an independent human rating study**. Before a substantive new experiment, define the rubric, arrange independent review where appropriate, and record disagreements. Replacing a saved answer requires a new review: the exporter checks an evidence digest rather than carrying old annotations forward silently.

## The schema and its statistical meaning

| Fields | Meaning |
|:--|:--|
| `task_id`, `task_family` | Identify the question and its scientific target. A trajectory with five calls remains one task outcome. |
| `condition`, `replicate` | Identify the intervention and repeated trajectory. The triple with task ID must be unique. |
| `data_kind` | Actual LLM, deterministic baseline, synthetic experiment, or scripted fixture. These origins are retained explicitly. |
| `model_id`, `prompt_version` | Identify the fixed model and prompt/source version. Baselines have no LLM model ID. |
| `source_record`, `source_sha256`, `call_ids` | Link a row to the exact recorded file and its zero-based call indices. |
| `raw_response`, `packed_ids`, `tool_trace` | Keep the generated text, supplied evidence IDs and tool observations, including failures. |
| `intent_correct`, `json_parseable`, `contract_valid` | Separately check intent, JSON transport and the interface contract. Null parseability means JSON was not required, as for the first plain-text explanation. A schema-valid answer can still be wrong. |
| `semantic_support`, `complete` | Is each claim supported, and does the response actually answer the question? Unreviewed values remain null. |
| `task_success`, `review_note` | Success requires correct intent, a valid contract, support, completeness, and an evidence-specific review note. |
| `llm_calls`, `tool_calls`, `latency_seconds` | Count the two types of operation separately. Latency sums recorded model-call time; loading and tool time are excluded. Baseline latency is unmeasured, not zero. |

Support labels are `supported`, `contradicted`, or `insufficient_evidence`. Null means no review. An answer can be supported but incomplete. The fixed Agent's safe refusal is procedurally valid but does not fulfill a request that its available registry could resolve. A capstone refusal is successful only when it preserves the requested settings and accurately identifies an unsupported case.

## Five fixtures that prevent a misleading score

For the question “Which resampling scheme can handle these dependent observations?” and evidence “Use an appropriate block resampling scheme,” all five rows below can have structurally valid citations. These are deliberately authored fixtures, **not model outputs**.

| Response | Support | Complete | Task success |
|:--|:--|:--|:--|
| Use an appropriate block bootstrap. | Supported | Yes | Yes |
| Dependence never matters; resample independently. | Contradicted | Yes | No |
| The sample mean is an estimator. | Insufficient evidence | No | No |
| Dependence matters. | Supported | No | No |
| A new answer awaiting review. | Null | Null | Null |

Completeness means addressing the requested information; it does not override a false claim. This explains the “complete but contradicted” row. A correct abstention on an unanswerable question requires an unanswerable-task rubric; it cannot be inferred just from an empty citation list.

## Reuse the same observations

Chapter 10 loads the actual RAG rows and distinguishes support from citation membership. Chapter 11 first selects the declared RAG task family, retaining every RAG task, and then joins baseline and LLM rows by task identity, producing one descriptive paired difference for this one question. It does not attach a population confidence interval to that difference. Its larger 240-task simulation remains a separately declared exercise in statistical inference.

Chapter 13 reads the same cost fields and converts its controlled retry/cache simulation to this schema with `data_kind="synthetic"`. Synthetic lookup attempts are tool calls, not LLM calls. It retains all four factorial cells, averages repetitions within tasks, and then resamples tasks. The real and synthetic rows are not pooled.

Chapter 14 audits all six interpreted request fields and checks that unsupported requests have zero simulation-tool calls. A person-reviewed request card is an explicit execution input; it is not an automatically generated gold label. The table cannot establish that arbitrary natural-language ambiguities have been resolved.

## Create a new row without losing failures

Use `make_row` in `labs/evaluation.py` to initialize fields to null or empty containers, fill mechanical evidence first, and add semantic labels only after review. Run `validate_rows`, then `paired_scores` for the declared conditions. The latter rejects duplicates, missing paired conditions, unreviewed scores and comparisons mixing synthetic and nonsynthetic outcomes. It never uses row order to pair questions.

`python scripts/export_evaluation.py` rebuilds the current public table from recorded evidence and the explicit editorial annotation file. That command performs no inference and invents no labels. For a new project, save a separate table, retain the task-sampling design, and state which sources of uncertainty the comparison actually addresses.
