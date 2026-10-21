---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# C · Four guided live notebooks

**One working boundary at a time.** Start after Chapter 6; return after Chapters 10, 12, and 14. These notebooks make actual calls to the pinned local Qwen2.5-1.5B instruction model. Each can run from a fresh kernel. The website and this appendix display saved outputs and audit records; ordinary book rendering does not perform new inference.

| Milestone | Read first | Download the executed live notebook |
|:--|:--|:--|
| One real LLM call | Chapter 6, A.5–A.8, A.11–A.12 | [01 · First LLM](https://chikuang.github.io/teaching/genai/_static/live-notebooks/01-first-llm.ipynb) |
| Same-pipeline RAG | Chapter 10, A.9, A.11 | [02 · RAG bridge](https://chikuang.github.io/teaching/genai/_static/live-notebooks/02-rag-bridge.ipynb) |
| Recovery after an observation | Chapters 11–12 | [03 · Agent bridge](https://chikuang.github.io/teaching/genai/_static/live-notebooks/03-agent-bridge.ipynb) |
| Interpreted simulation request | Chapter 14 | [04 · Capstone bridge](https://chikuang.github.io/teaching/genai/_static/live-notebooks/04-capstone-bridge.ipynb) |

The editable source notebooks are in `labs/notebooks/` in the [course repository](https://github.com/chikuang/Teaching-GenAI/tree/main/labs/notebooks). The downloads include executed outputs. Saving a download does not install Python, model weights, or the shared course modules. Keep it inside the cloned course folder.

## Prepare once, then run one notebook

Follow [Python setup](../python-workflow.md), then enter these commands in a **terminal at the repository root**:

```bash
python -m pip install -r requirements.txt -r requirements-llm.txt
python labs/download_model.py
python -m jupyterlab
```

The download is about 3 GB; CPU inference in float32 needs several additional GB of RAM. No GPU or paid API is required. Installation and initial download need internet; generation uses local files. Open one notebook in JupyterLab and run its cells in order with **Shift+Enter**. **Restart Kernel and Run All Cells** checks that no object from an earlier session is required. The first model-loading cell may take time; do not start several copies while it is busy.

Each notebook begins by finding the course root from its current directory or `GENAI_COURSE_ROOT`. It then imports the small course helpers. If the folder cannot be found, stop and fix the path. If `torch` cannot be imported, check the kernel interpreter against the course virtual environment. A model output rejected by a validator is an experimental outcome after setup has succeeded, not an installation error.

## C.1 · See the complete first call

Notebook 1 separates six operations: find the course, load the tokenizer and weights, construct messages, inspect serialized input and token IDs, generate once, and save evidence. `LocalLM` is an inspectable wrapper around `AutoTokenizer` and `AutoModelForCausalLM`, not a different modeling framework.

```python
messages = [
    {'role': 'system', 'content': 'Explain statistical ideas in one short sentence.'},
    {'role': 'user', 'content': 'What is Monte Carlo error?'},
]
with lm.torch.inference_mode():
    output = lm.model.generate(**inputs, max_new_tokens=64, do_sample=False,
                               pad_token_id=lm.tokenizer.eos_token_id)
new_ids = output[0, inputs.input_ids.shape[1]:]
answer = lm.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
```

`lm` and `inputs` are defined in earlier cells of the complete notebook. The notebook expands the actual generation and slicing operations, then calls a separate logging method that performs no further inference. The call uses greedy decoding, returns only newly generated text, and saves prompts, raw output, token counts, stop reason and runtime. Inspecting a token tensor before generation lets you connect this pretrained model to the next-character conditionals in Chapter 6. Different tokenizers make raw token-level scores incomparable without a declared normalization.

Technical acceptance checks one actual call, the input shape, and the output budget. The semantic rubric asks whether the answer identifies **finite simulation draws** as the source of Monte Carlo error and distinguishes this from estimation error and model misspecification. The recorded short answer below refers to random sampling but does not explicitly identify the finite simulation budget; it is an incomplete explanation under that rubric. A nonempty answer is not sufficient acceptance.

```{code-cell} python
from pathlib import Path
import os
import json
import sys


start = Path(os.environ.get('GENAI_COURSE_ROOT', Path.cwd())).resolve()
root = None
# Examine this folder, then each enclosing folder.
candidates = [start] + list(start.parents)
for candidate in candidates:
    if (candidate / 'labs/model.json').is_file():
        root = candidate
        break
if root is None:
    raise FileNotFoundError('Keep the notebook inside Teaching-GenAI, or set GENAI_COURSE_ROOT')
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
course_root = root
records = {}
for name in ['first-llm', 'rag-bridge', 'agent-bridge', 'capstone-bridge']:
    records[name] = json.loads((root / 'docs/_static/experiments' / (name+'.json')).read_text())
first = records['first-llm']['calls'][0]
print('SAVED ACTUAL MODEL CALL; no inference during this cell')
print(first['raw'])
print('Prompt/new tokens:', first['prompt_tokens'], first['generated_tokens'])
print('Stop reason:', first['stop_reason'])
```

## C.2 · Replace the generator while keeping the experiment fixed

Notebook 2 imports the exact six-note Chapter 10 pipeline from `labs/rag_core.py`. The course tests compare its definitions with the chapter. Chunking, IDs, vocabulary, TF-IDF, threshold, word budget and context serialization remain fixed when the generator changes.

| Component | Chapter 10 baseline | Guided real-model version |
|:--|:--|:--|
| Corpus through packed context | Six statistical notes | Same notes and functions |
| Generator input | `build_messages(question, selected)` | Identical messages |
| Generator | Copies first passage | Actual local model call |
| Transport | Python response dictionary | Bare JSON or exactly one enclosing JSON fence |
| Response contract | Shared `validate_response` | Same shared function |
| Semantic support | Must be checked separately | Must be checked separately |

The adapter permits one enclosing code fence, then applies strict JSON and field checks. It never repairs a citation, number, answer, or field name. A malformed response is retained and rejected. The extractive baseline is a separate comparison; it never replaces a failed LLM output.

The first guided run exposed an underspecified prompt: the old instruction requested an “object” without explicitly requiring JSON. The model returned YAML-like text and a shortened citation. The [initial trace](../_static/experiments/rag-bridge-initial.json) is preserved. We then changed the shared Chapter 10 prompt to state the exact JSON shape and full-ID requirement and reran the same question. This is **development after observed failure**, not new held-out evidence. Its corresponding source is archived in `labs/archived-guided-initial/`.

```{code-cell} python
rag = records['rag-bridge']['results']
print('Current status:', rag['result']['status'])
print('Answer:', rag['result']['answer'])
print('Citations:', rag['result']['citations'])
print('Context unchanged:', rag['result']['trace']['context_ids'] == rag['baseline']['trace']['context_ids'])
print('Messages unchanged:', rag['result']['trace']['messages'] == rag['baseline']['trace']['messages'])
```

The recorded answer preserves the need to resample appropriate clusters or blocks for dependence and cites the supplied bootstrap passage. That supports this narrow answer under the course rubric. It does not establish general RAG factuality. Inspect no-evidence, invalid-output and structurally-valid-but-unsupported cases separately; the notebook includes fixtures for those distinctions.

## C.3 · Observe a decision change after failure

Notebook 3 uses the current shared executor and one action schema, `{tool, arguments}`. `lookup` stores observed values by dataset identity; `summarize` receives that identity, not a model-supplied list of values. `finish` must match the latest computed result. A separate rubric checks whether that result answers the requested dataset and statistic.

The alias `spring_pilot` cannot be looked up directly. A failed lookup supplies an error observation. The new `resolve` tool consults a small registry and returns a canonical dataset name; the policy must then look up that dataset before summarizing it. An unknown alias must lead to abstention. Both baselines and the LLM have a maximum of five decisions and four tool attempts, including failed attempts. Finishing consumes a decision but no tool attempt.

```{code-cell} python
agent = records['agent-bridge']['results']
for name in ['fixed', 'programmed', 'actual']:
    result = agent[name]
    print(name, result['status'], 'answer:', result['answer'],
          'decisions:', result['decisions'], 'tool attempts:', result['tool_calls'])
for step in agent['actual']['trace']:
    print(step['decision'], step.get('action', {}).get('tool'), step['status'])
print('Exact dataset/statistic/value rubric:', agent['task_success'])
```

The corrected log stores immutable input snapshots: the five prompts have lengths **2, 4, 6, 8, 10**. The tokenizer-serialized prompt is also retained. The [old v2 trace](../_static/experiments/agent-bridge-v2.json) is preserved with its limitation: all its prompt lists were later mutated to the final conversation, so they are not valid per-turn input evidence. The original code is archived in `labs/archived-guided-v2/`. The new trace is a fresh execution, not a repair of historical model output.

The fixed workflow stops at failed lookup. The adaptive programmed policy and the actual LLM both complete the recovery in the recorded demonstration. This shows the observable branching mechanism. Because a short programmed rule also solves the task, it does not establish that an LLM is needed. The rule baselines receive structured intent; their comparison isolates decision sequencing, not natural-language parsing accuracy. A broader comparison must score interpretation and include its calls and costs as well.

## C.4 · Finish a bounded workflow before adding generated prose

Notebook 4 declares the architecture explicitly:

**Natural-language request → real LLM proposal → independently reviewed request card → full-field and scope checks → coverage tool → deterministic report.**

The LLM proposes **family, n, method, confidence, repetitions and dependence**. No field may be omitted. A separate request card represents the reader's review of what was actually asked, including ambiguities; it is an input safeguard, not an automatically discovered gold label. With no reviewed card the status is `needs_confirmation` and no simulation runs. A supported card that differs from the proposal produces `intent_mismatch`. An unsupported card leads to an explicit refusal, never a substitution.

The current live scope is iid normal/exponential, n=5–500, nominal 95%, the z interval with estimated SD, and 500 repetitions. The seven declared tasks test normal and exponential cases plus unsupported family, Student-t method, 99% confidence, 2000 repetitions and dependence. The report uses the same approximate z interval and Wilson Monte Carlo interval as Chapter 14; metadata are flat here and nested under `request` in the chapter. Human review of a card is still needed for arbitrary free text; the program does not prove that every unmodeled requirement was captured.

The report template inserts the tool's numbers and distinguishes nominal coverage, estimated coverage and Monte Carlo uncertainty. This is the declared reporting component for every request, not a fallback for a failed LLM report. The next experimental stage is LLM-written prose, scored separately against this evidence and retained even when it fails.

```{code-cell} python
for row in records['capstone-bridge']['results']:
    print('Request:', row['question'])
    print('Status:', row['status'], 'intent correct:', row['intent_correct'],
          'task success:', row['task_success'])
    print(row.get('report', row.get('error')))
```

In this fresh actual run, both supported requests completed. All five unsupported requests preserved their six interpreted settings and were refused before simulation. Thus seven of seven authored development cases met the declared task rubric; this is not a population accuracy estimate. The [previous v2 capstone](../_static/experiments/capstone-bridge-v2.json), including its failed Cauchy interpretation, remains available. The revised prompt, schema and request-review gate define a new condition, so the old and new counts should not be pooled.

For normal sampling, compare the estimated coverage to `2 * t.cdf(1.96, n-1) - 1`, not automatically to 0.95. The live notebook performs this independent analytic check. The exponential result needs a separate reference or numerical audit. Neither a narrow Monte Carlo interval nor a fluent explanation proves that the original request was interpreted correctly.

## C.5 · Evidence and exercise acceptance

Each notebook has three levels: predict an intermediate result, implement a bounded change with explicit fixtures, and design one new comparison. The source notebooks have empty outputs so you can start fresh; the published copies retain a full successful *execution* even when the model itself fails a task. An execution check asks whether all cells ran; task success asks whether the system met its substantive contract.

| Stage | Technical acceptance | Additional substantive acceptance |
|:--|:--|:--|
| First call | Real call, bounded tokens, recorded stop | Explanation meets a prewritten concept rubric |
| RAG | JSON and common response contract | Every claim supported by supplied evidence |
| Agent | Allowed actions, budgets, computed finish | Correct requested dataset and statistic |
| Capstone | Valid request, checked numerical tool | Original intent preserved; claims scoped correctly |

For a new evaluation table include task ID and family, condition, model/prompt version, raw response, packed source IDs or tool trace, parseability, contract validity, semantic review, task success, calls and elapsed time. Use development cases for fixes; freeze changes before new evaluation cases. Repeated outputs on one task are not independent tasks. These few authored cases warrant descriptive counts and failure analysis, not population confidence intervals.

The [official Transformers chat-template guide](https://huggingface.co/docs/transformers/v4.57.1/chat_templating) explains the model's message serialization. The course's statistical checks, validators, tools and evaluation rubrics are independently authored teaching components.

## C.6 · Carry the same rows into evaluation and your project

Download the [common evaluation table](../evaluation-record.md). It joins these exact saved calls with explicit editorial annotations, keeping parse/contract validity, semantic support, completeness and overall task success separate. For example, the first LLM answer is structurally acceptable but incomplete under the finite-simulation rubric. A rejected or incomplete answer remains in the denominator.

The same rows are loaded in Chapter 10 for evidence review, Chapter 11 for pairing, Chapter 13 for cost accounting, and Chapter 14 for request verification. The larger simulated examples remain separately labeled synthetic data. Use the [independent student assignments](../assignments.md) to create a new case; the 14 student notebooks contain TODOs and acceptance checks, while the chapter and live notebook downloads retain executed worked examples.
