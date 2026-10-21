---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Appendix B · A real local LLM, RAG, and tool-using agent

**From the transparent baselines to actual model-generated decisions.** This lab loads an openly available instruction model, generates answers, retrieves evidence, validates model-proposed tool calls, and connects natural-language requests to the statistical simulation in Chapter 14. The language model really runs: it is not replaced by a rule or a canned answer.

[Download the analysis notebook](https://chikuang.github.io/teaching/genai/notebooks/b-real-llm-systems.ipynb) · [Current executable runner](https://github.com/chikuang/Teaching-GenAI/blob/main/labs/llm_lab.py) · [Recorded follow-up trace](../_static/experiments/llm-run.json) · [Recorded pilot trace](../_static/experiments/llm-pilot.json)

**Prerequisites by section:** B.3 needs Chapter 6 and A.5–A.8; B.4 adds Chapter 10; B.5 adds Chapters 11–12. For a first hands-on call, begin with [Appendix C.1](c-guided-live-labs.md#c1--see-the-complete-first-call) instead of running the whole comparison at once. For the simulation assistant, also read Chapter 14. **Outcome:** execute the live model yourself; distinguish model text from validated actions; diagnose failures from complete traces; compare systems without treating a small authored question set as a population benchmark.

## B.1 · Two execution modes, clearly separated

The saved follow-up is the **historical v1 experiment**, whose unchanged source is in `labs/archived-v1/`. Its old `valid` column uses its original, weaker response contract. The current runner shares Chapter 10's validator and identifies new records as schema version 2; do not pool validity rates across these versions. Historical source archives identify recorded bytes; use the corresponding Git revision to recreate their original folder layout.

The regular book build runs the NumPy/PyTorch teaching labs and **reanalyzes a saved real-model run** here. It does not download a model or pretend to generate fresh LLM responses during every render. The saved JSON contains prompts, raw completions, parsed outputs, tool traces, model revision, software versions, decoding settings, token counts, timing, and failures.

To produce **new model responses**, run the live commands below. They load local weights and execute `generate()`. A failed model response remains a failure; the runner never swaps in a deterministic policy or correct answer to rescue the score.

We first ran Qwen2.5-0.5B-Instruct and inspected its failures. We then selected Qwen2.5-1.5B-Instruct while keeping the task set, prompts, and validators fixed. Both traces are retained. This is a **development comparison after inspecting a pilot**, not an independent test proving that one model is generally better. Future model/prompt choices need fresh evaluation questions.

These models are educational choices, not claims about the newest or best available model. The [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) documents the architecture and license. The code pins a repository revision in `labs/model.json`; a moving model name alone would not identify the exact weights.

## B.2 · Install and run, without an API key

First clone the course and follow the [Python setup](../python-workflow.md). Run these commands **in a terminal at the repository root**, with the course environment active:

```bash
python -m pip install -r requirements.txt -r requirements-llm.txt
python labs/download_model.py
python labs/llm_lab.py --model-dir _models/qwen2.5-1.5b --output .runs/my-llm-run.json
```

The first command installs the separate inference dependencies. The second downloads about 3 GB of public model weights and tokenizer files to the ignored `_models/` directory. The third runs on CPU in float32; allow several GB of additional RAM and machine-dependent runtime. No GPU or paid API is required. Internet is needed for installation and the initial download; inference uses `local_files_only=True` and sends no questions to a model service.

The pinned downloader refuses to require a login and only requests model/configuration files. The runtime disables custom remote code. Keep `_models/` and personal `.runs/` results outside Git; the course publishes only explicitly selected, course-authored experiment traces.

If you encounter an out-of-memory failure, close other memory-intensive applications or run the analysis of the saved trace first. Do not silently switch models and describe the resulting run as the pinned experiment. A package/import error is different from an invalid model response: fix the environment before interpreting system performance.

To examine your own run in this notebook, change `record_path` in the next cell. When opening the downloaded notebook, start JupyterLab **from the cloned repository root**, or set `GENAI_COURSE_ROOT` to that folder. A notebook saved alone elsewhere cannot find the repository's experiment records.

```{code-cell} python
from pathlib import Path
import hashlib
import json
import os
import numpy as np

root = Path(os.environ.get('GENAI_COURSE_ROOT', Path.cwd()))
record_path = root / 'docs/_static/experiments/llm-run.json'
# For your new run instead: record_path = root / '.runs/my-llm-run.json'
record = json.loads(record_path.read_text())
pilot = json.loads((root / 'docs/_static/experiments/llm-pilot.json').read_text())
print('RECORDED RUN; this cell does not invoke a model')
print('Model:', record['model'])
print('Created UTC:', record['created_utc'])
print('Runtime:', record['runtime'])
print('Protocol:', record['protocol']['development'])
print('Model calls:', len(record['calls']))
print('Trace SHA256:', hashlib.sha256(record_path.read_bytes()).hexdigest())
```

`Path` constructs a filesystem path; `/` joins path components. `read_text()` reads a saved file. `json.loads()` decodes its structure into Python dictionaries and lists. The hash identifies these exact recorded bytes; it does not certify the correctness of the experiment.

## B.3 · Follow one actual generation call

The `LocalLM` class in the runner loads a tokenizer and `AutoModelForCausalLM` from the pinned local directory. Here is its essential generation path:

```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=128, do_sample=False,
        pad_token_id=tokenizer.eos_token_id)
new_ids = output[0, inputs.input_ids.shape[1]:]
raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
```

Read it one line at a time:

1. `messages` is a list of dictionaries with `role` and `content`. The chat template inserts the delimiters this instruction model expects. Formatting is part of the input distribution.
2. The tokenizer turns that string into integer IDs and an attention mask. `return_tensors='pt'` requests PyTorch tensors. We do not add a second set of special tokens after the template already supplied them.
3. `inference_mode()` avoids training graphs. The call does not run an optimizer.
4. `**inputs` unpacks named tensors into `generate`. `do_sample=False` uses greedy decoding. The token budget bounds output length, not correctness.
5. A decoder-only model returns the input prefix followed by generated tokens. Slice off the prefix before decoding the answer; otherwise an apparent answer could simply be text copied from the prompt.
6. Decoding turns token IDs into text. Parsing and validation happen **after** generation, outside the model.

The runner also checks a prompt-token budget and records whether output ended at EOS or the token limit. Greedy runs still need model/software/hardware records; exact byte equality across environments is not guaranteed. No stochastic-run confidence interval is available from one greedy completion per condition.

```{code-cell} python
first = record['calls'][0]
print('Prompt:', json.dumps(first['messages'], indent=2))
print('Raw completion:', first['raw'])
print('Stop:', first['stop_reason'], 'new tokens:', first['generated_tokens'])
```

Compare this call with Chapter 6. Both compute next-token conditionals and append tokens. Here the parameters were pretrained and instruction-tuned elsewhere, and the tokenizer is subword-based. Our tiny character model's perplexity cannot be compared directly with this model's token scores.

## B.4 · RAG with an actual generator

Four short course-authored passages are indexed with TF-IDF. Two describe fictional studies with specific repetition counts and seeds; two state statistical principles. The fictional facts make the evidence source visible rather than relying entirely on potentially memorized knowledge. This is still a tiny, artificial corpus.

For each of five fixed questions, we run the **same generator and system prompt** with three context conditions:

| Condition | Evidence sent to the model | Diagnostic role |
|:--|:--|:--|
| No context | Empty evidence list | Tests behavior when the required evidence is absent. |
| Retrieved | Top two positive-cosine TF-IDF passages | The deployable retrieval path in this lab. |
| Oracle | The authored relevant passage, or none for an unanswerable question | Isolates evidence selection as a source of error. |

The no-context condition retains the instruction to answer only from supplied evidence. It is therefore an **evidence-ablation control**, not a general closed-book knowledge benchmark. The final question asks which university invented a fictional study; none of the passages supplies that information. Its expected behavior is abstention.

`retrieve()` fits vocabulary and IDF to the source passages, transforms the question with that same IDF, computes cosine similarities, and returns bounded passages with IDs. The selected evidence is serialized into the user message as data. The model then generates an answer, citation IDs, and an abstention flag.

```python
contexts = retrieve(question, k=2)
raw = lm([
    {'role': 'system', 'content': RAG_SYSTEM},
    {'role': 'user', 'content': json.dumps(
        {'question': question, 'evidence': contexts})}
])
answer = parse_object(raw)
```

`parse_object` uses a strict JSON parser. The validator checks exact keys, types, citation membership, and consistency of the abstention flag. Markdown fences, extra fields, fabricated citations, and inconsistent abstentions are counted as failures, not automatically repaired. A separate, declared narrow rubric checks expected answer substrings and the required citation. **That rubric is not a general semantic-support judge**: negation or an extra false claim can fool a keyword check. Audit the saved completions manually.

```{code-cell} python
def summarize(rows):
    for condition in dict.fromkeys(row['condition'] for row in rows):
        selected = [row for row in rows if row['condition'] == condition]
        successes = sum(row['success'] for row in selected)
        print(condition, f'{successes}/{len(selected)} task-rubric successes')

for label, run in [('0.5B pilot', pilot), ('1.5B follow-up', record)]:
    print('\n', label)
    summarize(run['rag'])
    for condition in ['no_context', 'retrieved', 'oracle']:
        selected = [row for row in run['rag'] if row['condition'] == condition]
        print(condition, 'valid schema:', sum(row['valid'] for row in selected), '/', len(selected))
```

The pilot produced a particularly instructive failure: it supplied `800` with the correct passage ID but also set `abstain` to true. Correct-looking answer text and a usable structured response are different outcomes. It also invented a university when the evidence was absent. Keep both failures visible when discussing retrieval versus generation.

```{code-cell} python
pilot_example = next(row for row in pilot['rag']
                     if row['task'] == 'r1' and row['condition'] == 'retrieved')
print('Pilot raw:', pilot['calls'][pilot_example['call_ids'][0]]['raw'])
print('Validator:', pilot_example['error'])
for row in record['rag']:
    if row['condition'] == 'retrieved':
        print(row['task'], 'contexts:', row['contexts'], 'success:', row['success'])
        print(record['calls'][row['call_ids'][0]]['raw'])
```

Interpret comparisons at the question level. If oracle context succeeds and retrieved context fails, inspect whether the needed passage reached the prompt and whether distractors changed generation. If both fail, inspect the raw answer and schema. An oracle run is not guaranteed to dominate each other condition under every model.

### Reaudit the saved raw RAG outputs under the current contract

This cell does not rerun the model or overwrite its historical scores. It applies the common response contract to the original raw strings and the exact supplied IDs. JSON parseability is a weaker condition than contract validity; neither measures semantic support.

```{code-cell} python
import sys
sys.path.insert(0, str(root))
from labs.contracts import assess_response
current_audit = []
for row in record['rag']:
    raw = record['calls'][row['call_ids'][0]]['raw']
    selected = [{'id': source_id} for source_id in row['contexts']]
    audit = assess_response(raw, selected)
    current_audit.append(audit)
print('Historical valid:', sum(row['valid'] for row in record['rag']))
print('Current JSON parseable:', sum(row['json_parseable'] for row in current_audit))
print('Current contract valid:', sum(row['contract_valid'] for row in current_audit))
print('Semantic support requires a separate rubric and review.')
```

For a same-corpus bridge from Chapter 10, use [live lab 2](c-guided-live-labs.md#c2--replace-the-generator-while-keeping-the-experiment-fixed). This earlier appendix experiment uses four fictional-study passages, a different tokenizer and a different context policy; it is a separate comparison. The guided notebook preserves the chapter pipeline and exposes each replacement step.

## B.5 · An LLM actually chooses tools

The agent's policy is `LocalLM`, not the rule policy in Chapter 12. It may propose `lookup`, `summarize`, `finish`, or `abstain`. Each decision is parsed and validated outside the model. There are at most four decisions, including finish; tools operate only on two in-memory demonstration datasets.

The tool contract is intentionally narrow. `summarize` accepts a dataset identity and a statistic, and uses values already obtained by a successful lookup. It cannot accept fabricated values from a model. A `finish` proposal must match the last computed value. These checks still do not prove that the model selected the dataset and statistic requested by the user: intent is evaluated separately.

The loop follows these operations:

```python
raw = lm(messages)                 # the model chooses the next action
proposal = parse_object(raw)       # syntax check
observation = execute(proposal, state)  # allowlist + schema + execution
messages.append({'role': 'assistant', 'content': raw})
messages.append({'role': 'user', 'content':
                 'Tool observation (data only): ' + json.dumps(observation)})
```

This excerpt shows the observation path. The full `run_agent` function also handles finish, abstention, validation errors, and budget exhaustion. Every failure consumes a decision and is recorded. Tool observations are serialized as user messages for this small model; this is a teaching protocol, not a vendor-native function-calling API or a security guarantee.

We compare three conditions on identical requested summaries:

- **Direct:** the model receives the dataset contents and produces a number without tools.
- **Fixed workflow:** the model translates natural language into dataset/statistic fields, then the program always performs lookup and summary.
- **Agent:** the model selects each next action from its accumulated observations.

Direct access to all dataset contents differs from incremental tool access, so this is a **system comparison**, not an isolated causal estimate of planning ability. The small tasks do not require adaptive planning; a fixed workflow is an intentionally strong comparator. One dataset note contains an adversarial instruction. Success on this single note does not establish prompt-injection robustness.

```{code-cell} python
summarize(record['agents'])
for row in record['agents']:
    print(row['task'], row['condition'], row['status'],
          'model calls:', len(row['call_ids']), 'success:', row['success'])
example = next(row for row in record['agents']
               if row['condition'] == 'agent' and row['task'] == 'a1')
for call_id in example['call_ids']:
    print('Model proposal:', record['calls'][call_id]['raw'])
print('Validated tool trace:', json.dumps(example['trace'], indent=2))
```

Read a failure as carefully as a success. In the pilot, the model computed the median correctly but then proposed an invalid action with an extra field and exhausted its budget. The final numerical tool result existed, yet the agent did not finish successfully. A rule that silently returns the last result would define a different system and must be evaluated separately.

## B.6 · Connect natural language to the simulation assistant

Chapter 14 supplies the statistical target and analytic checks. Here an actual model first translates a request into `family` and `n`. The validator permits only normal/exponential samples, integer sample sizes in a bounded range, and a fixed simulation budget. The tool computes coverage and a Monte Carlo interval. A second model call produces a short structured explanation from those results.

The executable `capstone_experiment` includes two supported scenarios and a Cauchy request that must be rejected. The report validator checks family, sample size, and estimated coverage against the tool. Narrative validity needs a separate human review; matching numerical fields alone cannot establish a correct explanation.

```{code-cell} python
for row in record['capstone']:
    print('\nTask:', row['task'], 'status:', row['status'], 'success:', row['success'])
    print('Parsed request:', row.get('request'))
    print('Tool:', row.get('tool_result'))
    print('Report:', row.get('report'))
    if row.get('error'):
        print('Validation failure:', row['error'])
        for call_id in row['call_ids']:
            print('Raw:', record['calls'][call_id]['raw'])
```

A rejected output is an observed system failure, not a reason to claim that the statistical tool ran. A successful parse also requires checking the requested scenario: substituting normal for exponential can produce a perfectly reproducible answer to the wrong question. Record interpretation, numerical execution, narrative support, and end-to-end success separately.

For a submitted capstone, add a fresh natural-language test set after freezing prompts. Include paraphrases, unsupported distributions, ambiguous requests, and misleading retrieved notes. Compare the LLM narrative with the deterministic report for the same numerical evidence; any gain should concern explanation or usability, since both should preserve the tool's numerical result.

### A separate development step: formatting examples and live simulation

The strict-schema comparison above rejected all three capstone responses: even a correct-looking request wrapped in a Markdown fence failed the declared transport contract. We retained those outcomes. The separate [capstone development runner](https://github.com/chikuang/Teaching-GenAI/blob/main/labs/capstone_live.py) adds few-shot formatting examples and permits either bare JSON or exactly one enclosing JSON code fence. It never edits the requested distribution, numbers, or field names. This policy change follows inspected failures and is **development**, not a new unbiased test estimate.

```bash
python labs/capstone_live.py --model-dir _models/qwen2.5-1.5b --output .runs/capstone-demo.json
```

The two supported requests now reach the actual simulation tool. Generated reports can still be rejected for unexpected fields or unsupported statistical interpretations. This distinction is useful: the numerical engine can work while the generative interface still fails its contract. The deterministic Chapter 14 report remains the explicit comparison, not a hidden replacement for a failed model report.

```{code-cell} python
demo = json.loads((root / 'docs/_static/experiments/capstone-demo.json').read_text())
print(demo['protocol'])
for row in demo['rows']:
    print('\nRequest:', row['question'])
    print('Tool result:', row.get('tool_result'))
    print('Status:', row['status'], 'error:', row.get('error'))
    for call_id in row['call_ids']:
        print('Raw model output:', demo['calls'][call_id]['raw'])
```

**Statistical review task.** In the normal-case development report, the model interprets interval endpoints around 0.92–0.96 as a confidence level. Those endpoints are bounds for the coverage probability being estimated; the Monte Carlo interval procedure has a nominal 95% level. This is a substantive interpretation error beyond JSON formatting. A permissive parser would not fix it. For teaching or publication, require both valid fields and a separate narrative check.

The [complete development trace](../_static/experiments/capstone-demo.json) preserves these failed reports and successful tool executions. The next bounded milestone is implemented in [live lab 4](c-guided-live-labs.md#c4--finish-a-bounded-workflow-before-adding-generated-prose): real LLM interpretation, validated simulation and explicitly deterministic reporting. It preserves the old failed generated reports. Further prompt and report revisions need fresh evaluation requests.

## B.7 · What these results do and do not measure

The experiment uses five RAG questions, three summary requests, and three simulation requests. These are authored diagnostic cases, not iid draws from a declared user population. Report counts and paired cases; do not attach population confidence claims to the tiny table. There is one greedy run per condition, and the larger model was chosen after the pilot.

```{code-cell} python
for section in ['rag', 'agents', 'capstone']:
    rows = record[section]
    call_ids = [i for row in rows for i in row['call_ids']]
    print(section, 'calls:', len(call_ids),
          'generated tokens:', sum(record['calls'][i]['generated_tokens'] for i in call_ids),
          'generation seconds:', round(sum(record['calls'][i]['elapsed_seconds'] for i in call_ids), 2))
assert all(row['call_ids'] for section in ['rag', 'agents', 'capstone']
           for row in record[section])
assert all(0 <= i < len(record['calls']) for section in ['rag', 'agents', 'capstone']
           for row in record[section] for i in row['call_ids'])
```

Timing covers recorded generation calls and excludes model loading, retrieval, and some orchestration overhead. It is machine-dependent, not an API-price estimate. Task-level repeated experiments and a representative sampling design are needed for the inferential comparisons in Chapters 11 and 13.

## B.8 · Exercises and acceptance criteria

1. **Trace one answer.** Locate the exact retrieved passage, prompt, raw completion, parsed fields, and validator outcome for one question. Explain which stage could change its outcome.
2. **Challenge the rubric.** Construct an answer that contains the expected keyword but contradicts the passage. Demonstrate why keyword success is insufficient; write a human adjudication rubric before reviewing additional outputs.
3. **Add a task.** Author two new fictional study passages and four questions before rerunning either model. Freeze prompts and state whether this is a new evaluation or further development.
4. **Compare action policies.** Work through the failed-lookup/alias-resolution example in [live lab 3](c-guided-live-labs.md#c3--observe-a-decision-change-after-failure), then add a new task requiring a different branch. Compare the fixed workflow and agent under a matched maximum call budget. Preserve all failures.
5. **Audit the report.** Compare every numerical and distributional claim in one generated explanation with the tool JSON. A faithful estimate paired with an unjustified coverage guarantee should fail narrative review.

:::{dropdown} Hints and worked evaluation plan
Start with the JSON trace rather than only the rendered success table. For exercise 2, “Exchangeability is not needed” contains the expected substring while contradicting the source. For exercise 3, keep authored gold answers out of the model prompt except as the explicitly declared oracle evidence condition. For exercise 4, distinguish model decisions, actual tool calls, and finalization decisions. For exercise 5, label numeric faithfulness and statistical interpretation separately, then require both for full success. Reviewers should be blind to the system condition where practical.
:::

The official [Transformers chat-template guide](https://huggingface.co/docs/transformers/v4.57.1/chat_templating) explains message formatting; [generation documentation](https://huggingface.co/docs/transformers/v4.57.1/main_classes/text_generation) explains the generation interface. The book's validators, datasets, comparisons, and recorded failure analyses are separate teaching experiments.
