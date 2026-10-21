---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Appendix A · Python and PyTorch, one operation at a time

**For a statistician learning a new programming language.** You do not need to relearn regression to learn Python. You do need to recognize which objects hold data, which operations transform them, and which instructions change fitted parameters.

[Download the executed notebook](https://chikuang.github.io/teaching/genai/notebooks/a-python-pytorch.ipynb) · [Installation](../python-workflow.md)

Read A.1–A.4 before Chapter 2, A.5–A.8 before Chapter 5, and A.9–A.10 before Chapters 10–14. Run each cell in order. Predict its output before pressing **Shift+Enter**. The examples are deliberately smaller than the chapter models; once each operation is familiar, return to the full experiment.

## A.1 · Names, values, and containers

An assignment `name = expression` evaluates the right side and associates its result with the name. It is not a mathematical equality constraint. Python indexes from zero. Parentheses call a function, brackets select an element, and braces construct a dictionary or set.

```{code-cell} python
n = 4                         # integer: a count
alpha = 0.05                  # float: an approximate real number
label = 'pilot'               # string: text
values = [2.0, 4.0, 6.0, 8.0] # list: an ordered container
request = {'dataset': label, 'n': n}  # dictionary: keys map to values
print(type(n).__name__, type(alpha).__name__, type(label).__name__)
print(values[0], values[-1])   # first and last items
print(values[1:3])             # positions 1 and 2; the stop index is excluded
print(request['dataset'])     # use a dictionary key, not a numeric position
```

`#` begins a comment. `print()` displays a value; it does not save a model. A string such as `'4'` is text, while `4` is an integer. The Boolean values `True` and `False` represent logical conditions. `None` means no value has been supplied; it is different from zero.

Lists have reference semantics. Copy before an independent modification:

```{code-cell} python
original = [2, 4]
alias = original
independent = original.copy()
alias.append(6)
print('original:', original, 'independent copy:', independent)
```

Both `alias` and `original` refer to the same list. This is why a saved model checkpoint needs independent tensors, not just another name for tensors that training will change.

## A.2 · Functions, indentation, and small loops

A function declares its inputs after `def`. The indented body runs when the function is called. `return` sends a result back to the caller. A default value is used only when that argument is omitted.

```{code-cell} python
def centered(numbers, center=0.0):
    result = []
    for number in numbers:
        result.append(number - center)
    return result

shifted = centered(values, center=5.0)
print(shifted)
```

Read the loop literally: take each number, subtract the center, append the result. The equivalent list comprehension is shorter, but not a different algorithm:

```{code-cell} python
compact = [number - 5.0 for number in values]
assert compact == shifted
for index, number in enumerate(values):
    print(index, number)
```

`enumerate()` supplies both index and value. `assert` raises an error when its condition is false. Use assertions for diagnostic invariants; use explicit exceptions for public input validation. A notebook kernel retains objects between cells. Restart and run all cells to discover hidden dependence on an earlier experiment.

## A.3 · Arrays, shapes, and axes

A NumPy array stores numerical values with a common data type. Import a library once, and use its short name as a prefix. `shape` is a tuple describing axis lengths. Rows usually index observations, but that convention is ours: the computer does not know what an observation is.

```{code-cell} python
import numpy as np
X = np.array([[1., 2.], [3., 4.], [5., 6.]])
print('X shape:', X.shape)
print('first observation:', X[0, :])
print('first variable:', X[:, 0])
print('column means:', X.mean(axis=0))
print('row means:', X.mean(axis=1))
```

An operation along `axis=0` reduces the row axis, leaving one answer per column. `axis=1` reduces the column axis. A colon selects all entries of that axis. `X[:, 0]` has shape `(3,)`; `X[:, :1]` has shape `(3, 1)`. Those shapes can behave differently in arithmetic.

```{code-cell} python
prediction = np.array([[0.2], [0.8]])
target = np.array([0., 1.])
wrong = prediction - target
right = prediction[:, 0] - target
print('unexpected residual matrix:', wrong, 'shape:', wrong.shape)
print('intended residual vector:', right, 'shape:', right.shape)
assert right.shape == target.shape
```

Broadcasting compares shapes from the right. Dimensions must be equal or one. `(2,1)` and `(2,)` therefore produce `(2,2)`: the code runs, but the estimand has changed. Check shapes before averaging a loss. `*` multiplies elementwise; `@` is matrix multiplication.

```{code-cell} python
beta = np.array([0.5, -0.25])
linear_predictor = X @ beta
print(linear_predictor, linear_predictor.shape)
assert np.allclose(linear_predictor, (X * beta).sum(axis=1))
```

`allclose()` compares floating-point numbers within tolerances. Exact `==` is useful for integer identities but often inappropriate for numerically equivalent real-valued calculations.

## A.4 · Random generators and reproducible simulation

A generator stores a pseudorandom state. A seed initializes it; each subsequent draw advances it. Recreating a generator inside every replicate with the same seed would repeat the same sample.

```{code-cell} python
rng = np.random.default_rng(2026)
first = rng.normal(size=3)
second = rng.normal(size=3)
restarted = np.random.default_rng(2026).normal(size=3)
print('first:', first.round(3))
print('next:', second.round(3))
assert np.array_equal(first, restarted)
assert not np.array_equal(first, second)
```

Reproducibility lets us inspect a computation. It does not establish external validity, independence of test tasks, or robustness across training seeds. `.round(3)` changes the displayed numerical values here; keep unrounded arrays for fitting and scoring.

## A.5 · Tensors and one derivative

A PyTorch tensor resembles a NumPy array but can record a computation graph. `requires_grad=True` asks PyTorch to track derivatives with respect to that tensor. For squared-error regression, you already know the derivative; use it to check the software.

```{code-cell} python
import torch
from torch import nn
torch.set_num_threads(1)
torch.manual_seed(2026)
x = torch.tensor([1., 2., 3.])
y = torch.tensor([2., 4., 6.])
beta_t = torch.tensor(0.5, requires_grad=True)
predicted = x * beta_t
loss = ((predicted - y) ** 2).mean()
loss.backward()
analytic = (2 * x * (predicted - y)).mean()
print('loss:', loss.item(), 'autodiff:', beta_t.grad.item())
assert torch.allclose(beta_t.grad, analytic)
```

`backward()` propagates derivatives to leaf parameters and stores them in `.grad`. `.item()` extracts a Python scalar for reporting. `.detach()` produces a tensor view without gradient tracking; `.detach().cpu().numpy()` converts a tensor to a NumPy array for analysis. Do not detach the loss before `backward()`: that would sever the derivative path.

Gradients accumulate until cleared. That allows deliberate accumulation over batches, but an ordinary training step must clear the preceding step's gradient.

## A.6 · What a model class means

A class groups state and operations. An *instance* is one concrete object made from the class. `self` means this instance. `__init__` runs at construction. `super().__init__()` initializes PyTorch's parent class, allowing it to register parameters and submodules. `forward()` declares the prediction calculation.

```{code-cell} python
class OneFeatureRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs):
        return self.linear(inputs)

model = OneFeatureRegression()
inputs = x.reshape(3, 1)
targets = y.reshape(3, 1)
print('input:', inputs.shape, 'output:', model(inputs).shape)
for name, parameter in model.named_parameters():
    print(name, tuple(parameter.shape))
```

`nn.Linear(1,1)` stores a slope and intercept. Calling `model(inputs)` invokes the module machinery and then `forward`; use this form rather than calling `forward` directly. The weights are initially random. A model object specifies a family and holds parameter values; creating it does not fit it.

## A.7 · Perform one update, then build a loop

The optimizer receives references to the model parameters. Its learning rate controls the size of an update. It does not choose our statistical loss or data split.

```{code-cell} python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
optimizer.zero_grad()                   # 1. erase old gradients
predicted = model(inputs)               # 2. calculate fitted values
loss = ((predicted - targets) ** 2).mean()  # 3. evaluate the objective
loss.backward()                        # 4. differentiate the objective
optimizer.step()                       # 5. update registered parameters
print('pre-update loss:', loss.item())
with torch.no_grad():
    print('post-update loss:', ((model(inputs)-targets)**2).mean().item())
```

The loss tensor still contains the pre-update loss; call the model again to evaluate new parameters. `with` establishes a context for the indented block. `torch.no_grad()` disables graph recording inside that block; it does not freeze parameters against all possible mutation.

Now repeat the same operations. This deliberately uses only three training observations to explain optimization; it does not estimate generalization performance.

```{code-cell} python
for epoch in range(60):
    optimizer.zero_grad()
    loss = ((model(inputs) - targets) ** 2).mean()
    loss.backward()
    optimizer.step()
print('trained predictions:', model(inputs).detach().squeeze(-1).numpy().round(2))
```

`.squeeze(-1)` removes a final axis only if its length is one. Prefer an explicit axis: a bare `.squeeze()` can also remove a batch axis when a batch has one observation.

`model.train()` and `model.eval()` change the behavior of modules such as dropout and batch normalization. They do **not** turn gradient tracking on and off. For validation we normally use both `model.eval()` and `torch.no_grad()`. Our simple linear module has neither dropout nor batch normalization, but the distinction matters later.

```{code-cell} python
import copy
checkpoint = copy.deepcopy(model.state_dict())
model.eval()
with torch.no_grad():
    validation_example = model(torch.tensor([[4.]]))
print('new input prediction:', validation_example.item())
model.load_state_dict(checkpoint)
```

`state_dict()` holds registered parameters and buffers. `deepcopy()` makes independent tensor copies. Chapter 2 saves such a snapshot only when validation loss improves. The test observations must remain outside that selection process.

## A.8 · Follow the transformer dimensions

For attention, axes represent batch, token position, head, and head coordinate. `reshape` changes how coordinates are grouped; `transpose` changes the order of axes. Neither operation learns weights.

```{code-cell} python
B, T, D, H = 2, 3, 8, 2
features = torch.arange(B*T*D).reshape(B, T, D).float()
by_head = features.reshape(B, T, H, D//H).transpose(1, 2)
restored = by_head.transpose(1, 2).contiguous().reshape(B, T, D)
print('before:', features.shape, 'heads:', by_head.shape, 'after:', restored.shape)
assert torch.equal(features, restored)
```

Start with `(B,T,D)`. Split the last axis to obtain `(B,T,H,D/H)`. Swap axes 1 and 2 to obtain `(B,H,T,D/H)`. Each head now has its own token matrix. `.contiguous()` creates a conventional memory layout when needed; it does not alter the mathematical values. `reshape` can copy when required, while some uses of `view` require a compatible layout.

In Chapter 5, `q, k, v = ...chunk(3, dim=-1)` splits three equal feature blocks. This list comprehension:

```python
q, k, v = [a.reshape(B, T, H, D//H).transpose(1, 2) for a in (q, k, v)]
```

applies the same operation separately to three tensors. It is equivalent to writing three explicit assignments. For shapes `(B,H,T,d)`, `q @ k.transpose(-2,-1)` has shape `(B,H,T,T)`; softmax over `dim=-1` normalizes each query's weights over keys.

```{code-cell} python
scores = torch.zeros(3, 3)
allowed = torch.ones(3, 3, dtype=torch.bool).tril()
weights = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
print(weights)
assert torch.allclose(weights.sum(-1), torch.ones(3))
assert torch.equal(weights.triu(1), torch.zeros(3, 3))
```

`tril()` retains the lower triangle, including the diagonal. `~allowed` negates the Boolean mask. Mask **scores before softmax** so disallowed keys receive zero probability after normalization. At least one allowed key is required per row.

## A.9 · Structured requests, validation, and errors

JSON is an interchange format for objects such as dictionaries, lists, strings, numbers, booleans, and null. Parsing JSON only checks syntax. A valid JSON object can still request an unsupported operation.

```{code-cell} python
import json
from dataclasses import dataclass

@dataclass(frozen=True)
class SummaryRequest:
    dataset: str
    statistic: str

def parse_request(text):
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {'dataset', 'statistic'}:
        raise ValueError('Expected exactly dataset and statistic')
    if not isinstance(payload['dataset'], str):
        raise ValueError('Dataset must be text')
    if payload['statistic'] not in ('mean', 'median'):
        raise ValueError('Unsupported statistic')
    return SummaryRequest(**payload)

request = parse_request('{"dataset": "pilot_a", "statistic": "mean"}')
print(request, request.statistic)
```

A decorator such as `@dataclass` transforms the class definition; here it supplies an initializer and a readable representation. `frozen=True` prevents ordinary reassignment of its fields. `**payload` unpacks dictionary entries into named arguments. Type annotations document intent; they do not validate values automatically. The explicit checks still matter.

```{code-cell} python
try:
    parse_request('{"dataset": "pilot_a", "statistic": "delete"}')
except ValueError as error:
    print('Rejected:', error)
```

`try` attempts an operation; `except` handles a specified failure. Do not catch every error and substitute a successful-looking answer. Preserve status, error type, and trace. In the LLM lab, text proposals never become arbitrary Python through `eval()` or `exec()`.

## A.10 · Exercises: predict, implement, investigate

1. **Predict.** What are the shapes of `X[0]`, `X[:1]`, and `X[:, :1]`? Predict before printing.
2. **Implement.** Complete `column_center` below without a Python loop. Check that it preserves shape and produces zero column means.
3. **Debug.** Deliberately remove `zero_grad()` from a copy of the training loop. Explain the changed update in terms of accumulated derivatives; do not claim a general optimizer ranking from this example.
4. **Trace.** For `(B,T,D,H)=(2,5,12,3)`, write every shape from features to attention scores and back.
5. **Validate.** Extend `parse_request` with a dataset allowlist. Test malformed JSON, extra keys, an unknown dataset, and a supported request.

```python
def column_center(array):
    # Replace ... with a vectorized expression.
    return ...

# Run after completing the function:
# centered_X = column_center(X)
# assert centered_X.shape == X.shape
# assert np.allclose(centered_X.mean(axis=0), 0)
```

:::{dropdown} Hints
A slice preserves the sliced axis. Use `mean(axis=0, keepdims=True)` for column centering. For debugging gradients, write the first two updates algebraically before changing the code. A schema and an allowlist answer different validation questions.
:::

:::{dropdown} Worked solutions and checks
1. The shapes are `(2,)`, `(1,2)`, and `(3,1)`.
2. Return `array - array.mean(axis=0, keepdims=True)`. This subtracts one training-column mean per variable. For validation data, reuse means fitted on training data rather than recentering validation independently.
3. Without clearing, the second backward call adds the current derivative to the derivative left from the first call. The optimizer steps using the accumulated quantity. This is not ordinary independent minibatch SGD.
4. `(2,5,12)` becomes `(2,5,3,4)`, then `(2,3,5,4)`. Scores have shape `(2,3,5,5)`. Multiplying weights by values gives `(2,3,5,4)`; transpose and reshape restore `(2,5,12)`.
5. Reject a dataset unless it is in an explicit allowed set. Ensure every invalid test raises `ValueError` and the supported request returns the requested fields unchanged.
:::

Use the official [Python tutorial](https://docs.python.org/3/tutorial/), [NumPy broadcasting guide](https://numpy.org/doc/stable/user/basics.broadcasting.html), and [PyTorch autograd introduction](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) as references. Learn these operations through small computations before adding another framework.
