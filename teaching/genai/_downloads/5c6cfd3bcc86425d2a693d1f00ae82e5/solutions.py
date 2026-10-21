"""Separate worked answers. Attempt the student notebooks before opening this file."""
import numpy as np


def counts_error(counts, probabilities):
    counts, probabilities = np.asarray(counts), np.asarray(probabilities, dtype=float)
    return float(np.max(np.abs(counts/counts.sum()-probabilities)))


def logistic_loss_gradient(X, y, beta):
    logits = X @ beta
    return float(np.mean(np.logaddexp(0, logits)-y*logits)), X.T @ (1/(1+np.exp(-logits))-y)/len(y)


def neighbor_overlap(first, second, k):
    return len(set(first[:k]) & set(second[:k]))/k


def causal_average(scores, values):
    masked = np.where(np.tril(np.ones_like(scores, dtype=bool)), scores, -np.inf)
    weights = np.exp(masked-masked.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, weights @ values


def prefix_unchanged(before, after, stop):
    return bool(np.allclose(before[:, :stop], after[:, :stop], atol=1e-6, rtol=1e-5))


def grammar_hit(generation, allowed):
    return generation['text'] in allowed and generation['stop_reason'] == 'end_token'


def near_center_mass(samples, centers, radius):
    distance = np.sqrt(((samples[:, None, :]-centers[None, :, :])**2).sum(axis=2)).min(axis=1)
    return float(np.mean(distance <= radius))


def normal_score(x, mean, variance):
    return -(np.asarray(x)-mean)/variance


def lora_parameter_count(din, dout, rank):
    return rank*(din+dout)


def citation_contract(response, allowed_ids):
    from labs.contracts import validate_response
    try:
        validate_response(response, [{'id': key} for key in allowed_ids])
        return True
    except (ValueError, TypeError):
        return False


def paired_differences(a, b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonempty finite paired vectors required')
    return b-a


def checked_finish(value, state):
    from labs.llm_lab import numeric
    return bool(numeric(value) and state['results'] and value == state['results'][-1]['value'])


def factorial_interaction(means):
    return means[(3,True)]-means[(3,False)]-means[(1,True)]+means[(1,False)]


def can_execute(proposal, confirmed):
    from labs.guided_systems import validate_spec, scope_reasons
    try:
        validate_spec(proposal)
        validate_spec(confirmed)
        return proposal == confirmed and not scope_reasons(confirmed)
    except (ValueError, TypeError):
        return False


SOLUTIONS = [counts_error, logistic_loss_gradient, neighbor_overlap, causal_average,
             prefix_unchanged, grammar_hit, near_center_mass, normal_score,
             lora_parameter_count, citation_contract, paired_differences, checked_finish,
             factorial_interaction, can_execute]
