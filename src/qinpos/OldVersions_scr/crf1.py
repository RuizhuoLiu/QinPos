"""Linear-chain CRF over the (candidate, hand) fingering lattice.

The perceptron - single argmax path; the CRF - probability,
per-note MARGINALS defined - good for the fingerboard visualisation.

"""

from __future__ import annotations

import math
import random
from pathlib import Path

from .candidates import candidates_for
from .learn import PieceSequence, WeightVector, gold_path
from .theory import Candidate, Note
from .viterbi import FEATURES, _next_hand, arc_cost, node_cost, path_features


def _logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    m = max(values)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def _lattice(notes: list[Note], kinds=None) -> list[list[Candidate]]:
    cols = []
    for i, n in enumerate(notes):
        cands = candidates_for(n)
        if kinds is not None and kinds[i] is not None:
            restricted = [c for c in cands if c.kind == kinds[i]]
            cands = restricted or cands
        cols.append(cands)
    return cols


def forward_backward(cols: list[list[Candidate]], w):
    """Forward-backward (candidate, hand) expansion."""
    n = len(cols)
    # states[i]: list of (j, hand); alpha/beta: parallel lists of logs
    states: list[list[tuple[int, float | None]]] = []
    alphas: list[list[float]] = []

    # forward
    first: dict[tuple[int, float | None], float] = {}
    for j, c in enumerate(cols[0]):
        key = (j, _next_hand(None, c))
        score = -node_cost(c, w)
        first[key] = _logsumexp([first.get(key, float("-inf")), score])
    states.append(list(first.keys()))
    alphas.append([first[k] for k in states[0]])

    for i in range(1, n):
        nxt: dict[tuple[int, float | None], float] = {}
        prev_states, prev_alpha = states[i - 1], alphas[i - 1]
        for j, cur in enumerate(cols[i]):
            nc = -node_cost(cur, w)
            for (k, hand), pa in zip(prev_states, prev_alpha):
                s = pa + nc - arc_cost(cols[i - 1][k], cur, w, hand)
                key = (j, _next_hand(hand, cur))
                nxt[key] = _logsumexp([nxt.get(key, float("-inf")), s])
        states.append(list(nxt.keys()))
        alphas.append([nxt[k] for k in states[i]])

    logZ = _logsumexp(alphas[-1])

    # backward
    betas: list[list[float]] = [[] for _ in range(n)]
    betas[n - 1] = [0.0] * len(states[n - 1])
    for i in range(n - 2, -1, -1):
        cur_index = {s: t for t, s in enumerate(states[i + 1])}
        col_next = cols[i + 1]
        out = []
        for k, hand in states[i]:
            prev_cand = cols[i][k]
            terms = []
            for j, cur in enumerate(col_next):
                key = (j, _next_hand(hand, cur))
                t = cur_index.get(key)
                if t is None:
                    continue
                terms.append(-node_cost(cur, w) - arc_cost(prev_cand, cur, w, hand) + betas[i + 1][t])
            out.append(_logsumexp(terms))
        betas[i] = out

    # expectations
    exp_f = {k: 0.0 for k in FEATURES}

    # node terms: marginal of each state, summed into its candidate's features
    from .viterbi import arc_features, node_features

    for i in range(n):
        for (j, hand), a, b in zip(states[i], alphas[i], betas[i]):
            p = math.exp(a + b - logZ)
            if p <= 0.0:
                continue
            for k, v in node_features(cols[i][j]).items():
                if v:
                    exp_f[k] += p * v

    # edge terms: P(prev_state, cur_state) for every allowed transition
    for i in range(1, n):
        cur_index = {s: t for t, s in enumerate(states[i])}
        for (k, hand), pa in zip(states[i - 1], alphas[i - 1]):
            prev_cand = cols[i - 1][k]
            for j, cur in enumerate(cols[i]):
                key = (j, _next_hand(hand, cur))
                t = cur_index.get(key)
                if t is None:
                    continue
                ac = arc_cost(prev_cand, cur, w, hand)
                p = math.exp(pa - ac - node_cost(cur, w) + betas[i][t] - logZ)
                if p <= 0.0:
                    continue
                for name, v in arc_features(prev_cand, cur, hand).items():
                    if v:
                        exp_f[name] += p * v
    return logZ, exp_f  # expected_features is the model expectation


def nll_and_grad(seq: PieceSequence, w, gold: list[Candidate] | None = None):
    """NLL of the gold path and its gradient wrt w.

    gold defaults to learn.gold_path(seq, w): the expert pinned wherever reachable, best completion elsewhere
    """
    if gold is None:
        gold = gold_path(seq, w)
    cols = _lattice(seq.notes)
    logZ, exp_f = forward_backward(cols, w)
    f_gold = path_features(gold)
    cost_gold = w.dot(f_gold)  # works for WeightVector and Weights alike
    nll = cost_gold + logZ
    grad = {k: f_gold.get(k, 0.0) - exp_f.get(k, 0.0) for k in FEATURES}
    return nll, grad


def train_crf(
    train_seqs: list[PieceSequence],
    epochs: int = 20,
    lr: float = 0.5,
    l2: float = 1e-4,
    init=None,
    seed: int = 0,
    verbose: bool = True,
) -> WeightVector:
    """Adagrad SGD on the CRF negative log-likelihood, one piece/step. Adagrad (per-feature adaptive step, Duchi et al. 2011)
    Since the features live on wildly different scales:
        string/band - 0/1, hand_travel sums 10 of hui per piece.

    l2 regularisation keeps pushing weights outward as long as the model is not certain,
    and ~20 training pieces will happily overfit without it.
    """
    rng = random.Random(seed)
    w = WeightVector(init)
    g2 = {k: 1e-8 for k in FEATURES}  # accumulated squared gradients
    order = list(range(len(train_seqs)))
    n_notes = sum(len(s.notes) for s in train_seqs)
    for ep in range(epochs):
        rng.shuffle(order)
        total_nll = 0.0
        for idx in order:
            seq = train_seqs[idx]
            nll, grad = nll_and_grad(seq, w)
            total_nll += nll
            for k in FEATURES:
                g = grad[k] + l2 * w[k]
                g2[k] += g * g
                w[k] -= lr * g / math.sqrt(g2[k])
        if verbose:
            print(f"epoch {ep + 1:2d}  mean NLL/note = {total_nll / n_notes:.4f}")
    return w


def note_marginals(notes: list[Note], w, kinds=None) -> list[dict[Candidate, float]]:
    """Per-note marginal probability of each candidate

    For visualisation: for note i, a dict mapping each playable Candidate to P(candidate_i = c | melody).
    Probabilities in each dict sum to 1.
    """
    cols = _lattice(notes, kinds)
    n = len(cols)
    # rerun forward-backward but keep per-state marginals
    # (duplicate of forward_backward but separate so the training path stays lean; both call the same scorers)
    states: list[list[tuple[int, float | None]]] = []
    alphas: list[list[float]] = []
    first: dict[tuple[int, float | None], float] = {}
    for j, c in enumerate(cols[0]):
        key = (j, _next_hand(None, c))
        first[key] = _logsumexp([first.get(key, float("-inf")), -node_cost(c, w)])
    states.append(list(first.keys()))
    alphas.append([first[k] for k in states[0]])
    for i in range(1, n):
        nxt: dict[tuple[int, float | None], float] = {}
        for j, cur in enumerate(cols[i]):
            nc = -node_cost(cur, w)
            for (k, hand), pa in zip(states[i - 1], alphas[i - 1]):
                s = pa + nc - arc_cost(cols[i - 1][k], cur, w, hand)
                key = (j, _next_hand(hand, cur))
                nxt[key] = _logsumexp([nxt.get(key, float("-inf")), s])
        states.append(list(nxt.keys()))
        alphas.append([nxt[k] for k in states[i]])
    logZ = _logsumexp(alphas[-1])
    betas: list[list[float]] = [[] for _ in range(n)]
    betas[n - 1] = [0.0] * len(states[n - 1])
    for i in range(n - 2, -1, -1):
        cur_index = {s: t for t, s in enumerate(states[i + 1])}
        out = []
        for k, hand in states[i]:
            prev_cand = cols[i][k]
            terms = []
            for j, cur in enumerate(cols[i + 1]):
                key = (j, _next_hand(hand, cur))
                t = cur_index.get(key)
                if t is not None:
                    terms.append(-node_cost(cur, w) - arc_cost(prev_cand, cur, w, hand) + betas[i + 1][t])
            out.append(_logsumexp(terms))
        betas[i] = out

    result: list[dict[Candidate, float]] = []
    for i in range(n):
        acc: dict[Candidate, float] = {}
        for (j, hand), a, b in zip(states[i], alphas[i], betas[i]):
            c = cols[i][j]
            acc[c] = acc.get(c, 0.0) + math.exp(a + b - logZ)
        # normalise away accumulated float error
        z = sum(acc.values()) or 1.0
        result.append({c: p / z for c, p in acc.items()})
    return result
