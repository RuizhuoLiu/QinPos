import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase 3b: linear-chain CRF

    Trains `qinpos.crf` on the same by-piece split as `train_weights.py`.
    The CRF adds two things the perceptron cannot give: per-note
    marginals (the visualisation input) and an l2 knob.

    Verified once (Aug 2026): finite-difference gradient check 5e-10,
    brute-force path probabilities sum to 1.000000, inference exact at
    246 states/column.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    from qinpos.crf import note_marginals, train_crf
    from qinpos.learn import (build_sequences, context_free_eval,
                              context_free_table, evaluate, train)
    from qinpos.viterbi import (ARC_FEATURES, BAND_FEATURES, CONTEXT_FEATURES)

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data/GQ39/score_annotation"
    CLEAN = ROOT / "data/gq39_clean.csv"
    WEIGHTS_OUT = ROOT / "data/crf_weights.json"

    EPOCHS, LR, L2 = 10, 0.5, 0.1  # NLL plateaus by epoch 5
    return (
        ARC_FEATURES,
        BAND_FEATURES,
        CLEAN,
        CONTEXT_FEATURES,
        DATA,
        EPOCHS,
        L2,
        LR,
        WEIGHTS_OUT,
        build_sequences,
        context_free_eval,
        context_free_table,
        evaluate,
        json,
        note_marginals,
        train,
        train_crf,
    )


@app.cell
def _(CLEAN, DATA, build_sequences):
    seqs = build_sequences(DATA, CLEAN)
    seqs.sort(key=lambda s: s.piece)
    test_seqs = [s for i, s in enumerate(seqs) if i % 4 == 0]
    train_seqs = [s for i, s in enumerate(seqs) if i % 4 != 0]
    print(f"train {len(train_seqs)} / test {len(test_seqs)} pieces")
    print("test:", [s.piece for s in test_seqs])
    return test_seqs, train_seqs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Baselines
    """)
    return


@app.cell
def _(
    context_free_eval,
    context_free_table,
    evaluate,
    test_seqs,
    train,
    train_seqs,
):
    _cf = context_free_eval(context_free_table(train_seqs), test_seqs)
    print(f"context-free lookup: exact={_cf['exact_acc']:.1%} (n={_cf['n']})")

    _p = evaluate(test_seqs, train(train_seqs, epochs=15, lr=0.05))
    print(f"perceptron:          exact={_p['exact_acc']:.1%} "
          f"kind={_p['kind_acc']:.1%} string={_p['string_acc']:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Metrics used below

    `evaluate` gives Viterbi-path exact match. `topk` and `calibration`
    read the marginals instead: the first is the shortlist metric this
    task actually wants (a pitch with five valid expert realisations
    cannot be scored fairly by top-1), the second checks the marginals
    are meaningful enough to drive the fingerboard heat map.
    """)
    return


@app.cell
def _(note_marginals):
    def _rank(m):
        seen, order = set(), []
        for c, _ in sorted(m.items(), key=lambda kv: -kv[1]):
            if (c.string, c.kind) not in seen:
                seen.add((c.string, c.kind))
                order.append((c.string, c.kind))
        return order

    def topk(sequences, w, label, ks=(1, 2, 3, 5)):
        n, hits = 0, {k: 0 for k in ks}
        for seq in sequences:
            for i, m in enumerate(note_marginals(seq.notes, w)):
                if not seq.scored[i]:
                    continue
                n += 1
                order = _rank(m)
                truth = (seq.expert[i].string, seq.expert[i].kind)
                for k in ks:
                    hits[k] += truth in order[:k]
        print(f"  {label} (n={n}): " + "  ".join(
            f"top-{k} {hits[k] / max(1, n):.1%}" for k in ks))

    def calibration(sequences, w, bins=((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))):
        rows = {b: [0, 0] for b in bins}
        for seq in sequences:
            for i, m in enumerate(note_marginals(seq.notes, w)):
                if not seq.scored[i]:
                    continue
                top_c, top_p = max(m.items(), key=lambda kv: kv[1])
                for b in bins:
                    if b[0] <= top_p < b[1]:
                        rows[b][0] += 1
                        rows[b][1] += ((top_c.string, top_c.kind)
                                       == (seq.expert[i].string, seq.expert[i].kind))
                        break
        print(f"  {'top marginal':>14} {'notes':>7} {'accuracy':>9}")
        for b in bins:
            n, hit = rows[b]
            print(f"  {b[0]:.1f} - {b[1]:.1f}      {n:7d} "
                  + (f"{hit / n:8.1%}" if n else "       --"))

    return calibration, topk


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. l2 sweep

    Four orders of magnitude move held-out accuracy by under 2pp: at
    lr=0.5 the per-piece gradient is tens of feature counts and l2*w is
    a rounding error below ~0.1. The train-test gap tracks which
    FEATURES are on, not how hard they are penalised — which is what
    section 4 measures.
    """)
    return


@app.cell
def _(EPOCHS, LR, evaluate, test_seqs, train_crf, train_seqs):
    def l2_sweep(grid=(1e-4, 1e-2, 1e-1, 3e-1, 1.0)):
        print(f"  {'l2':>7} {'train':>7} {'test':>7} {'gap':>7} {'max|w|':>7}")
        for l2 in grid:
            w = train_crf(train_seqs, epochs=EPOCHS, lr=LR, l2=l2, verbose=False)
            tr = evaluate(train_seqs, w)["exact_acc"]
            te = evaluate(test_seqs, w)["exact_acc"]
            print(f"  {l2:7.4f} {tr:7.1%} {te:7.1%} {tr - te:+7.1%} "
                  f"{max(abs(v) for v in w.values()):7.2f}")

    l2_sweep()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Ablation grid

    Every row retrains from scratch with the named features frozen at zero, so the survivors re-fit around the absence.

    The GAP tracks the arc features and nothing else. Arc features condition on the model's own previous decision, right about half the time here; a confidently weighted transition rule applied from a wrong anchor excludes the correct candidate rather than merely failing to help. The v6 context features read the next note's candidate SET, which follows from the input melody and cannot be got wrong — same information, no propagation.

    Bands and context both encode register-dependent string preference, one by memorising 28 parameters and one by generalising over 5, and 1286 training notes only support one of them.
    """)
    return


@app.cell
def _(
    ARC_FEATURES,
    BAND_FEATURES,
    CONTEXT_FEATURES,
    EPOCHS,
    L2,
    LR,
    evaluate,
    test_seqs,
    train_crf,
    train_seqs,
):
    ABLATIONS = {
        "node only (v4 set)": ARC_FEATURES + CONTEXT_FEATURES,
        "node + context": ARC_FEATURES,
        "node + arc (v5)": CONTEXT_FEATURES,
        "everything": (),
        "context, no bands": ARC_FEATURES + BAND_FEATURES,
    }

    def run_ablation():
        out = {}
        print(f"  {'configuration':22s} {'train':>7} {'test':>7} {'gap':>7} "
              f"{'kind':>7} {'string':>7} {'str|kind':>9}")
        print("  " + "-" * 70)
        for name, frozen in ABLATIONS.items():
            w = train_crf(train_seqs, epochs=EPOCHS, lr=LR, l2=L2,
                          frozen=frozen, verbose=False)
            te, tr = evaluate(test_seqs, w), evaluate(train_seqs, w)
            kg = evaluate(test_seqs, w, kinds_known=True)
            out[name] = w
            print(f"  {name:22s} {tr['exact_acc']:7.1%} {te['exact_acc']:7.1%} "
                  f"{tr['exact_acc'] - te['exact_acc']:+7.1%} {te['kind_acc']:7.1%} "
                  f"{te['string_acc']:7.1%} {kg['string_acc']:9.1%}")
        print("\n  reference: perceptron 48.5%, n-gram backoff 53.3% top-1 / 79.4% top-3")
        return out

    ablated = run_ablation()
    w_best = ablated["context, no bands"]
    return (w_best,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Best configuration
    """)
    return


@app.cell
def _(CONTEXT_FEATURES, calibration, test_seqs, topk, w_best):
    topk(test_seqs, w_best, "context, no bands")
    print()
    calibration(test_seqs, w_best)
    print("\n  context weights:")
    for _k in CONTEXT_FEATURES:
        print(f"    {_k:18s} {w_best[_k]:+.3f}")
    return


@app.cell
def _(WEIGHTS_OUT, json, w_best):
    json.dump(dict(w_best), WEIGHTS_OUT.open("w"), indent=1)
    print(f"saved {WEIGHTS_OUT}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Marginals demo

    Uncertain notes are where the fingerboard visual shows several
    half-lit alternatives, and they are the notes to put to the
    collaborating performer: "are these both acceptable here?" turns
    model uncertainty into a measurement of the task's real ambiguity.
    """)
    return


@app.cell
def _(note_marginals, test_seqs, w_best):
    _seq = test_seqs[0]
    _margs = note_marginals(_seq.notes, w_best)
    _tops = [max(m.values()) for m in _margs]
    print(f"piece {_seq.piece}: {len(_tops)} notes, "
          f"{sum(t > 0.8 for t in _tops)} confident (P>0.8), "
          f"{sum(t < 0.5 for t in _tops)} uncertain (P<0.5)")
    print("\nfirst 8 notes, top-3 candidates:")
    for _i, _m in enumerate(_margs[:8]):
        _s = "   ".join(f"{c}:{p:.2f}"
                        for c, p in sorted(_m.items(), key=lambda kv: -kv[1])[:3])
        print(f"  {_i:2d}  {_s}   <- expert: {_seq.expert[_i]}")
    return


if __name__ == "__main__":
    app.run()
