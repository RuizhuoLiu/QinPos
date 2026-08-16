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
    # Phase 3: perceptron weight learning + style control

    Trains the decoder's cost weights from GQ39 expert paths with an
    averaged structured perceptron, on full sequences with free timbre
    choice. Split is by piece, never by note, to avoid leakage.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    from qinpos.learn import (build_sequences, check_idx, context_free_ceiling,
                              context_free_eval, context_free_table, evaluate,
                              reach_report, train)
    from qinpos.viterbi import (ARC_FEATURES, BAND_FEATURES, FEATURES,
                                STRING_BIAS_FEATURES, Weights, decode)

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data/GQ39/score_annotation"
    CLEAN_CSV = ROOT / "data/gq39_clean.csv"
    WEIGHTS_JSON = ROOT / "data/learned_weights.json"

    EPOCHS, LR = 40, 0.05
    return (
        ARC_FEATURES,
        BAND_FEATURES,
        CLEAN_CSV,
        DATA,
        EPOCHS,
        FEATURES,
        LR,
        WEIGHTS_JSON,
        Weights,
        build_sequences,
        check_idx,
        context_free_ceiling,
        context_free_eval,
        context_free_table,
        decode,
        evaluate,
        json,
        reach_report,
        train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Loader sanity: does sorting on idx preserve order?
    """)
    return


@app.cell
def _(DATA, check_idx):
    check_idx(DATA)
    return


@app.cell
def _(CLEAN_CSV, DATA, build_sequences):
    seqs = build_sequences(DATA, CLEAN_CSV)
    seqs.sort(key=lambda s: s.piece)
    test_seqs = [s for i, s in enumerate(seqs) if i % 4 == 0]
    train_seqs = [s for i, s in enumerate(seqs) if i % 4 != 0]

    _n = sum(len(s.notes) for s in seqs)
    _sc = sum(sum(s.scored) for s in seqs)
    print(f"{len(seqs)} pieces, {_n} notes, {_sc} scored")
    print(f"train: {len(train_seqs)} pieces   "
          f"test: {[s.piece for s in test_seqs]}")
    return seqs, test_seqs, train_seqs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Reachability

    Run with `scored_only=False`: unreachable notes there are an
    independent check on the `needs_review` tier.
    """)
    return


@app.cell
def _(reach_report, seqs):
    reach_report(seqs, scored_only=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Context-free baselines

    Honest baseline: fit on train, score on test — the number to beat.

    Oracle ceiling: fit and score on the same notes, so not a baseline;
    it bounds what any pitch-only mapping could reach (54.7%).
    """)
    return


@app.cell
def _(context_free_eval, context_free_table, test_seqs, train_seqs):
    cf_table = context_free_table(train_seqs)
    cf = context_free_eval(cf_table, test_seqs)
    print("context-free lookup (fit on train, scored on test)")
    print(f"  exact={cf['exact_acc']:.1%}  (n={cf['n']}, "
          f"{cf['unseen_pitch']} pitches unseen in training)")
    return (cf,)


@app.cell
def _(context_free_ceiling, test_seqs):
    print("per-piece oracle ceiling (test pieces)")
    context_free_ceiling(test_seqs, per_piece=True, detail=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Hand-crafted weights (Path A baseline)

    `Weights()` defaults from `viterbi.py`, set by domain knowledge.
    """)
    return


@app.cell
def _(Weights, evaluate, test_seqs):
    _b = evaluate(test_seqs, Weights())
    print("hand-crafted (test, free choice)")
    print(f"  kind={_b['kind_acc']:.1%}  string={_b['string_acc']:.1%}  "
          f"exact={_b['exact_acc']:.1%}  (n={_b['n']})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Train the full model

    Then: lr sweep (does lr explain the arc penalty or not), arc-scale sweep (is the optimum at zero or not), and error clustering (do arc features turn isolated errors into runs or not).
    """)
    return


@app.cell
def _(EPOCHS, FEATURES, LR, WEIGHTS_JSON, Weights, json, train, train_seqs):
    learned = train(train_seqs, epochs=EPOCHS, lr=LR, init=Weights())
    WEIGHTS_JSON.write_text(json.dumps(dict(learned), indent=2))
    print(f"saved {WEIGHTS_JSON}")
    for _k in FEATURES:
        if abs(learned[_k]) > 1e-9:
            print(f"  {_k:14s} {learned[_k]:+.3f}")
    return (learned,)


@app.cell
def _(evaluate, learned, test_seqs, train_seqs):
    _free = evaluate(test_seqs, learned)
    _kg = evaluate(test_seqs, learned, kinds_known=True)
    _tr = evaluate(train_seqs, learned)
    print(f"v4 full (test, free):   kind={_free['kind_acc']:.1%} "
          f"string={_free['string_acc']:.1%} exact={_free['exact_acc']:.1%}")
    print(f"v4 full (test, timbre given): string={_kg['string_acc']:.1%} "
          f"({_kg['kind_fallback']} notes lost the constraint)")
    print(f"v4 full (train, free):  exact={_tr['exact_acc']:.1%} "
          f"(below test = underfitting, above test = overfitting)")
    return


@app.cell
def _(ARC_FEATURES, Weights, evaluate, test_seqs, train, train_seqs):
    def lr_sweep(label, frozen=(), lrs=(0.2, 0.05, 0.01, 0.002), epochs=40):
        print(f"{label}")
        print(f"  {'lr':>7} {'train':>7} {'test':>7}")
        for lr in lrs:
            w = train(train_seqs, epochs=epochs, lr=lr,
                      init=Weights(), frozen=frozen)
            tr = evaluate(train_seqs, w)["exact_acc"]
            te = evaluate(test_seqs, w)["exact_acc"]
            print(f"  {lr:7.3f} {tr:7.1%} {te:7.1%}")

    lr_sweep("full model (arc + bands)")
    lr_sweep("node only (no arc)", frozen=ARC_FEATURES)
    return


@app.cell
def _(ARC_FEATURES, evaluate, learned, test_seqs):
    def arc_scale_sweep(w0, scales=(1.0, 0.75, 0.5, 0.25, 0.1, 0.0)):
        print(f"  {'arc scale':>10} {'kind':>7} {'string':>7} {'exact':>7}")
        for s in scales:
            w = w0.copy()
            for k in ARC_FEATURES:
                w[k] *= s
            r = evaluate(test_seqs, w)
            print(f"  {s:10.2f} {r['kind_acc']:7.1%} "
                  f"{r['string_acc']:7.1%} {r['exact_acc']:7.1%}")

    arc_scale_sweep(learned)
    return


@app.cell
def _(
    ARC_FEATURES,
    EPOCHS,
    LR,
    Weights,
    decode,
    learned,
    test_seqs,
    train,
    train_seqs,
):
    def error_runs(w, label):
        from collections import Counter
        runs = Counter()
        for seq in test_seqs:
            pred = decode(seq.notes, w)
            cur = 0
            for p, g, ok in zip(pred, seq.expert, seq.scored):
                if not ok:
                    continue
                if p.kind == g.kind and p.string == g.string:
                    if cur:
                        runs[cur] += 1
                    cur = 0
                else:
                    cur += 1
            if cur:
                runs[cur] += 1
        n_err = sum(k * v for k, v in runs.items())
        n_run = sum(runs.values())
        longest = max(runs) if runs else 0
        print(f"  {label:22s} errors={n_err:4d} runs={n_run:4d} "
              f"mean_len={n_err / max(1, n_run):4.2f} longest={longest}")

    _node = train(train_seqs, epochs=EPOCHS, lr=LR,
                  init=Weights(), frozen=ARC_FEATURES)
    print("error clustering (test)")
    error_runs(_node, "node only")
    error_runs(learned, "full (arc + bands)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Ablation grid

    Every row retrains from scratch with the named features frozen at
    zero, so the survivors re-fit around the absence — unlike zeroing
    weights after training. Any row below the lookup table at the
    bottom is not earning its use of context.
    """)
    return


@app.cell
def _(
    ARC_FEATURES,
    BAND_FEATURES,
    EPOCHS,
    LR,
    Weights,
    cf,
    evaluate,
    test_seqs,
    train,
    train_seqs,
):
    def ablation_grid():
        configs = [
            ("v4 full (arc + bands)", ()),
            ("no arc (node only)", ARC_FEATURES),
            ("no bands (v3 node set)", BAND_FEATURES),
            ("no arc, no bands", ARC_FEATURES + BAND_FEATURES),
        ]
        print(f"  {'configuration':26s} {'kind':>7} {'string':>7} "
              f"{'exact':>7} {'train':>7}")
        print("  " + "-" * 60)
        out = {}
        for label, frozen in configs:
            w = train(train_seqs, epochs=EPOCHS, lr=LR,
                      init=Weights(), frozen=frozen)
            te = evaluate(test_seqs, w)
            tr = evaluate(train_seqs, w)
            out[label] = w
            print(f"  {label:26s} {te['kind_acc']:6.1%} {te['string_acc']:6.1%} "
                  f"{te['exact_acc']:6.1%} {tr['exact_acc']:6.1%}")
        print("  " + "-" * 60)
        print(f"  {'context-free lookup':26s} {'-':>6} {'-':>6} "
              f"{cf['exact_acc']:6.1%}")
        return out

    ablated = ablation_grid()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Did the per-string biases learn anything musical?

    A global weight ordering that mirrors the usage column is
    memorising base rates; the band crossings exist to give
    register-dependent preference instead. Compare the two columns.
    """)
    return


@app.cell
def _(BAND_FEATURES, learned, seqs):
    def string_usage(sequences, w):
        from collections import Counter
        gold = Counter(c.string for s in sequences
                       for c, ok in zip(s.expert, s.scored) if ok)
        tot = sum(gold.values())
        print(f"  {'string':9s} {'expert':>14s} {'global w':>10s}   "
              f"bands (low -> high hui)")
        for s in range(1, 8):
            bands = " ".join(
                f"{w[k]:+6.2f}" for k in BAND_FEATURES if k.startswith(f"sb_{s}_")
            )
            print(f"  string_{s}   {gold[s]:5d} ({gold[s] / tot:5.1%})  "
                  f"{w[f'string_{s}']:+9.3f}   {bands}")

    string_usage(seqs, learned)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Style control: open/ harmonic 散音/泛音 preference slider

    Negative bias = cheaper = more of that timbre.

    Timbre mix and exact agreement move independently: matching the expert's overall stopped/open/harmonic 按/散/泛 proportions says nothing about matching them note by note.
    """)
    return


@app.cell
def _(mo, seqs):
    piece_pick = mo.ui.dropdown(
        options={s.piece: i for i, s in enumerate(seqs)},
        value=seqs[0].piece, label="piece",
    )
    open_bias = mo.ui.slider(-12.0, 12.0, step=0.5, value=0.0,
                             label="Open 散音 bias (- more open, + fewer)")
    harm_bias = mo.ui.slider(-12.0, 12.0, step=0.5, value=0.0,
                             label="Harmonic 泛音 bias (- more harmonics, + fewer)")
    mo.vstack([piece_pick, open_bias, harm_bias])
    return harm_bias, open_bias, piece_pick


@app.cell
def _(decode, harm_bias, learned, open_bias, piece_pick, seqs):
    import unicodedata
    from collections import Counter as _Counter

    def _dwidth(s: str) -> int:
        # CJK is double-width
        return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)

    def _pad(s: str, n: int) -> str:
        return s + " " * max(0, n - _dwidth(s))

    def _hui_fen(pos: float) -> str:
        hui = int(pos)
        fen = int(round((pos - hui) * 10))
        if fen >= 10:
            hui, fen = hui + 1, 0
        return f"{hui}hui徽" if fen == 0 else f"{hui}hui徽{fen}fen分"

    def _fmt(c) -> str:
        if c.kind == "open":
            return f"散open {c.string}弦string"
        if c.kind == "harmonic":
            return f"泛harmonic {c.string}弦string{_hui_fen(c.position)}"
        return f"按stopped {c.string}弦string{_hui_fen(c.position)}"

    def show(seq, w, limit=None, errors_only=False):
        pred = decode(seq.notes, w)
        pairs = list(zip(pred, seq.expert, seq.scored))

        mix = _Counter(c.kind for c in pred)
        gold = _Counter(c.kind for c in seq.expert)
        n_scored = sum(s for *_, s in pairs)
        agree = sum(p.kind == g.kind and p.string == g.string
                    for p, g, s in pairs if s)

        print(f"{seq.piece} - {len(pred)} notes ({n_scored} scored)")
        print(f"  predicted:  按stopped {mix.get('stopped', 0):3d} | "
              f"散open {mix.get('open', 0):3d} | 泛harmonic {mix.get('harmonic', 0):3d}")
        print(f"  expert:     按stopped {gold.get('stopped', 0):3d} | "
              f"散open {gold.get('open', 0):3d} | 泛harmonic {gold.get('harmonic', 0):3d}")
        print(f"  exact agreement (scored only): {agree}/{n_scored} "
              f"= {agree / max(1, n_scored):.1%}")
        print("  · = not scored\n")

        print(f"  {'#':>3} {'pitch':>6}  {_pad('predicted', 16)} "
              f"{_pad('expert', 16)}")
        print("  " + "-" * 48)
        for i, (p, g, s) in enumerate(pairs):
            if not s:
                mark = "·"
            elif p.kind == g.kind and p.string == g.string:
                mark = "✓"
            else:
                mark = "✗"
            if errors_only and mark != "✗":
                continue
            if limit is not None and i >= limit:
                break
            print(f"  {i:3d} {seq.notes[i].semitones:6.0f}  "
                  f"{_pad(_fmt(p), 16)} {_pad(_fmt(g), 16)} {mark}")

    show(seqs[piece_pick.value],
         learned.biased(open_bias.value, harm_bias.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Which pitches does the model get wrong?

    `variants` and `majority` measure that pitch's ambiguity. Errors on
    low-majority pitches are largely inherent to the task; errors on
    high-majority pitches are model failures worth chasing.
    """)
    return


@app.cell
def _(decode, learned, test_seqs):
    def error_by_pitch(sequences, w):
        from collections import Counter
        tot, bad, gold_mix = Counter(), Counter(), {}
        for seq in sequences:
            pred = decode(seq.notes, w)
            for note, p, g, ok in zip(seq.notes, pred, seq.expert, seq.scored):
                if not ok:
                    continue
                key = round(note.semitones, 1)
                tot[key] += 1
                gold_mix.setdefault(key, Counter())[(g.string, g.kind)] += 1
                if not (p.kind == g.kind and p.string == g.string):
                    bad[key] += 1
        print(f"  {'pitch':>6} {'n':>5} {'wrong':>6} {'err':>7} "
              f"{'variants':>9} {'majority':>9}")
        for key in sorted(tot):
            c = gold_mix[key]
            maj = c.most_common(1)[0][1] / sum(c.values())
            print(f"  {key:6.0f} {tot[key]:5d} {bad[key]:6d} "
                  f"{bad[key] / tot[key]:7.1%} {len(c):9d} {maj:9.1%}")

    error_by_pitch(test_seqs, learned)
    return


if __name__ == "__main__":
    app.run()
