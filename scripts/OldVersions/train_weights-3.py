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
    # Phase 3: path-difference weight learning + style control

    Trains the decoder's cost weights from GQ39 expert paths with an
    averaged structured perceptron (`qinpos.learn`), on FULL
    sequences — 散音/泛音 included, no timbre constraint — so the
    model learns when experts choose those timbres, not only which
    string. Split is BY PIECE (every 4th piece held out), never by
    note, to avoid same-piece leakage.

    Read the numbers in this order:

    1. `check_idx` — is melodic order preserved by the idx sort?
    2. `reach_report` — can the lattice express the expert at all?
    3. context-free baseline — what does a pitch lookup table get?
    4. learned model — does the sequence model beat that table?

    Step 4 is the whole point. A sequence model that loses to a
    context-free lookup is not using context, and the arc features
    need re-examining before anything else is tuned.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    from qinpos.learn import (build_sequences, check_idx, context_free_ceiling,
                              context_free_eval, context_free_table, evaluate,
                              reach_report, train)
    from qinpos.viterbi import FEATURES, Weights, decode

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data/GQ39/score_annotation"
    CLEAN_CSV = ROOT / "data/gq39_clean.csv"
    WEIGHTS_JSON = ROOT / "data/learned_weights.json"
    return (
        CLEAN_CSV,
        DATA,
        FEATURES,
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

    On scored notes this is near-tautological (same tuning theory on
    both sides). The informative run is `scored_only=False`, where
    unreachable notes are an independent check on `needs_review`.
    """)
    return


@app.cell
def _(reach_report, seqs):
    reach_report(seqs)
    return


@app.cell
def _(reach_report, seqs):
    reach_report(seqs, scored_only=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Context-free baselines

    **Honest baseline**: fit the pitch table on training pieces, score on held-out pieces. This is the number the sequence model has to beat, and it belongs in the dissertation's results table.

    **Oracle ceiling**: fit and score on the same notes. Leaky, so not a baseline. It bounds what any pitch-only mapping could reach, and
    the gap from it to 100% is the share of the task that provably needs context.
    """)
    return


@app.cell
def _(context_free_eval, context_free_table, test_seqs, train_seqs):
    cf_table = context_free_table(train_seqs)
    _cf = context_free_eval(cf_table, test_seqs)
    print("context-free lookup (fit on train, scored on test)")
    print(f"  exact={_cf['exact_acc']:.1%}  (n={_cf['n']}, "
          f"{_cf['unseen_pitch']} pitches unseen in training)")
    return


@app.cell
def _(context_free_ceiling, seqs):
    context_free_ceiling(seqs)
    return


@app.cell
def _(context_free_ceiling, test_seqs):
    print("per-piece oracle ceiling (test pieces)")
    context_free_ceiling(test_seqs, per_piece=True, detail=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Hand-crafted weights (Path A baseline)

    `Weights()` defaults from `viterbi.py`, set by domain knowledge and
    nudged by the Phase 2 coarse grid. Free timbre choice.
    """)
    return


@app.cell
def _(Weights, evaluate, test_seqs):
    _b = evaluate(test_seqs, Weights())
    print("hand-crafted (test, free choice)")
    print(f"  kind={_b['kind_acc']:.1%}  string={_b['string_acc']:.1%}  "
          f"exact={_b['exact_acc']:.1%}  (n={_b['n']})")
    print(f"  reach_rate={_b['reach_rate']:.1%}  "
          f"exact|reach={_b['exact_given_reach']:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Train (averaged structured perceptron)
    """)
    return


@app.cell
def _(FEATURES, WEIGHTS_JSON, Weights, json, train, train_seqs):
    learned = train(train_seqs, epochs=40, lr=0.05, init=Weights())
    WEIGHTS_JSON.write_text(json.dumps(dict(learned), indent=2))
    print(f"saved {WEIGHTS_JSON}")
    for _k in FEATURES:
        print(f"  {_k:14s} {learned[_k]:+.3f}")
    return (learned,)


@app.cell
def _(evaluate, learned, test_seqs, train_seqs):
    _free = evaluate(test_seqs, learned)
    _kg = evaluate(test_seqs, learned, kinds_known=True)
    _tr = evaluate(train_seqs, learned)
    print(f"LEARNED (test, free):   kind={_free['kind_acc']:.1%} "
          f"string={_free['string_acc']:.1%} exact={_free['exact_acc']:.1%}")
    print(f"LEARNED (test, timbre given): string={_kg['string_acc']:.1%} "
          f"(different protocol, not comparable to the line above)")
    print(f"LEARNED (train, free):  exact={_tr['exact_acc']:.1%} "
          f"(below test = underfitting, above test = overfitting)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Where the weights came from

    Per-string bias vs how often experts actually use each string. If
    the weight is just the mirror image of the frequency, the feature
    is memorising base rates rather than anything musical.
    """)
    return


@app.cell
def _(learned, seqs):
    def string_usage(sequences, w):
        from collections import Counter
        gold = Counter(c.string for s in sequences
                       for c, ok in zip(s.expert, s.scored) if ok)
        tot = sum(gold.values())
        for s in range(1, 8):
            print(f"  string_{s}: expert {gold[s]:5d} ({gold[s] / tot:5.1%})  "
                  f"weight {w[f'string_{s}']:+.3f}")

    string_usage(seqs, learned)
    return


@app.cell
def _(evaluate, learned, test_seqs):
    _a = evaluate(test_seqs, learned.without_string_bias())
    print("post-hoc ablation: per-string biases zeroed")
    print(f"  kind={_a['kind_acc']:.1%}  string={_a['string_acc']:.1%}  "
          f"exact={_a['exact_acc']:.1%}")
    print("  (upper bound on their contribution - a true ablation "
          "removes them from FEATURES and retrains)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Style control: 散音/泛音 preference slider

    Negative bias = cheaper = MORE of that timbre in the output;
    positive = fewer. Slider ranges are set wider than the learned
    weights so the bias can actually overcome them.

    Note the timbre mix and the exact agreement move independently:
    matching the expert's overall 按/散/泛 proportions says nothing
    about matching them note by note.
    """)
    return


@app.cell
def _(mo, seqs):
    piece_pick = mo.ui.dropdown(
        options={s.piece: i for i, s in enumerate(seqs)},
        value=seqs[0].piece, label="piece",
    )
    open_bias = mo.ui.slider(-12.0, 12.0, step=0.5, value=0.0,
                             label="Open 散音 bias (− more open, + fewer)")
    harm_bias = mo.ui.slider(-12.0, 12.0, step=0.5, value=0.0,
                             label="Harmonic 泛音 bias (− more harmonics, + fewer)")
    mo.vstack([piece_pick, open_bias, harm_bias])
    return harm_bias, open_bias, piece_pick


@app.cell
def _(decode, harm_bias, learned, open_bias, piece_pick, seqs):
    import unicodedata
    from collections import Counter as _Counter

    def _dwidth(s: str) -> int:
        # CJK glyphs are double-width in monospace fonts
        return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)

    def _pad(s: str, n: int) -> str:
        return s + " " * max(0, n - _dwidth(s))

    def _hui_fen(pos: float) -> str:
        hui = int(pos)
        fen = int(round((pos - hui) * 10))
        if fen >= 10:
            hui, fen = hui + 1, 0
        return f"{hui}徽" if fen == 0 else f"{hui}徽{fen}分"

    def _fmt(c) -> str:
        if c.kind == "open":
            return f"散 {c.string}弦"
        if c.kind == "harmonic":
            return f"泛 {c.string}弦{_hui_fen(c.position)}"
        return f"按 {c.string}弦{_hui_fen(c.position)}"

    def show(seq, w, limit=None, errors_only=False):
        pred = decode(seq.notes, w)
        pairs = list(zip(pred, seq.expert, seq.scored))

        mix = _Counter(c.kind for c in pred)
        gold = _Counter(c.kind for c in seq.expert)
        n_scored = sum(s for *_, s in pairs)
        agree = sum(p.kind == g.kind and p.string == g.string
                    for p, g, s in pairs if s)

        print(f"{seq.piece} — {len(pred)} notes ({n_scored} scored)")
        print(f"  predicted:  按 {mix.get('stopped', 0):3d} | "
              f"散 {mix.get('open', 0):3d} | 泛 {mix.get('harmonic', 0):3d}")
        print(f"  expert:     按 {gold.get('stopped', 0):3d} | "
              f"散 {gold.get('open', 0):3d} | 泛 {gold.get('harmonic', 0):3d}")
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
    ## 8. Which pitches does the model get wrong?

    Groups errors by pitch so they can be lined up against the
    per-pitch `majority` figures in section 3. High pitches have more
    realisations and lower majority — expect the errors to cluster
    there.
    """)
    return


@app.cell
def _(decode, learned, test_seqs):
    def error_by_pitch(sequences, w):
        from collections import Counter
        tot, bad = Counter(), Counter()
        for seq in sequences:
            pred = decode(seq.notes, w)
            for note, p, g, ok in zip(seq.notes, pred, seq.expert, seq.scored):
                if not ok:
                    continue
                key = round(note.semitones, 1)
                tot[key] += 1
                if not (p.kind == g.kind and p.string == g.string):
                    bad[key] += 1
        print(f"  {'pitch':>6} {'n':>5} {'wrong':>6} {'err':>7}")
        for key in sorted(tot):
            print(f"  {key:6.1f} {tot[key]:5d} {bad[key]:6d} "
                  f"{bad[key] / tot[key]:7.1%}")

    error_by_pitch(test_seqs, learned)
    return


@app.cell
def _(evaluate, learned, test_seqs):
    _w = learned.copy()
    for _k in ("string_cross", "hand_travel", "reposition"):
        _w[_k] = 0.0
    _a = evaluate(test_seqs, _w)
    print("post-hoc ablation: arc (transition) weights zeroed")
    print(f"  kind={_a['kind_acc']:.1%}  string={_a['string_acc']:.1%}  "
          f"exact={_a['exact_acc']:.1%}")
    return


@app.cell
def _(Weights, evaluate, test_seqs, train, train_seqs):
    ARC = ("string_cross", "hand_travel", "reposition")
    node_only = train(train_seqs, epochs=40, lr=0.05, init=Weights(), frozen=ARC)
    _a = evaluate(test_seqs, node_only)
    print("TRUE ablation: retrained with arc features removed")
    print(f"  kind={_a['kind_acc']:.1%}  string={_a['string_acc']:.1%}  "
          f"exact={_a['exact_acc']:.1%}")
    return


if __name__ == "__main__":
    app.run()
