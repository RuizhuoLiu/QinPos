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

    The last section is the interactive style control: a bias on
    the learned `is_open` / `is_harmonic` weights shifts how often
    the decoder chooses 散/泛. Whether "fewer 散/泛" maps to
    "easier" is a musical judgement to confirm with a performer —
    the slider itself is neutral machinery, and the live timbre
    counts let you see exactly what each setting does.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    from qinpos.learn import (WeightVector, build_sequences, evaluate,
                              train)
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
        decode,
        evaluate,
        json,
        train,
    )


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


@app.cell
def _(seqs):
    from qinpos.learn import reach_report
    reach_report(seqs)
    return (reach_report,)


@app.cell
def _(reach_report, seqs):
    reach_report(seqs, scored_only=False)
    return


@app.cell
def _():
    from qinpos.theory import HARMONIC_SEMITONES_AT_HUI
    print(sorted(HARMONIC_SEMITONES_AT_HUI.items()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline: hand-crafted weights, free timbre choice
    """)
    return


@app.cell
def _(Weights, evaluate, test_seqs):
    _b = evaluate(test_seqs, Weights())
    print(f"hand-crafted (test, free choice)")
    print(f"  kind={_b['kind_acc']:.1%}  string={_b['string_acc']:.1%}  " # kind - stop, open, harmonic
          f"exact={_b['exact_acc']:.1%}  (n={_b['n']})") # extact - kind + string both correct
    print(f"  reach_rate={_b['reach_rate']:.1%}  "
          f"exact|reach={_b['exact_given_reach']:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train (averaged structured perceptron)
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
def _(evaluate, learned, test_seqs):
    _w = learned.copy()
    for _s in range(1, 8):
        _w[f"string_{_s}"] = 0.0
    _a = evaluate(test_seqs, _w)
    print(f"no string bias: kind={_a['kind_acc']:.1%} "
          f"string={_a['string_acc']:.1%} exact={_a['exact_acc']:.1%}")
    return


@app.cell
def _(DATA):
    from collections import defaultdict
    from qinpos.dataset_gq39 import load_all

    def check_idx(data_dir, limit=12):
        spans = defaultdict(list)
        for e in load_all(data_dir):
            spans[(e.piece, e.section)].append(e.idx)
        for (p, sec), idxs in sorted(spans.items())[:limit]:
            print(f"{p:12s} {str(sec):24s} n={len(idxs):4d}  "
                  f"idx {min(idxs)}..{max(idxs)}")

    check_idx(DATA)
    return


@app.cell
def _(seqs):
    from collections import Counter as _Counter, defaultdict as _defaultdict

    def context_free_ceiling(sequences, per_piece=False):
        """Best possible accuracy for any deterministic pitch -> fingering map."""
        def tally(seqs_):
            by_pitch = _defaultdict(_Counter)
            for seq in seqs_:
                for note, g, ok in zip(seq.notes, seq.expert, seq.scored):
                    if ok:
                        by_pitch[round(note.semitones, 1)][(g.string, g.kind)] += 1
            n = sum(sum(c.values()) for c in by_pitch.values())
            m = sum(c.most_common(1)[0][1] for c in by_pitch.values())
            return m, n, by_pitch

        if per_piece:
            for seq in sequences:
                m, n, _ = tally([seq])
                if n:
                    print(f"{seq.piece:14s} ceiling {m}/{n} = {m/n:6.1%}")
        m, n, by_pitch = tally(sequences)
        print(f"\nCONTEXT-FREE CEILING: {m}/{n} = {m/n:.1%}")
        for pitch, c in sorted(by_pitch.items()):
            tot = sum(c.values())
            print(f"  pitch {pitch:5.1f}  n={tot:5d}  {len(c)} realisations  "
                  f"majority {c.most_common(1)[0][1]/tot:5.1%}")

    context_free_ceiling(seqs)
    return


@app.cell
def _(learned, seqs):
    from collections import Counter
    _gold = Counter(c.string for s in seqs for c, ok in zip(s.expert, s.scored) if ok)
    _tot = sum(_gold.values())
    for _s in range(1, 8):
        print(f"string_{_s}: expert {_gold[_s]:5d} ({_gold[_s]/_tot:5.1%})  "
              f"weight {learned[f'string_{_s}']:+.3f}")
    return


@app.cell
def _(evaluate, learned, test_seqs, train_seqs):
    _free = evaluate(test_seqs, learned)
    _kg = evaluate(test_seqs, learned, kinds_known=True)
    _tr = evaluate(train_seqs, learned)
    print(f"LEARNED (test, free):   kind={_free['kind_acc']:.1%} "
          f"string={_free['string_acc']:.1%} exact={_free['exact_acc']:.1%}")
    print(f"  ceiling={_free['reach_rate']:.1%}  "
          f"exact|reach={_free['exact_given_reach']:.1%}")
    print(f"LEARNED (test, timbre given): string={_kg['string_acc']:.1%}")
    print(f"LEARNED (train, free):  exact={_tr['exact_acc']:.1%} "
          f"(overfit check)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Style control: 散音/泛音 preference slider

    Negative bias = cheaper = MORE of that timbre in the output;
    positive = fewer. Pick a piece, drag, and watch the timbre mix
    and the fingering change. (Direction-to-difficulty mapping is
    yours to decide with your performer.)
    """)
    return


@app.cell
def _(mo, seqs):
    piece_pick = mo.ui.dropdown(
        options={s.piece: i for i, s in enumerate(seqs)},
        value=seqs[0].piece, label="piece",
    )
    open_bias = mo.ui.slider(-6.0, 6.0, step=0.5, value=0.0,
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
        print(f"  · = not scored\n")

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
            print(f"  {i:3d} {seq.notes[i].semitones:6.1f}  "
                  f"{_pad(_fmt(p), 16)} {_pad(_fmt(g), 16)} {mark}")

    show(seqs[piece_pick.value],
         learned.biased(open_bias.value, harm_bias.value))
    return


@app.cell
def _(evaluate, learned, test_seqs):
    # Remove the string bias and see how much it actually contributes
    w_nostring = learned.copy()
    for s in range(1, 8):
        w_nostring[f"string_{s}"] = 0.0
    print(evaluate(test_seqs, w_nostring))
    return


if __name__ == "__main__":
    app.run()
