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
    # How much context worth

    Pitch-only mapping cannot exceed 54.7%,
    node-only model - 48.5% of it.

    Every target above
    ~55% therefore has to come from context.
    """)
    return


@app.cell
def _():
    from collections import Counter, defaultdict
    from pathlib import Path

    from qinpos.learn import build_sequences

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data/GQ39/score_annotation"
    CLEAN = ROOT / "data/gq39_clean.csv"

    MIN_COUNT = 2  # 1: a single accident may override a backoff level
    return CLEAN, Counter, DATA, MIN_COUNT, build_sequences, defaultdict


@app.cell
def _(CLEAN, DATA, build_sequences):
    seqs = build_sequences(DATA, CLEAN)
    test_seqs = [s for i, s in enumerate(seqs) if i % 4 == 0]
    train_seqs = [s for i, s in enumerate(seqs) if i % 4 != 0]
    print(f"train {len(train_seqs)} / test {len(test_seqs)} pieces")
    print("test:", [s.piece for s in test_seqs])
    return seqs, test_seqs, train_seqs


@app.cell
def _():
    START = "<START>"
    END = "<END>"

    def realisation(c):
        return (c.string, c.kind)  # position follows from these plus pitch

    def pitch(seq, i):
        return round(seq.notes[i].semitones, 1)

    def prev_pitch(seq, i):
        return pitch(seq, i - 1) if i > 0 else START

    def next_pitch(seq, i):
        return pitch(seq, i + 1) if i + 1 < len(seq.notes) else END

    def prev_real(seq, i):
        return realisation(seq.expert[i - 1]) if i > 0 else START

    def prev_kind(seq, i):
        return seq.expert[i - 1].kind if i > 0 else START

    return next_pitch, pitch, prev_kind, prev_pitch, prev_real, realisation


@app.cell
def _(next_pitch, pitch, prev_kind, prev_pitch, prev_real):
    # observable - from the input melody alone; oracle - uses a neighbour for the chain model
    CONTEXTS = {
        "pitch only": (lambda s, i: (pitch(s, i),), "observable"),
        "+ prev pitch": (lambda s, i: (prev_pitch(s, i), pitch(s, i)), "observable"),
        "+ next pitch": (lambda s, i: (pitch(s, i), next_pitch(s, i)), "observable"),
        "+ both pitches": (
            lambda s, i: (prev_pitch(s, i), pitch(s, i), next_pitch(s, i)),
            "observable",
        ),
        "+ prev kind": (lambda s, i: (prev_kind(s, i), pitch(s, i)), "oracle"),
        "+ prev realisation": (lambda s, i: (prev_real(s, i), pitch(s, i)), "oracle"),
        "+ prev real, next pitch": (
            lambda s, i: (prev_real(s, i), pitch(s, i), next_pitch(s, i)),
            "oracle",
        ),
    }
    return (CONTEXTS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Oracle ceilings

    Bucket the test notes by context key, count how many the bucket's majority realisation covers — `context_free_ceiling()` generalised past pitch-only.

    `singleton` is the proportion of notes alone in their bucket, correct by construction; high - the ceiling is measuring memorisation rather than information.
    """)
    return


@app.cell
def _(CONTEXTS, Counter, defaultdict, realisation, test_seqs):
    def oracle_ceiling(sequences, key_fn):
        buckets = defaultdict(Counter)
        for seq in sequences:
            for i in range(len(seq.notes)):
                if seq.scored[i]:
                    buckets[key_fn(seq, i)][realisation(seq.expert[i])] += 1
        n = sum(sum(c.values()) for c in buckets.values())
        hit = sum(c.most_common(1)[0][1] for c in buckets.values())
        singles = sum(1 for c in buckets.values() if sum(c.values()) == 1)
        return {
            "n": n,
            "acc": hit / max(1, n),
            "buckets": len(buckets),
            "singleton": singles / max(1, n),
        }

    def _report():
        print(f"  {'context':26s} {'kind':11s} {'ceiling':>8} "
              f"{'buckets':>8} {'singleton':>10}")
        print("  " + "-" * 66)
        for name, (fn, kind) in CONTEXTS.items():
            r = oracle_ceiling(test_seqs, fn)
            print(f"  {name:26s} {kind:11s} {r['acc']:7.1%} "
                  f"{r['buckets']:8d} {r['singleton']:9.1%}")
        print(f"\n  (n = {oracle_ceiling(test_seqs, CONTEXTS['pitch only'][0])['n']} "
              f"scored test notes)")

    _report()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Honest backoff

    Fit on train, predict each test note from the most specific context whose training bucket has at least `MIN_COUNT` observations, else back off. This is an n-gram classifier with backoff — a real sequence model, and the number the CRF has to beat.

    `used` shows which level fired, i.e. how often the richer context was available on held-out data.
    """)
    return


@app.cell
def _(
    Counter,
    MIN_COUNT,
    defaultdict,
    next_pitch,
    pitch,
    prev_pitch,
    realisation,
):
    def fit_table(sequences, key_fn):
        buckets = defaultdict(Counter)
        for seq in sequences:
            for i in range(len(seq.notes)):
                if seq.scored[i]:
                    buckets[key_fn(seq, i)][realisation(seq.expert[i])] += 1
        return {
            k: c.most_common(1)[0][0]
            for k, c in buckets.items()
            if sum(c.values()) >= MIN_COUNT
        }

    def global_majority(sequences):
        c = Counter()
        for seq in sequences:
            for i in range(len(seq.notes)):
                if seq.scored[i]:
                    c[realisation(seq.expert[i])] += 1
        return c.most_common(1)[0][0]

    def backoff_eval(train_sequences, test_sequences, levels):
        """levels: [(name, key_fn), ...] most specific first."""
        tables = [(name, fn, fit_table(train_sequences, fn)) for name, fn in levels]
        fallback = global_majority(train_sequences)
        n = hit = 0
        used, used_hit = Counter(), Counter()
        for seq in test_sequences:
            for i in range(len(seq.notes)):
                if not seq.scored[i]:
                    continue
                n += 1
                choice, level = fallback, "global"
                for name, fn, table in tables:
                    got = table.get(fn(seq, i))
                    if got is not None:
                        choice, level = got, name
                        break
                used[level] += 1
                if choice == realisation(seq.expert[i]):
                    hit += 1
                    used_hit[level] += 1
        return {"n": n, "acc": hit / max(1, n), "used": used, "used_hit": used_hit}

    LADDERS = {
        "pitch only": [("pitch", lambda s, i: (pitch(s, i),))],
        "prev pitch -> pitch": [
            ("prev+pitch", lambda s, i: (prev_pitch(s, i), pitch(s, i))),
            ("pitch", lambda s, i: (pitch(s, i),)),
        ],
        "next pitch -> pitch": [
            ("pitch+next", lambda s, i: (pitch(s, i), next_pitch(s, i))),
            ("pitch", lambda s, i: (pitch(s, i),)),
        ],
        # v1 ladder - backed off through prev+pitch. Retrospective pitch context carries almost no generalisable signal (oracle 69.1% -> honest 45.7%), so it is kept only as the labelled control and dropped from the working ladder.
        "both -> prev -> pitch (v1)": [
            ("both", lambda s, i: (prev_pitch(s, i), pitch(s, i), next_pitch(s, i))),
            ("prev+pitch", lambda s, i: (prev_pitch(s, i), pitch(s, i))),
            ("pitch", lambda s, i: (pitch(s, i),)),
        ],
        "both -> next -> pitch": [
            ("both", lambda s, i: (prev_pitch(s, i), pitch(s, i), next_pitch(s, i))),
            ("pitch+next", lambda s, i: (pitch(s, i), next_pitch(s, i))),
            ("pitch", lambda s, i: (pitch(s, i),)),
        ],
    }
    return LADDERS, backoff_eval


@app.cell
def _(LADDERS, backoff_eval, test_seqs, train_seqs):
    def _report():
        print(f"  {'backoff ladder':24s} {'test acc':>9}   level usage (acc within level)")
        print("  " + "-" * 74)
        for name, levels in LADDERS.items():
            r = backoff_eval(train_seqs, test_seqs, levels)
            parts = []
            for lvl, cnt in r["used"].most_common():
                acc = r["used_hit"][lvl] / cnt
                parts.append(f"{lvl} {cnt} ({acc:.0%})")
            print(f"  {name:24s} {r['acc']:8.1%}   " + "  ".join(parts))
        print("\n  reference: v4 node-only model 48.5%, oracle pitch ceiling 54.7%")

    _report()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2b. Top-k

    Exact matching requires that the performer precisely reproduce the specific way of realization. However, in tasks with actual ambiguity, this is unfair: for pitch 14, there are five different expert realizations, and indicating a different valid realization has the same score as indicating an unplayable realization.
    Top-k asks whether the expert's choice is in the shortlist — the honest framing for a system that outputs suggestions.

    `coverage` is the share of notes whose bucket held at least k realisations; below that, top-k saturates for structural reasons.
    """)
    return


@app.cell
def _(
    Counter,
    MIN_COUNT,
    defaultdict,
    next_pitch,
    pitch,
    realisation,
    test_seqs,
    train_seqs,
):
    def fit_ranked(sequences, key_fn):
        buckets = defaultdict(Counter)
        for seq in sequences:
            for i in range(len(seq.notes)):
                if seq.scored[i]:
                    buckets[key_fn(seq, i)][realisation(seq.expert[i])] += 1
        return {
            k: [r for r, _ in c.most_common()]
            for k, c in buckets.items()
            if sum(c.values()) >= MIN_COUNT
        }

    def topk_backoff(train_sequences, test_sequences, levels, ks=(1, 2, 3, 5)):
        tables = [(fn, fit_ranked(train_sequences, fn)) for _, fn in levels]
        n = 0
        hits = Counter()
        covered = Counter()
        for seq in test_sequences:
            for i in range(len(seq.notes)):
                if not seq.scored[i]:
                    continue
                n += 1
                ranked = []
                for fn, table in tables:
                    got = table.get(fn(seq, i))
                    if got:
                        ranked = got
                        break
                truth = realisation(seq.expert[i])
                for k in ks:
                    if len(ranked) >= k:
                        covered[k] += 1
                    if truth in ranked[:k]:
                        hits[k] += 1
        return n, hits, covered, ks

    def _report():
        levels = [
            ("pitch+next", lambda s, i: (pitch(s, i), next_pitch(s, i))),
            ("pitch", lambda s, i: (pitch(s, i),)),
        ]
        n, hits, covered, ks = topk_backoff(train_seqs, test_seqs, levels)
        print(f"  next-pitch backoff, n = {n} scored test notes")
        print(f"    {'k':>3} {'top-k acc':>10} {'coverage':>10}")
        for k in ks:
            print(f"    {k:3d} {hits[k] / max(1, n):9.1%} {covered[k] / max(1, n):10.1%}")
        print("\n  A trained model shares statistics across pitches through")
        print("  features instead of memorising one bucket per context, so")
        print("  these are a FLOOR for what the CRF should reach, not a target.")

    _report()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 泛音段 Harmonic section structure

    Testing whether harmonics are sectional (泛起 … 泛止) rather than isolated, which in a chain model means P(harmonic | prev harmonic)
    far above P(harmonic).

    A flat `is_harmonic` node cost cannot make the second harmonic of a passage cheaper than the first; `harm_run`/`harm_enter`/`harm_exit`
    can. 散音 is the negative control.
    """)
    return


@app.cell
def _(Counter, test_seqs, train_seqs):
    def run_structure(sequences, kind, label):
        runs, cur = [], 0
        n = n_kind = n_kind_after_kind = 0
        for seq in sequences:
            for i in range(len(seq.notes)):
                if not seq.scored[i]:
                    continue
                is_k = seq.expert[i].kind == kind
                n += 1
                n_kind += is_k
                if is_k:
                    cur += 1
                elif cur:
                    runs.append(cur)
                    cur = 0
                if i > 0 and seq.scored[i - 1]:

                    if seq.expert[i - 1].kind == kind:
                        n_kind_after_kind += (1 if is_k else 0)
            if cur:
                runs.append(cur)
                cur = 0
        prior = n_kind / max(1, n)
        n_prev = sum(
            1
            for seq in sequences
            for i in range(1, len(seq.notes))
            if seq.scored[i] and seq.scored[i - 1] and seq.expert[i - 1].kind == kind
        )
        cond = n_kind_after_kind / max(1, n_prev)
        print(f"  {label}")
        print(f"    P({kind})              = {prior:.3f}   ({n_kind}/{n} notes)")
        print(f"    P({kind} | prev {kind}) = {cond:.3f}   lift x{cond / max(1e-9, prior):.2f}")
        if runs:
            hist = Counter(runs)
            print(f"    {len(runs)} runs, mean length {sum(runs) / len(runs):.2f}, "
                  f"longest {max(runs)}")
            print("    run lengths: " + "  ".join(
                f"{ln}x{hist[ln]}" for ln in sorted(hist)))
        print()

    run_structure(train_seqs, "harmonic", "harmonic 泛音 (train pieces)")
    run_structure(test_seqs, "harmonic", "harmonic 泛音 (test pieces)")
    run_structure(train_seqs, "open", "open 散音 (train pieces)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Repeated pitches

    Error analysis on ciou01 found the expert alternating timbre on
    repeated pitches (散7弦 → 按5弦10徽 → 散7弦, open 7 → string 5 hui 10 → open 7, five times in one piece) while the model held one fingering.

    A context-free model maps pitch to a single realisation, so it always repeats; the expert's repeat rate is therefore a hard bound
    on its accuracy at those notes. `repeat_identical` captures this.
    """)
    return


@app.cell
def _(Counter, realisation, seqs, test_seqs, train_seqs):
    def repeat_behaviour(sequences, label):
        tally = Counter()
        for seq in sequences:
            for i in range(1, len(seq.notes)):
                if not (seq.scored[i] and seq.scored[i - 1]):
                    continue
                if round(seq.notes[i].semitones, 1) != round(seq.notes[i - 1].semitones, 1):
                    continue
                a, b = seq.expert[i - 1], seq.expert[i]
                if realisation(a) == realisation(b):
                    tally["identical"] += 1
                elif a.kind != b.kind:
                    tally["kind changed"] += 1
                elif a.string != b.string:
                    tally["string changed"] += 1
                else:
                    tally["position only"] += 1
        total = sum(tally.values())
        print(f"  {label}: {total} adjacent same-pitch pairs")
        if not total:
            print()
            return
        for k, v in tally.most_common():
            print(f"    {k:16s} {v:5d}  {v / total:6.1%}")
        print(f"    -> a context-free model scores at most {tally['identical'] / total:.1%} "
              f"on the SECOND note of each pair")
        print()

    repeat_behaviour(train_seqs, "train pieces")
    repeat_behaviour(test_seqs, "test pieces")
    repeat_behaviour(seqs, "all pieces")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Left-hand geometry

    A linear slope on |Δposition| can only say "further is worse", and
    misses two shapes: the 走手音 range (a technique that left-hand movement with a range of 1-3 hui can be decorative expressions, where 上/下/绰/注/吟/猱 live) may be *preferred* over staying put, and crossing to an adjacent string at the same hui barely moves the finger at all.

    Answered by the `travel_*` buckets and `same_hui_cross`. Read the two tables against each other — they are the across string when play stopped, and hui across when play harmonic "按音走弦、泛音走徽"
    contrast.
    """)
    return


@app.cell
def _(Counter, seqs):
    def geometry(sequences, kinds=("stopped",)):
        POS_BINS = ((0.0, 0.3, "0"), (0.3, 3.5, "1-3"), (3.5, 7.5, "4-7"), (7.5, 99.0, "8+"))

        def pos_bin(d):
            for lo, hi, name in POS_BINS:
                if lo <= d < hi:
                    return name
            return "8+"

        tally = Counter()
        for seq in sequences:
            for i in range(1, len(seq.notes)):
                if not (seq.scored[i] and seq.scored[i - 1]):
                    continue
                a, b = seq.expert[i - 1], seq.expert[i]
                if a.kind not in kinds or b.kind not in kinds:
                    continue
                ds = min(abs(a.string - b.string), 3)
                tally[(ds, pos_bin(abs(a.position - b.position)))] += 1

        total = sum(tally.values())
        names = [n for _, _, n in POS_BINS]
        print(f"  consecutive {'/'.join(kinds)} pairs: {total}")
        print(f"    {'|Δstring|':>10} " + "".join(f"{n:>9}" for n in names))
        for ds in range(4):
            row = [tally[(ds, n)] for n in names]
            if not sum(row):
                continue
            label = "3+" if ds == 3 else str(ds)
            print(f"    {label:>10} " + "".join(f"{v:9d}" for v in row)
                  + f"   ({sum(row) / max(1, total):5.1%})")
        same_hui_cross = sum(tally[(ds, "0")] for ds in (1, 2, 3))
        print(f"    same hui, different string: {same_hui_cross} "
              f"({same_hui_cross / max(1, total):.1%})")
        print(f"    stayed put (Δstring 0, Δpos 0): {tally[(0, '0')]} "
              f"({tally[(0, '0')] / max(1, total):.1%})")
        print(f"    short slide (Δstring 0, 1-3 hui): {tally[(0, '1-3')]} "
              f"({tally[(0, '1-3')] / max(1, total):.1%})")
        print()

    geometry(seqs, kinds=("stopped",))
    geometry(seqs, kinds=("harmonic",))
    return


if __name__ == "__main__":
    app.run()
