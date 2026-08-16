# QinPos

Automatic **string and hui-position assignment for the guqin (古琴)**: give it a
melody in jianpu (numbered notation), get back a playable fingering — which
string, which hui position, and which timbre (散音 open / 按音 stopped / 泛音
harmonic) for every note.

MSc Artificial Intelligence for Media, National Centre for Computer Animation,
Bournemouth University.

## Motivation

Guqin music is written in **jianzipu (減字譜)**, a tablature that records
*actions* — press the third string at the seventh hui — but not pitches.
Jianpu records *pitches* but not actions. Turning one into the other is the
first stage of 打谱 (dapu), the traditional practice of realising a piece from
notation, and it is done by hand by experienced players.

The choice is not obvious: a single pitch can usually be produced on several
strings, in three different timbres, so a short phrase has thousands of
possible fingerings. Which one an expert picks depends on hand economy, timbre
convention, and what comes next.

QinPos automates the pitch-to-position half of that decision. The hui position
itself is **physically determined** once a string is chosen (string-length
ratios), so the real problem is a structured choice over strings and timbres —
solved here with a candidate lattice and a linear-chain CRF.

**Out of scope:** left- and right-hand technique (吟猱綽注, 挑抹勾剔 …) is not
predicted. The tablature output leaves those slots empty.

## What's here

```
src/qinpos/       library code (physics, dataset loader, features, decoder, CRF)
scripts/          marimo notebooks: data cleaning, training, evaluation, fingerboard
streamlit_app.py  the interactive tool
data/             GQ39 clone + generated files (not committed)
```

## Setup

```bash
uv venv && uv pip install -e .
git clone --depth 1 https://github.com/yufenhuang/Guqin-dataset.git data/GQ39
uv run marimo edit scripts/clean_gq39.py      # build data/gq39_clean.csv
uv run marimo edit scripts/train_crf.py       # train, writes data/crf_weights.json
uv run streamlit run streamlit_app.py
```

## Data

- **GQ39** — Guqin dataset of 39 annotated pieces, used for training and
  evaluation: https://github.com/yufenhuang/Guqin-dataset
  Huang, Y., Liang, Z., Wei, C. and Su, L. (2020) 'A dataset and a
  transfer-learning approach for guqin transcription', *ISMIR*.

## Third-party work used

- **JianZiPu font / glyphs** — jianzipu character rendering:
  https://github.com/neuralfirings/JianZiPu (SIL Open Font License)
- **jianzipu (alephpi)** — reference for jianzipu encoding structure:
  https://github.com/alephpi/jianzipu
- **marimo** — reactive Python notebooks: https://marimo.io
- **Streamlit** — application front end: https://streamlit.io

Hui-position ratios were checked against 陳應時《琴律學》(Shanghai Conservatory
of Music Press, 2015).

No pre-trained machine-learning model is used. The CRF and the structured
perceptron are implemented from scratch in pure Python (`src/qinpos/crf.py`,
`src/qinpos/learn.py`); there is no PyTorch/TensorFlow dependency.

## Licence

Code: MIT. The GQ39 dataset and the JianZiPu font remain under their own
licences and are not redistributed here.
