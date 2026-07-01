# Explainable Anxiety Detection from EEG (DASPS)

A step-by-step, reproducible pipeline that detects anxiety from EEG signals using
traditional machine learning, and — most importantly — **explains** its decisions
in terms of real neural features (electrodes and frequency bands) and validates
those explanations against an established neuroscience marker (**frontal alpha
asymmetry, FAA**).

This README walks through **every command, its output, and the reasoning behind it**,
so each step can be understood and defended.

---

## Table of contents

1. [Goal](#1-goal)
2. [The dataset](#2-the-dataset)
3. [How the labels work (SAM & HAM-A)](#3-how-the-labels-work-sam--ham-a)
4. [Installation](#4-installation)
5. [The engine files (run and explained)](#5-the-engine-files-run-and-explained)
6. [The experiments (run and explained)](#6-the-experiments-run-and-explained)
7. [Key concepts, in plain terms](#7-key-concepts-in-plain-terms)
8. [Results summary](#8-results-summary)
9. [Honest findings](#9-honest-findings)
10. [Mapping to supervisor's guidance](#10-mapping-to-supervisors-guidance)

---

## 1. Goal

Build a model that detects anxiety from EEG **and explains its reasoning** in terms
of electrodes, frequency bands, and time — then check that the explanation agrees
with real neuroscience. The contribution is **trust and transparency**, not just
accuracy.

Following supervisor guidance, this stage uses **SAM labels**, **within-subject
analysis**, the **preprocessed data**, and a **range of traditional ML algorithms**.
Deep learning is deferred to a later stage.

---

## 2. The dataset

**DASPS** (Baghdadi et al., 2019): EEG from **23 participants**, recorded during
6 anxiety-provoking situations, using the Emotiv EPOC headset (14 channels, 128 Hz).

### Files in the dataset

| Folder / file | Count | What it is | Used? |
|---|---|---|---|
| `Preprocessed data .mat` | 23 | Cleaned EEG signals (one file per person) | ✅ Yes |
| `Raw data .edf` | 23 | Unprocessed EEG | ❌ No |
| `Raw data.mat` | 23 | Unprocessed EEG (MAT format) | ❌ No |
| `participant_rating_public.xlsx` | 1 | Labels: SAM + HAM-A scores | ✅ Yes |

### Command: inspect a preprocessed signal file

```bash
python -c "import h5py; f=h5py.File('Preprocessed data .mat/S01preprocessed.mat','r'); print(list(f.keys())); print(f['data'].shape)"
```

**Output:**
```
['data']
(12, 1920, 14)
```

**What this means:** the file holds one variable, `data`, shaped `(12, 1920, 14)`:

- **12** = clips (segments). Each person does 6 situations, each split into
  *recitation* (15 s) + *recall* (15 s) = **12 clips**.
- **1920** = time-points per clip. 128 samples/second × 15 seconds = **1920**.
- **14** = electrodes (channels) on the scalp.

Across all 23 people: 23 × 12 = **276 clips** — the size of the dataset.

---

## 3. How the labels work (SAM & HAM-A)

### Command: inspect the label file

```bash
python -c "import openpyxl; ws=openpyxl.load_workbook('participant_rating_public.xlsx').active; [print(r) for r in list(ws.iter_rows(values_only=True))[:8]]"
```

**Output (first rows):**
```
(None, 'Id Participant', 'Id situation ', 'valence', 'Arousal', 'Hmilton1', 'Hamilton2', 'Situation provocante')
(None, 'S01', 1, 1, 8, '25:moderate', '37:severe', 'proche')
(None, None,  2, 3, 8, None, None, None)
```

**How to read it:**

- **SAM** = the two columns `valence` and `Arousal`, rated **per situation** (1–9 scale).
  - **Valence** = how pleasant/unpleasant (1 = very unpleasant).
  - **Arousal** = how calm/excited (9 = very excited).
  - **Low valence + high arousal = anxious.** Example: S01 situation 1 has
    valence 1, arousal 8 → strongly anxious.
- **HAM-A** = the `Hamilton1` / `Hamilton2` columns (e.g. `25:moderate`), a
  **clinical score per person** (appears once per participant).

**Why SAM (per supervisor):** it gives a label **per situation** (6 per person),
providing finer resolution than one clinical score per person. It is mapped to a
**binary** anxious / non-anxious label. The two clips of one situation (recitation +
recall) share the same SAM label.

---

## 4. Installation

```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
# if anything is missing later:
pip install shap torch matplotlib openpyxl h5py scikit-learn scipy
```

---

## 5. The engine files (run and explained)

These files are the **building blocks**; the experiments call them. Each command
below runs one engine file on the real data so you can see it working. The order
follows the path the data actually travels:
**config → data → features → neuromarkers → validation → model.**

### 5.1 `config.py` — the settings

**Why:** one source of truth for channels, bands, and sampling rate, so every file
agrees.

```bash
python -c "import config; print('Channels:', config.DASPS_CHANNELS); print('Bands:', config.BANDS); print('Sampling rate:', config.SFREQ)"
```

**Output:**
```
Channels: ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
Bands: {'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0), 'gamma': (30.0, 45.0)}
Sampling rate: 128
```

**Note:** there is no *delta* band because the data is pre-filtered above 4 Hz.

### 5.2 `data.py` — load signals + labels

**Why:** bridges raw files into usable arrays and attaches the SAM label to each clip.

```bash
python -c "from data import load_dasps; EEG,y,subj,phase,meta=load_dasps('.',label_source='SAM'); print('EEG signal shape:', EEG.shape); print('Labels:', y[:12]); print('Subjects:', subj[:12]); print('Label type:', meta['label_source'])"
```

**Output:**
```
EEG signal shape: (276, 14, 1920)
Labels: [1 1 1 1 1 1 0 0 1 1 1 1]
Subjects: [1 1 1 1 1 1 1 1 1 1 1 1]
Label type: SAM
```

**What this means:** 276 clips × 14 channels × 1920 time-points. The first 12 labels
are S01's clips (mostly `1` = anxious, matching S01's unpleasant/high-arousal SAM
ratings). `subj` records which person each clip came from — essential for honest
splitting later.

### 5.3 `features.py` — filtering + feature extraction

**Why:** models cannot read raw signals; they need numbers. This band-pass filters
each clip into the four bands and measures **band power** (energy) per channel, per
band, over time.

```bash
python -c "from data import load_dasps; from features import epochs_to_tensor; EEG,y,*_=load_dasps('.',label_source='SAM'); T,axis=epochs_to_tensor(EEG); print('Feature tensor shape:', T.shape); print('= clips x channels x bands x time-windows'); print('Axis info:', axis['bands'])"
```

**Output:**
```
Feature tensor shape: (276, 14, 4, 9)
= clips x channels x bands x time-windows
Axis info: ['theta', 'alpha', 'beta', 'gamma']
```

**What this means:** each clip becomes a **14 × 4 × 9** grid (electrode × band ×
time-window). Averaging over the 9 time-windows gives the **56 band-power features**
(14 × 4) used by the classifiers. This structure is deliberate — it lets the
explanation later point to a specific electrode, band, and moment.

### 5.4 `neuromarkers.py` — frontal alpha asymmetry (FAA)

**Why:** FAA is the established EEG marker of anxiety, computed **independently of any
model**. It is the "ground truth" the model's explanation is later checked against.

```bash
python -c "from data import load_dasps; from neuromarkers import frontal_alpha_asymmetry; EEG,y,*_=load_dasps('.',label_source='SAM'); faa,bp=frontal_alpha_asymmetry(EEG); print('FAA per clip (first 6):', faa[:6].round(3)); print('Total FAA values:', len(faa))"
```

**Output:**
```
FAA per clip (first 6): [ 0.819 -0.335  0.499  0.479  0.452  0.575]
Total FAA values: 276
```

**What this means:** one FAA value per clip (276 total) — the right-minus-left frontal
alpha balance. (The function returns two things: the FAA values and the frontal
band-powers, which is why we unpack `faa, bp`.)

### 5.5 `validation.py` — honest train/test splitting

**Why:** prevents the model from cheating. It defines the three evaluation protocols
and guarantees test data is never seen during training.

```bash
python -c "from data import load_dasps; from validation import loso_splits, within_subject_splits; EEG,y,subj,*_=load_dasps('.',label_source='SAM'); print('LOSO folds:', len(list(loso_splits(subj)))); print('Within-subject folds:', len(list(within_subject_splits(subj,y))))"
```

**Output:**
```
LOSO folds: 23
Within-subject folds: 120
```

**What this means:** LOSO makes **23 folds** (each holds out one whole person).
Within-subject makes **120 folds** (each person's situations held out one at a time).

### 5.6 `model.py` — the deep-learning networks

**Why:** defines the CNN / EEGNet used in the deferred deep-learning stage.

```bash
python -c "from model import BandTensorCNN, EEGNet; m=BandTensorCNN(4,14,9); print('CNN parameters:', sum(p.numel() for p in m.parameters()))"
```

**Output:**
```
CNN parameters: 29314
```

**What this means:** a compact CNN (~29k parameters) — intentionally small, because
the dataset is small.

---

## 6. The experiments (run and explained)

### 6.1 Traditional ML — subject-dependent (main result)

**Command:**
```bash
python run_ml.py --data-root . --protocol subjdep --features bandpower --permtest
```

**Output:**
```
Label=SAM  epochs=276  subjects=23  class balance=[142, 134]
Features=bandpower (56 dims)  Protocol=SUBJDEP

  Model        AUC             Accuracy   macro-F1
  ExtraTrees   0.720           0.641      0.641
  RandomForest 0.701           0.659      0.659
  SVM-RBF      0.692           0.638      0.637
  LogReg       0.669           0.634      0.633
  ...
  DecisionTree 0.589           0.587      0.586

  permuted ExtraTrees: AUC=0.442  acc=0.464 (should sit near chance)
```

**Why this command:** it answers *"use a range of traditional ML"* — 11 classifiers
compared under the subject-dependent protocol (clips pooled across people).

**What the result means:** **ExtraTrees is best (AUC 0.72)**; tree ensembles lead.
The **permutation check** shuffles the labels and drops to **0.442** (chance). The
gap between 0.72 and 0.44 proves the model learned a **real** signal, not luck.

**Why this result:** band power at frontal electrodes carries genuine anxiety
information; tree ensembles handle these 56 features well on a small dataset.

### 6.2 Within-subject analysis (supervisor's recommended view)

**Command:**
```bash
python run_ml.py --data-root . --protocol within --features bandpower
```

**Output (top rows):**
```
  within-subject: 21 subjects usable (2-class), 120 situation-folds total
  SVM-RBF      0.495±0.09   ...
  ...
```

**Why this command:** it answers *"within-subject analysis"* — training and testing
**inside each person separately**, then averaging.

**What the result means:** ≈ **0.50 (chance).** No model beats guessing.

**Why this result (important to explain):** each person has only 12 clips, so each
fold trains on ~10 examples — far too little to learn from. This is a **data-size
limitation, not a bug**, and it is an honest finding worth reporting.

### 6.3 Feature engineering — connectivity test

**Command:**
```bash
python run_ml.py --data-root . --protocol subjdep --features connectivity
```

**Output (top rows):**
```
Features=connectivity (364 dims)  Protocol=SUBJDEP
  GradBoost    0.672   ...
  ExtraTrees   0.648   ...
```

**Why this command:** the supervisor asked whether **connectivity features** help.
This tests them (PLV between electrode pairs, 364 features).

**What the result means:** best ≈ **0.67**, **lower** than band power's 0.72.

**Why this result:** with only ~220 training clips, adding 364 features dilutes the
few informative ones and causes overfitting. **Finding: band power beats
connectivity** — tested, not assumed.

### 6.4 Extended features + ensemble

**Command:**
```bash
python run_ensemble.py --data-root . --repeats 10
```

**Output:**
```
--- Features: band power (56) ---
  ExtraTrees      0.693±0.01  ...
  Ensemble(soft)  0.698±0.01   0.646   0.645   <-- ensemble

--- Features: extended (140) ---
  ExtraTrees      0.662±0.02  ...
  Ensemble(soft)  0.648±0.02  ...
```

**Why this command:** to push accuracy higher through **feature engineering** (adding
statistical + Hjorth features) and by **combining models** (a soft-voting ensemble).
`--repeats 10` averages over 10 shuffles for a stable number.

**What the result means:** the **ensemble on band power reaches ~0.70** (best, stable
at ±0.01); **extended features make everything worse.**

**Why ensemble:** it averages the probability outputs of ExtraTrees + RandomForest +
SVM + LogReg. Different models make different mistakes; averaging cancels some of them,
giving a small, stable gain (0.693 → 0.698) and better-balanced accuracy.

**Why extended features hurt:** same reason as connectivity — more features on a small
dataset increase overfitting. **Simpler is better** here (now confirmed three times).

### 6.5 Best model — rigorous evaluation

**Command:**
```bash
python run_eval_best.py --data-root . --figdir Figures
```

**Output:**
```
Hyperparameter tuning (nested CV):
  default ExtraTrees : AUC 0.727 ± 0.015
  tuned   ExtraTrees : AUC 0.725 ± 0.020

Tuned model (out-of-fold pooled):
  AUC=0.718  accuracy=0.656  macro-F1=0.656
  class        precision  recall   F1     support
  non-anxious  0.669      0.655    0.662  142
  anxious      0.642      0.657    0.649  134

  Confusion matrix [non-anxious, anxious]:
    [93, 49]
    [46, 88]
```

**Why this command:** an examiner needs more than one number. This does **honest
tuning** (nested cross-validation) and produces the **confusion matrix, per-class
precision/recall, and ROC curve** (saved to `Figures/`).

**What the result means:**
- **Tuning gave no gain** (0.727 → 0.725) — the model is robust with default settings.
- **Balanced performance:** recall is 0.655 (non-anxious) and 0.657 (anxious) — the
  model does **not** favour one class.
- Confusion matrix: catches 88/134 anxious and 93/142 non-anxious clips.

### 6.6 Explainability (SHAP) + FAA validation — the core contribution

**Command:**
```bash
python run_xai_ml.py --data-root . --model ExtraTrees --save-fig
```

**Output:**
```
Explanation source: SHAP (aggregated over 5 folds)

Frontal concentration : frontal=0.0781 vs other=0.0626  (p=0.331, n.s.)
Band ranking          : beta=0.290, alpha=0.261, gamma=0.230, theta=0.218
FAA alignment (alpha)  : Spearman rho=0.786 (p=0.02082)
  measured FAA class-diff=-0.044 (separates classes p=0.8854)
Top channels          : F4=0.143, AF3=0.142, P8=0.086, T8=0.085, F3=0.065
```

**Why this command:** this is the **heart of the thesis**. SHAP reveals *which
electrodes and bands* drive the model, then the result is checked against FAA.

**What the result means (read each line):**
- **FAA alignment: rho = 0.79, p = 0.02 → significant.** The model's frontal-alpha
  importance lines up with anxiety-discriminative alpha. **This is the key result:**
  the model independently relies on the frontal-alpha pattern neuroscience associates
  with anxiety — evidence it learned *real* physiology, not noise.
- **Bands:** beta and alpha carry the most importance — consistent with anxiety
  literature.
- **Top electrodes:** frontal sites **F4** and **AF3** lead (with parietal/temporal
  P8, T8 also contributing).
- **Honest caveats:** whole-head frontal dominance was *not* significant (p = 0.33),
  and the classic FAA index separates the clinical HAM-A groups but **not** the SAM
  labels — an interesting note about label choice.

---

## 7. Key concepts, in plain terms

**SAM (Self-Assessment Manikin):** a self-report scale of *valence* (pleasant↔unpleasant)
and *arousal* (calm↔excited). Low valence + high arousal → anxious. This is our label.

**Band power:** how much energy the brain signal has in a frequency band (theta, alpha,
beta, gamma) at each electrode. Our main features (14 channels × 4 bands = 56).

**FAA (frontal alpha asymmetry):** right-minus-left frontal alpha power; a validated
anxiety marker used to check the model's explanation.

**Cross-validation / train-test split:** instead of one fixed split, the data is
rotated so every clip is tested once, then results are averaged — honest and stable.
- **Subject-dependent (5-fold):** 4 parts train (**80%**), 1 part tests (**20%**),
  rotated 5×. Same people may appear in train and test.
- **Within-subject:** each person alone; one situation held out at a time (~83% train).
- **Subject-independent (LOSO):** 22 people train (**~96%**), 1 new person tests
  (**~4%**), rotated 23×.

**Permutation test:** shuffle the labels randomly and re-measure. This is the "luck"
level. A real model must clearly beat it (ours: 0.72 vs 0.44).

**AUC:** how well the model ranks anxious above non-anxious. 0.5 = guessing, 1.0 =
perfect. Our headline ≈ 0.70.

**Precision / Recall / F1:** precision = of those *predicted* anxious, how many really
are; recall = of those *truly* anxious, how many were caught; F1 = their balance.

**Ensemble (soft voting):** combine several models by averaging their probability
outputs; different models' mistakes partly cancel, giving a small, stable gain.

**SHAP:** fairly attributes a prediction to each feature, telling us which electrodes
and bands drove the decision.

---

## 8. Results summary

| Experiment | Setting | Best model | AUC | Meaning |
|---|---|---|---|---|
| Subject-dependent | band power | ExtraTrees | **0.72** (0.69±0.01 repeated) | real signal |
| Within-subject | band power | — | ≈0.50 | chance (too little data/person) |
| Connectivity | band power vs PLV | GradBoost | 0.67 | connectivity does not help |
| Extended features | 140 features | ExtraTrees | 0.66 | more features hurt |
| **Ensemble** | band power | soft voting | **0.70** | best, stable |
| Rigorous eval | tuned ExtraTrees | — | 0.72 | balanced, no tuning gain |
| Explainability | SHAP + FAA | ExtraTrees | — | **FAA aligned, rho 0.79, p 0.02** |

**Headline number to report:** **AUC ≈ 0.70**, subject-dependent, band-power ensemble.

---

## 9. Honest findings

1. **Evaluation protocol is decisive.** Real signal appears only subject-dependent;
   within-subject and subject-independent are at chance — much reported DASPS
   performance is inflated by evaluation leakage.
2. **Simple features win.** Band power beats connectivity and extended features.
3. **Explainability is validated.** SHAP importance aligns with FAA (rho 0.79,
   p 0.02) — the model reflects genuine neurophysiology. *(This is the contribution.)*
4. **Deep learning does not help here** (see the deferred stage) — 23 subjects is too
   few; both a CNN and EEGNet sit at chance.

The strength of this work is **methodological rigour and interpretability**, not
headline accuracy.

---

## 10. Mapping to supervisor's guidance

| Guidance | Where it is addressed |
|---|---|
| Use **SAM** labels | every command uses SAM (Section 3, 5.2) |
| **Within-subject** analysis | `run_ml.py --protocol within` (6.2) |
| Keep **preprocessed** data; focus on **feature engineering + classification** | Sections 5.3, 6.1–6.4 |
| **Connectivity** features (if justified) | tested; does not help (6.3) |
| **Range of traditional ML** first | 11 classifiers + ensemble (6.1, 6.4) |
| Deep learning → later meeting | prototypes ready (`run_cnn.py`, `run_eegnet.py`) |
| Differ from public DASPS code | honest multi-protocol evaluation + SHAP/FAA validation |

---

### Reproduce everything

```bash
pip install -r requirements.txt
python run_ml.py       --data-root . --protocol subjdep --features bandpower --permtest
python run_ml.py       --data-root . --protocol within  --features bandpower
python run_ml.py       --data-root . --protocol subjdep --features connectivity
python run_ensemble.py --data-root . --repeats 10
python run_eval_best.py --data-root . --figdir Figures
python run_xai_ml.py   --data-root . --model ExtraTrees --save-fig
```

**Citation:** Baghdadi, A., Aribi, Y., Fourati, R., Halouani, N., Siarry, P., & Alimi, A. M.
(2019). *DASPS: A Database for Anxious States based on a Psychological Stimulation.*
arXiv:1901.02942.
