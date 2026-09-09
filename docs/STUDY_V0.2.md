# TILKI NEURO v0.2 — spatial EEG baseline study

## Abstract

We evaluated four established EEG classifiers on motor imagery and cue-defined
rest in PhysioNet EEGMMIDB. A version-controlled protocol was frozen before
accessing six new evaluation subjects. Personal calibration gave 64.7% mean
subject balanced accuracy for binary imagery and 48.5% for three-class decoding.
Intervals were wide and reliability gates frequently disabled output. The result
supports further investigation of subject calibration but does not establish a
usable interface, novel method, or superiority to published systems.

## Audit trail

- Protocol freeze: `f814acf6648840f55be186bbac98bcddfd8697d8`.
- Implementation freeze, after 21 software checks and before held-out evaluation:
  `d87c4022588bb571d6354aa4ad0faf919f249d12`.
- One executed study, with no hyperparameter changes or re-selection after its
  test results. Subsequent changes concern presentation, documentation and CI.
- This is a timestamped repository protocol, not registration in an external
  preregistration registry. Subjects 1–6 were already inspected in v0.1.

## Methods

Recordings: subjects 1–12, runs 4/8/12 only (left/right motor imagery, not actual
movement). Sampling 160 Hz. Common average across the 64 recorded EEG channels,
then fixed selection of the 21 FC/C/CP motor channels. One three-second window
starts 0.5 seconds after each eligible annotation. No epoch crosses the next
annotation. Label 0 is T0 rest, 1 is T1 left, 2 is T2 right. Original file hashes
and source event identifiers permit audit. No raw neural signals are published.

Quality screening rejects nonfinite windows, flat channels (SD <0.1 uV), or
peak-to-peak >1,000 uV. The changed amplitude threshold is an engineering choice
frozen before new-subject evaluation, not a validated artifact detector. v0.1's
200 uV criterion removed 44% of its training windows; that observation motivated
a less aggressive threshold, not proof that the excluded windows were clean.
Quality screening affects training and gated outputs. Ungated model test scores
include all finite test windows so hard examples are not silently discarded.

Models:

1. Log-bandpower: Welch bands 4–8, 8–13, 13–30 Hz, training-fitted standardization,
   balanced logistic regression, C=1.
2. CSP/LDA: 8–30 Hz fourth-order Butterworth, six CSP components, covariance
   regularization 0.1, shrinkage LDA and uniform class priors.
3. Riemannian: 8–30 Hz, OAS covariance, Riemannian tangent space with reference
   fitted on training only (`tsupdate=False`), standardized balanced logistic regression.
4. Filter-bank CSP: four components each at 8–12, 12–16, 16–24, 24–30 Hz,
   regularization 0.1, concatenation, standardized balanced logistic regression.

Filtering is zero-phase within each epoch. It uses future samples inside the
window and is unsuitable for claims about causal real-time latency. We used
three-second acquisition windows; reported batch timing excludes acquisition.

Transfer: train subjects 1–4, validate 5–6, test all three runs of 7–12.
Personal: for each subject 7–12, train run 4, validate run 8, test run 12.
These are same-day runs, not longitudinal sessions. Binary imagery is explicitly
separate from idle-inclusive decoding. The highest validation balanced accuracy
chooses the model; ties use the frozen candidate order. All test scores are
reported to avoid hiding unsuccessful candidates.

For three-class output, thresholds are selected on validation only. A gate needs
idle false-activation fraction <=5% and correct active-command recall >=20%.
Among qualifying thresholds, maximize recall, then coverage, then threshold.
If none qualifies, output is disabled. This utility criterion is a research
setting, not a clinical standard. The same small validation set chooses both
model and threshold; overfitting and unstable selection remain concerns.

## Results and interpretation

| Mode / task | Mean subject balanced accuracy | 95% subject-bootstrap interval |
|---|---:|---:|
| Transfer / three-class | 37.4% | 34.4–40.6% |
| Personal / three-class | 48.5% | 41.2–56.1% |
| Transfer / binary imagery | 54.3% | 46.2–62.8% |
| Personal / binary imagery | 64.7% | 51.8–80.5% |

Bootstrap: 2,000 resamples of six subject scores. These are descriptive,
small-cohort percentile intervals, not multiplicity-adjusted tests or evidence of
population-wide efficacy. Chance balanced accuracy is 33.3% and 50%, respectively.
The methods differ in training and test sets; these figures do not isolate the
causal effect of calibration, referencing, channels, filters or classifier choice.
v0.1 used different subjects, windows and preprocessing, so its 33.34% is not a
matched improvement comparator.

Personal binary performance ranged from 39.3% to 100%. The maximum comes from
only 15 test trials and must not become the headline. The gate enabled only
subjects 7 and 12. Subject 7 exceeded the idle target on test (1/15 idle windows),
and subject 12 observed 0/15. Neither establishes a sufficiently low operational
error rate. Disabled systems have zero errors because they emit nothing; that is
not successful communication. Coverage and useful recall must accompany errors.

## Outstanding research

1. Replicate on a new dataset with repeated days and realistic continuous idle.
2. Compare a common held-out run and matched calibration budgets with enough
   subjects for precise paired estimates; preregister primary endpoints.
3. Add calibrated uncertainty and explicit artifact models while preserving
   useful throughput. Evaluate eye/EMG contamination instead of assuming neural origin.
4. Design causal online processing; measure complete acquisition-to-command latency
   and actual processor energy. No embedded-energy or physical-chip claim exists here.
5. For participant research, collaborate with a neuroengineering institution and
   obtain the relevant ethics review and informed consent.

## Reuse

This release integrates established open scientific tools. Cite PhysioNet and the
underlying MNE/pyRiemann methods as applicable. The repository contains no claim
of original CSP or Riemannian algorithms, clinical restoration, or a new neurochip.
