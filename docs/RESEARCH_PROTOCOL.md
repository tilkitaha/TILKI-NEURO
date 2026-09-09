# Research protocol v0.1

## Hypothesis and comparison

A future rejection/adaptation system should reduce unintended commands at matched
useful-command rate and a specified computation budget. The present implementation
is a baseline and rejection harness, not evidence for that hypothesis.

Train on 50% of subject groups, validate on 25%, test on 25% for the default 12
subjects (other group counts round according to GroupShuffleSplit). Fit feature
scaling and logistic regression only on quality-passing training windows. Choose
a threshold from 81 candidates using validation labels only: require idle false
activation fraction <=0.05, maximize correct active-command recall, then coverage,
then threshold. Do not tune on test results. A validation target is not a test-time
guarantee. For any changed protocol, create a new experiment record.

## Metrics

- Coverage: non-abstained windows / all windows (includes classified idle).
- Accepted accuracy: correct / non-abstained; null when coverage is zero.
- Correct command recall: correctly emitted active labels / true active windows.
- Wrong commands: any active output that differs from truth, including idle truth.
- Idle false activation fraction: active outputs / true idle windows.
- Idle false activations/minute: active outputs divided by total sampled idle-window
  minutes. This counts window classifications, NOT confirmed board messages or
  continuous device-use events. It excludes gaps between sampled windows.

Report rejection benefits jointly with coverage and command recall. The current
benchmark is a single deterministic split. Before making scientific claims, run
multiple predefined subject splits, report per-subject results and subject-level
bootstrap intervals, add stronger baselines, and seek independent replication.

## Stress tests and next milestones

Implemented: one flat channel on all held-out windows; invalid input unit tests.
Not implemented: realistic electrode drift, ocular/EMG interference, temporal
adaptation, calibrated uncertainty, independent hardware energy measurement.

No custom silicon or human experimentation belongs in the initial software task.
Future participant collection needs an institutional neuroengineering partner,
consent and ethics review, and an appropriate data management plan. EEG findings
cannot establish intracortical implant performance. Any hardware claim must list
processor, sampling, latency including acquisition, memory, power measurement
method, and energy per useful command.
