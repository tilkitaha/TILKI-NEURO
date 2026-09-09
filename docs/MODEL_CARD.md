# Model card

Intended use: software experiments and education about EEG classification and
abstention. Outputs: 0 idle, 1 left, 2 right, -1 rejected. Phrase labels are UI
choices; the system does not infer words or free-form thoughts.

Model: log mean spectral power in 4–8, 8–13, 13–30 Hz bands per channel,
StandardScaler, multiclass logistic regression. Two-second non-overlapping windows
in shipped data. No pretrained model is shipped; each run trains locally.

Synthetic data: 12 artificial subjects with engineered lateralized mu attenuation,
subject gain/frequency variation, and additive noise. Labels are generated, not
measured intentions. This deliberately simple data is unsuitable for efficacy claims.

Quality heuristic: reject non-finite signals, any channel standard deviation under
0.1 uV, or any channel peak-to-peak over 200 uV. These thresholds are not validated
across headsets, reference configurations, populations, or impairments. Inputs in
volts rather than uV would invalidate quality assessment.

Known limitations: no headset integration, no implanted recording support, no
online adaptation, no confidence calibration, no patient study, no security or
clinical certification, no battery/ASIC prototype. Logistic scores can be confident
on unfamiliar inputs. Quality checks miss many artifacts. Rest annotations may not
represent real-world idle. A toy result cannot justify medical use.

Privacy: software runs locally; explicit download connects to dataset providers.
The desktop demo stores phrases in process memory only. No cloud telemetry, LLM,
message sending, or account is required. Do not commit identifiable neural data.

## v0.2 spatial research extension

The retained v0.1 model above remains reproducible. v0.2 adds 21 motor channels,
common-average reference, CSP/LDA, filter-bank CSP, and pyRiemann tangent-space
models. Its study protocol, quality thresholds, real-data population, results,
and limitations are in `STUDY_V0.2.md`. No online adaptation is implemented.

The research console displays stored real-data results. The original communication
demo remains explicitly synthetic and uses its original decoder. No trained
human-data model is silently substituted into the toy demo. Binary imagery scores
exclude idle and cannot justify autonomous commands.
