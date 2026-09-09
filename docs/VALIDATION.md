# Validation record

Executed in the authoring environment on 2026-09-09:

- Default synthetic benchmark; report and replay in `examples/synthetic/`.
- NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, Python 3.12.14.

- 12 tests passed: split isolation, deterministic data, serialization, invalid and
  flat signal rejection, channel mismatch, command confirmation/rearming/undo,
  validation-only threshold choice, abstention metrics, full synthetic pipeline,
  and MNE importer units/channel order/annotation boundaries/missing files.
- Python source compilation passed.
- Tk imports successfully; visual desktop execution was not verified because this
  environment has no graphical display. Communication logic is tested headlessly.

- Imported 18 real EDF files from six PhysioNet subjects into 540 windows.
- Real EEG evaluation completed: baseline balanced accuracy 33.34%; rejection
  coverage 36.67% and active-command recall 3.33%. This fails useful decoding.
- GitHub Research checks completed successfully on the initial code commit.

Not established: useful assistive performance, clinical utility, superiority to published
methods, adaptive drift recovery, embedded power consumption, fabricated hardware.
The desktop interface depends on Tk/display availability; state-machine tests can
run without a GUI. GitHub Actions status must be checked on GitHub separately.

## v0.2

- 21 tests passed, including all four spatial pipelines, isolated window filtering,
  subject-leakage rejection, fail-closed gates, and deterministic subject bootstrap.
- Frozen protocol and source committed before the new six-subject evaluation.
- Complete study ran successfully on 36 EDFs / 12 subjects / 1,080 windows.
- The two tasks and both transfer/personal modes ran; all candidates disclosed.
- Evidence dashboard rendered from JSON and visually inspected.
- Desktop console code compiled; Tk/display interaction not verified in this
  headless environment. An attempted display dependency installation was unavailable.
- No meaningful end-to-end communication, clinical or hardware claim is supported.
