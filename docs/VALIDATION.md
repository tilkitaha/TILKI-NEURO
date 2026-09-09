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
