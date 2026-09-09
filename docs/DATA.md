# Dataset contract

Use `tilki_neuro.data.Dataset` to validate and save an NPZ. Loading uses
`allow_pickle=False` and requires these keys:

| Key | Type / meaning |
|---|---|
| x | Finite numeric array `[windows, channels, samples]`, microvolts |
| y | Integer labels `[windows]`: 0 idle, 1 left, 2 right |
| groups | Stable subject IDs `[windows]`; do not use window IDs |
| sfreq | Scalar Hz, >=64; window length >=1 second |
| source | Nonempty provenance string |

Use a consistent channel order, reference scheme, and sample rate across subjects.
Store only complete, non-overlapping windows. Maintain source file/run/event
provenance externally for custom data. Never split adjacent overlapping windows
between evaluation partitions. Reject invalid samples or document any imputation.
The current PhysioNet importer fixes channel order, but does not re-reference,
perform ICA, remove ocular contamination, or guarantee genuine neural intent.

The benchmark refuses missing classes in any partition. Quality rejection may
remove a whole training class, in which case training fails. Dataset groups must
represent people to support the reported cross-subject interpretation.
