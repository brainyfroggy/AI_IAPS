# BIDS validation gate

Both production inputs passed the official precompiled BIDS Validator before
Sub5 preprocessing:

| Input | Validator | Bundled schema | Files | Errors | Warnings |
|---|---:|---:|---:|---:|---:|
| Full Sub4/5/6 pilot BIDS | 3.0.1 | 1.2.7 | 118 | 0 | 250 |
| Isolated Sub5 February SyN input | 3.0.1 | 1.2.7 | 18 | 0 | 35 |

The remaining warnings are recommendation-only: `TOO_FEW_AUTHORS`,
`JSON_KEY_RECOMMENDED`, and `SIDECAR_KEY_RECOMMENDED`. There are no missing-file,
participant, naming, metadata-type, or structural errors.

The first February-subset validation correctly found that the old builder had
copied the full three-subject `participants.tsv`. The builder now generates a
one-row Sub5 table without modifying the canonical source. The pre-fix subset
and validator report are retained with `_pre_participant_fix` in their names.

Machine-readable reports:

- `pilot_bids_validator_v3.0.1.json`
- `sub05_feb_syn_bids_validator_v3.0.1.json`

Reports carrying `_schema-tag-v1.10.0` are retained only as an audit of an
incorrect validator invocation. That tag selected an older schema-package
release and produced spurious warnings; production gates use Validator 3.0.1's
bundled schema.
