# fed-causal committed artifacts

`reference.json` is the **precomputed deployed-default reference** the hosted
backend can serve without re-scoring. It is generated deterministically from the
seeded synthetic event panel, so it is fully reproducible.

It contains:

- `request` — the deployed-default request (`event_window=1`, `estimation_window=120`,
  `model="market"`, `surprise="all"`, `n_placebo=500`, `data_source_pref="synthetic"`,
  `seed=7`, `alpha=0.05`).
- `summary` — the deployed-default summary block (`car_mean`, `car_tstat`,
  `car_hac_pvalue`, `bmp_stat`, `placebo_pctile`, `n_events`, `did_coef`,
  `did_pvalue`, `n_tests`, `did_net_spread`, `fed_effect_is_tradable`,
  `data_source`). The verdict is **`false`** — the honest-null deliverable.
- `known_car_recovery` — the injected CAR and the CAR recovered on the
  rate-sensitive ("treated") vs. control names (ground-truth recovery).
- `pure_noise_honest_null` — the no-effect control's summary: placebo p-value and
  HAC p-value are both non-significant and every verdict condition is `false`.

## Regenerating

```bash
python scripts/build_reference.py
```

The build script and the `test_reference_artifact_matches_live` regression test
keep this artifact in lock-step with the live `run_analysis` output. Do not edit
`reference.json` by hand.
