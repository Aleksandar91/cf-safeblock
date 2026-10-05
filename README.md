# Score archive

These files are the scored runs. This repository does not contain a manuscript, training code, images, or model weights.

Each `verdict.json` includes the per-seed records. A mean in a table is the arithmetic mean of the corresponding seed field (`intended_asr`, `source_recall`, `sink_share`, or `per_class_recall`). `SHA256SUMS` is the SHA-256 of each `verdict.json`.

| File | Condition recorded by that file |
| --- | --- |
| `phase1/results/verdict.json` | Fashion-MNIST pair 0 to 1, label flip, missing-class scale 0, five move fractions |
| `phase2/results/verdict.json` | Ten Fashion-MNIST pairs, label flip, missing-class scale 0 |
| `phase3/results/verdict.json` | MNIST pair 0 to 1, label flip, missing-class scale 0 |
| `phase3/pv_results/verdict.json` | Four-class Apple slice, label flip, shared regime and target monopoly |
| `phase4_rho1/fashion_dose/verdict.json` | Fashion-MNIST pair 0 to 1, label flip, `honest_rho` 1, five move fractions |
| `phase4_rho1/fashion_pairs/verdict.json` | Ten Fashion-MNIST pairs, label flip, `honest_rho` 1 |
| `phase4_rho1/mnist/verdict.json` | MNIST pair 0 to 1, label flip, `honest_rho` 1 |
| `clean_baseline/verdict.json` | Fashion-MNIST, no label flip, scale 0 and scale 1, move fractions 0 and 1 |

`phase1/results/verdict.json` does not repeat the scale inside each seed row. The condition is the one named in this table. The other files carry `honest_rho` or `rho` in the JSON.
