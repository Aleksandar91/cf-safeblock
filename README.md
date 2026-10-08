# Score archive

These files are the scored runs behind the tables. This repository does not contain a manuscript, training code, images, or model weights.

Each `verdict.json` includes the per-seed records. A reported mean is the arithmetic mean of the matching seed field. `SHA256SUMS` lists the SHA-256 of every `verdict.json`, with two spaces before the path.

On a flip row the fields are `intended_asr`, `source_recall`, `sink_class`, and `sink_share`. On a clean row, where no label was rewritten, the fraction predicted as class 1 is stored as `predicted_as_class_1`. Per-class recall is `per_class_recall`, with class 0 at index 0.

| File | Condition recorded by that file |
| --- | --- |
| `phase1/results/verdict.json` | Fashion-MNIST pair 0 to 1, label flip, missing-class scale 0, five move fractions |
| `phase2/results/verdict.json` | Ten Fashion-MNIST pairs, label flip, missing-class scale 0 |
| `phase3/results/verdict.json` | MNIST pair 0 to 1, label flip, missing-class scale 0 |
| `phase3/pv_results/verdict.json` | Four-class Apple slice, label flip, shared regime and target monopoly |
| `phase4_rho1/fashion_dose/verdict.json` | Fashion-MNIST pair 0 to 1, label flip, ordinary softmax, five move fractions |
| `phase4_rho1/fashion_pairs/verdict.json` | Ten Fashion-MNIST pairs, label flip, ordinary softmax |
| `phase4_rho1/mnist/verdict.json` | MNIST pair 0 to 1, label flip, ordinary softmax |
| `clean_baseline/verdict.json` | Fashion-MNIST, no label flip, scale 0 and scale 1, move fractions 0 and 1 |
| `clean_baseline_mnist/verdict.json` | MNIST pair 0 to 1, no label flip, scale 0 and scale 1, move fractions 0 and 1 |
| `centralized_head/verdict.json` | Fashion-MNIST pooled client-training features, one linear head, no clients, no label flip |

`phase1/results/verdict.json` does not repeat the scale inside each seed row. The condition is the one named in this table. The later flip files carry `honest_rho`. The two clean files carry `rho` and `"label_flip": false`. The centralized file is one head on the pooled sample. The MNIST clean file and the centralized file were added after the 5 October archive. The eight files above them are unchanged.
