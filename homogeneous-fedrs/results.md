# Homogeneous FedRS, pair 0 to 1, development pool

Each head was trained twice. Repeated weights and scores matched, and the saved source predictions reproduce the scores below.

Counts are five-seed means out of 200 true class-0 images.

| M_t | class 0 | class 1 | class 3 | class 6 | accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.100 | 174.8 | 3.6 | 10.0 | 6.6 | 0.871 |
| 1.000 | 0.0 | 190.4 | 2.6 | 3.0 | 0.754 |

Shared-endpoint source recall by seed: 0.875, 0.875, 0.880, 0.880, 0.860. Monopoly intended ASR by seed: 0.950, 0.950, 0.955, 0.955, 0.950. Class-1 recall is 0.975 at the shared endpoint and 1.000 at monopoly, on every seed.

The lifted attack in the archived development dose has source recall 0 at both endpoints, shared mean ASR 0.042 with sink class 6, and monopoly mean ASR 0.949 with sink class 1.
