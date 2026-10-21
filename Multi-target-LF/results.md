# Shared-target split contrast, 2026-10-21

Evaluation pool: `repr_select`, 200 class-0 images. Forty heads, about 5 seconds each. `concentrate_1` matched every archived development seed for pair 0 to 1 at move fraction 0. Client 0's FedAvg weight stayed 0.190.

Counts below are means over the five seeds, out of 200 true class-0 images.

| rho | policy | class 0 | class 1 | class 6 | other | accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 0 | clean | 190.6 | 0.0 | 4.0 | 5.4 | 0.858 |
| 0 | concentrate_1 | 0.0 | 8.4 | 170.2 | 21.4 | 0.811 |
| 0 | split_1_6 | 0.0 | 1.0 | 183.8 | 15.2 | 0.811 |
| 0 | uniform_1_9 | 0.0 | 0.0 | 173.8 | 26.2 | 0.809 |
| 1 | clean | 0.0 | 0.0 | 177.0 | 23.0 | 0.809 |
| 1 | concentrate_1 | 0.0 | 10.0 | 166.6 | 23.4 | 0.811 |
| 1 | split_1_6 | 0.0 | 1.0 | 181.8 | 17.2 | 0.811 |
| 1 | uniform_1_9 | 0.0 | 0.0 | 174.2 | 25.8 | 0.809 |

At rho 1, total variation from the matched clean histogram is 0.050–0.060 for one target, 0.025–0.030 for the half split, and 0.010–0.015 for the uniform split. Total variation from the attacker's label allocation stays 0.495 for the half split and about 0.75 for the uniform split.

At rho 0 the clean histogram still sits on class 0, so distance to that histogram is large for every attack. Distance to the label allocation is also large. All three attacks land on class 6.

Class-1 recall stays between 0.961 and 0.975. The target row was not monopolised.
