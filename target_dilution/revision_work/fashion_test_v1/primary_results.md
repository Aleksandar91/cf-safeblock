# Fashion-MNIST held-out evaluation: primary pair 0 to 1

All rates use the official 10,000-image final test set (1,000 per class).
Means and sample SDs are across five head initializations/minibatch orders
conditional on one fixed trunk, partition, and test set. The five seeds
are not five independently sampled test populations.

Clean target rate is the fraction of true class-0 images predicted as class 1
without a label flip. Paired delta is the seed-matched attack rate minus
that clean rate. Entries/exits count actual class-0 image prediction
transitions into/out of class 1 between paired clean and attacked heads.
ASR is intended-target ASR on attacked rows.

| rho | move f | measured Mt | clean source recall | attack source recall | clean target rate | attack ASR | paired delta ASR | clean sink share | attack sink share | entries | exits | clean acc. | attack acc. | clean class-1 recall | attack class-1 recall | attack sink classes | dev attack ASR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 0.0 | 0.00 | 0.100000 | 0.961 +/- 0.002 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.032 +/- 0.002 | 0.032 +/- 0.002 | 0.011 +/- 0.001 | 0.840 +/- 0.005 | 31.800 +/- 1.924 | 0.000 +/- 0.000 | 0.854 +/- 0.000 | 0.798 +/- 0.001 | 0.960 +/- 0.000 | 0.967 +/- 0.002 | 6,6,6,6,6 | 0.042 |
| 0.0 | 0.25 | 0.325474 | 0.961 +/- 0.002 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.033 +/- 0.001 | 0.033 +/- 0.001 | 0.012 +/- 0.001 | 0.838 +/- 0.003 | 32.600 +/- 0.894 | 0.000 +/- 0.000 | 0.854 +/- 0.000 | 0.798 +/- 0.000 | 0.959 +/- 0.000 | 0.965 +/- 0.001 | 6,6,6,6,6 | 0.045 |
| 0.0 | 0.50 | 0.550947 | 0.959 +/- 0.002 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.030 +/- 0.002 | 0.030 +/- 0.002 | 0.012 +/- 0.001 | 0.837 +/- 0.002 | 30.400 +/- 1.673 | 0.000 +/- 0.000 | 0.854 +/- 0.001 | 0.798 +/- 0.001 | 0.959 +/- 0.000 | 0.963 +/- 0.000 | 6,6,6,6,6 | 0.044 |
| 0.0 | 0.75 | 0.774526 | 0.957 +/- 0.001 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.024 +/- 0.001 | 0.024 +/- 0.001 | 0.013 +/- 0.001 | 0.839 +/- 0.004 | 24.400 +/- 0.548 | 0.000 +/- 0.000 | 0.855 +/- 0.000 | 0.797 +/- 0.000 | 0.957 +/- 0.001 | 0.956 +/- 0.002 | 6,6,6,6,6 | 0.036 |
| 0.0 | 1.00 | 1.000000 | 0.959 +/- 0.002 | 0.000 +/- 0.000 | 0.004 +/- 0.000 | 0.959 +/- 0.001 | 0.955 +/- 0.001 | 0.011 +/- 0.001 | 0.959 +/- 0.001 | 955.000 +/- 1.225 | 0.000 +/- 0.000 | 0.850 +/- 0.000 | 0.757 +/- 0.001 | 0.971 +/- 0.001 | 0.981 +/- 0.000 | 1,1,1,1,1 | 0.949 |
| 1.0 | 0.00 | 0.100000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.004 +/- 0.000 | 0.038 +/- 0.002 | 0.034 +/- 0.002 | 0.865 +/- 0.003 | 0.829 +/- 0.005 | 33.800 +/- 2.168 | 0.000 +/- 0.000 | 0.798 +/- 0.001 | 0.798 +/- 0.001 | 0.965 +/- 0.001 | 0.968 +/- 0.001 | 6,6,6,6,6 | 0.050 |
| 1.0 | 0.25 | 0.325474 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.003 +/- 0.000 | 0.044 +/- 0.002 | 0.041 +/- 0.002 | 0.866 +/- 0.005 | 0.823 +/- 0.003 | 41.000 +/- 2.000 | 0.000 +/- 0.000 | 0.798 +/- 0.001 | 0.797 +/- 0.000 | 0.965 +/- 0.001 | 0.966 +/- 0.000 | 6,6,6,6,6 | 0.068 |
| 1.0 | 0.50 | 0.550947 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.003 +/- 0.000 | 0.044 +/- 0.006 | 0.041 +/- 0.006 | 0.867 +/- 0.002 | 0.818 +/- 0.007 | 41.000 +/- 6.325 | 0.000 +/- 0.000 | 0.798 +/- 0.001 | 0.797 +/- 0.000 | 0.963 +/- 0.000 | 0.963 +/- 0.000 | 6,6,6,6,6 | 0.065 |
| 1.0 | 0.75 | 0.774526 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.001 +/- 0.000 | 0.039 +/- 0.005 | 0.038 +/- 0.005 | 0.866 +/- 0.005 | 0.820 +/- 0.007 | 38.400 +/- 5.128 | 0.000 +/- 0.000 | 0.798 +/- 0.001 | 0.797 +/- 0.001 | 0.961 +/- 0.001 | 0.958 +/- 0.001 | 6,6,6,6,6 | 0.059 |
| 1.0 | 1.00 | 1.000000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.860 +/- 0.007 | 0.855 +/- 0.009 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.728 +/- 0.015 | 0.702 +/- 0.001 | 0.257 +/- 0.155 | 0.001 +/- 0.001 | 6,6,6,6,6 | 0.000 |

These results are the complete pre-specified matrix. No threshold was
applied after seeing them and no cell was rerun. Differences from the
development pool are descriptive, not a basis for excluding outcomes.

Metadata note: the run JSONs store split=repr_select because the existing
target-move helper drops the evaluation_indices marker from its returned
bundle. The frozen runner loads final_test features/labels before applying
that helper; all saved source-prediction indices exactly match the 1,000
class-0 IDs marked final_test in the manifest. Thus this is a mislabeled
split field, not a development-set evaluation. Original run JSONs were
preserved unchanged. Stored split values: ['repr_select'].
