# Homogeneous FedRS attacker, locked before the scored run

Date: 2026-10-30. This check was specified before any scored homogeneous run.

Question. In the published missing-class attack, client 0 rewrites class 0 to class 1 after warmup and sets its own scale to 1, including on class 0, which is then absent. What happens on the same partition if client 0 instead keeps nominal rho 0 and the scale is recomputed from the rewritten labels?

Fixed setup. Fashion-MNIST linear head, the same frozen trunk, the same ten-client partition, pair 0 to 1 only, honest rho 0, rounds 30, warmup 10, seeds 1101, 1207, 1309, 2101, 2207. Two move fractions only: 0 and 1. Evaluation is repr_select, 200 images per class. No rho 1 run, no other pairs, no MNIST, no official test, and no edit of an archived verdict.

Attacker rule. On flipped rounds client 0 rewrites local labels 0 to 1 and passes rho 0 into the local epoch. class_scales then sets a present class to 1 and an absent class to 0. After the rewrite, class 0 is absent, so its scale is 0 at every client. The runner aborts unless that scale is 0 and the class-1 scale is 1.

Outputs. JSON only under homogeneous-fedrs/runs/. Each head is trained twice and the two score dictionaries must match.
