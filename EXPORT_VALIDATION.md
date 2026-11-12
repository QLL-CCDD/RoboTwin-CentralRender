# Export verification — 2026-11-13

The existing analyzer was run against the exported raw traces, without any GPU run.
It reproduced formal throughput gain 0.20718266804280594 and total wall reduction
-0.003163034877215587. Action sequences match in both the selected pair and the
later native sensitivity run. Formal wall values reproduce exactly:
6.831209547817707 / 5.658803533762693 / 6.031630117446184 seconds.

All ten source/model-metadata hashes listed in closed-loop-provenance.json match
the exported bytes. Archive traversal and file sizes were checked; credential
patterns were checked in text files and recursively inside the seed archive/wheel.
No GPU experiment was added during export.

To reproduce analysis on a machine with Python (no GPU/model required):

```bash
python analyze_policy_pair.py \
  evidence/policy-cloud/closed-loop/native-k2g1-r2 \
  evidence/policy-cloud/closed-loop/central-k2g1-r3 \
  --sensitivity-native evidence/policy-cloud/closed-loop/native-k2g1-r3 \
  --output /tmp/robotwin-comparison.json
```

The output embeds absolute evidence paths, so its bytes can differ by checkout
location while the numerical results remain the same.
