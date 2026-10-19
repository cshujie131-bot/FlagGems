# DSV4 C550 deployment dependency

Base: upstream 3123859968915e361e8452dd796dc6b27c956324 (5.0.2).
This branch preserves the already evaluated deployment compatibility layer.
It is not an additional upstream operator-performance PR.

- topk_softplus_sqrt: log1p numerical behavior and nonfinite index safety.
- ops/dsv4_masked_fill.py: existing row-broadcast in-place implementation;
  selected explicitly by FL, not a new global ATen override.
- runtime/metax_launch_policy.py: explicit opt-in mm/addmm configuration policy,
  preserving the accepted initialization order.

The FL example pins an immutable commit from this branch. Installation uses
ordinary package builds; there is no patch application during installation.
DS model smoke and historical-evaluation provenance are recorded by the FL PR.
