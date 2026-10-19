# K3 validated-base integration

Base: ad9f11380686edac17edc97856bc83f058329290.

This branch backports reviewed changes without upgrading unrelated operators:

- [Argmax #7300](https://github.com/flagos-ai/FlagGems/pull/7300): valid lanes, first NaN and lowest tie index. Correctness only.
- [MetaX addmm #7302](https://github.com/flagos-ai/FlagGems/pull/7302): retain the accepted legal per-operator tile on C550. Draft compatibility fix; broader vendor performance review remains open. Not a performance claim.

These PRs are unmerged at integration time. K3-specific launch handles and
routing policy live in plugin-FL; this branch has no K3 export, model adapter,
PP code or install-time patching. The wider addmm work in #6807 remains its
author's separate contribution and is not copied here.
