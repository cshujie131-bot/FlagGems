# MiniMax vLLM 0.24 integration
Base: flagos-ai/FlagGems a7620cc191a0b42e040194622c5758b22a7a25dc.
Selective backport from 22f54d03b6e983cade7e67b3dac89c3a6312884f:
https://github.com/flagos-ai/FlagGems/pull/7320 (not merged).
Only shared Linear traversal arguments and the untransposed GEMV primitive are included.
The older base's generic Linear fallback is preserved; the newer MetaX backend and its
default shape selection are deliberately not transplanted. FL owns MiniMax TP shapes,
fixed launch parameters and ATen registration. No global tuner/compiler changes.
