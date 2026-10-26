# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Validated FP32 projection tuning region; M is total flattened rows."""

FP32_SHORT_MAX_ROWS = 1024
FP32_SHORT_FEATURES = frozenset(
    {(3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096)}
)
