"""Lazy optional companion for stable raw-byte E64/K4 routing on Ascend950."""
from functools import lru_cache
import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load():
    library = Path(__file__).resolve().parent / "lib" / "libsgl_kernel_npu_moe_stable.so"
    if not library.is_file():
        return False
    torch.ops.load_library(str(library))
    logger.info(
        "Loaded stable MoE companion: module=%s library=%s sha256=%s",
        __file__, library, hashlib.sha256(library.read_bytes()).hexdigest(),
    )
    return True


def route_prequantized(x, ids, scale):
    if not _load():
        return None
    # Preserve the qualified library's exact ABI and namespace.
    return torch.ops.moe_stable_proof.route_fp8(x, ids, scale, 8)
