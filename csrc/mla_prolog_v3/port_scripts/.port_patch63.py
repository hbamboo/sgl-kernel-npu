"""Make _WeightHolder an nn.Module so it can replace a child module attribute."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''class _WeightHolder:
    """Minimal stand-in exposing ``.weight`` (what forward_mlaprolog reads)."""

    def __init__(self, weight):
        self.weight = weight'''
NEW = '''class _WeightHolder(torch.nn.Module):
    """Minimal stand-in exposing ``.weight`` (what forward_mlaprolog reads).

    Derives from ``nn.Module`` because it replaces a child module attribute.
    """

    def __init__(self, weight):
        super().__init__()
        self.weight = weight'''

assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("holder -> nn.Module")
