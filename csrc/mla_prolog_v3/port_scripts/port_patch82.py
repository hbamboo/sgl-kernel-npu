"""Fix SureShapeOpt: an absent optional must match any zero-element shape.

The cached entry stores {0} for an empty output (DimsOf of a 1-D empty tensor), but the
fast path passes nullopt for those, so the comparison always failed and the fast path
never hit.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """bool SameShapeOpt(const std::vector<int64_t> &dims, const c10::optional<at::Tensor> &t)
{
    if (!t.has_value()) {
        return dims.empty();
    }
    return SameShape(dims, *t);
}"""
NEW = """bool SameShapeOpt(const std::vector<int64_t> &dims, const c10::optional<at::Tensor> &t)
{
    if (!t.has_value()) {
        // absent tensor in this call: matches any cached zero-element shape (empty
        // outputs are stored as e.g. {0} while the call passes nullopt)
        int64_t elems = 1;
        for (int64_t d : dims) {
            elems *= d;
        }
        return elems == 0;
    }
    return SameShape(dims, *t);
}"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("SameShapeOpt fixed")
