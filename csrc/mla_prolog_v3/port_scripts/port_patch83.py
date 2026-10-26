"""Fix: the read and write paths declared two separate thread_local arrays."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

# hoist the ring next to FastEntry
OLD1 = """constexpr size_t kFastEntries = 16;"""
NEW1 = """constexpr size_t kFastEntries = 16;

// One ring per thread: serving threads are long lived and a launch repeats with the same
// shapes thousands of times.
thread_local FastEntry g_fastEntries[kFastEntries];
thread_local size_t g_fastNext = 0;"""
assert s.count(OLD1) == 1
s = s.replace(OLD1, NEW1)

OLD2 = """    {
        static thread_local FastEntry entries[kFastEntries];
        static thread_local size_t nextSlot = 0;
        for (size_t i = 0; i < kFastEntries; ++i) {
            FastEntry &e = entries[i];"""
NEW2 = """    {
        for (size_t i = 0; i < kFastEntries; ++i) {
            FastEntry &e = g_fastEntries[i];"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)

OLD3 = """            return;
        }
        nextSlot = (nextSlot + 1) % kFastEntries;
    }"""
NEW3 = """            return;
        }
    }"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)

OLD4 = """    FastEntry *slotToFill = nullptr;
    {
        static thread_local FastEntry entries[kFastEntries];
        static thread_local size_t nextSlot = 0;
        slotToFill = &entries[nextSlot];
        nextSlot = (nextSlot + 1) % kFastEntries;
        slotToFill->valid = false;
    }"""
NEW4 = """    FastEntry *slotToFill = &g_fastEntries[g_fastNext];
    g_fastNext = (g_fastNext + 1) % kFastEntries;
    slotToFill->valid = false;"""
assert s.count(OLD4) == 1
s = s.replace(OLD4, NEW4)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("ring shared between read and write")
