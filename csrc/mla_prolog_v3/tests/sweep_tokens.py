"""Token-boundary sweep: hunt for size-dependent bugs like the PA-stride one at T=192.

Checks every interesting boundary around the kernel's internal tile sizes (128-token PA
blocks, 128-token steps, 16/32-alignment) against the CPU golden.
"""
import json
import os
import subprocess
import sys

TESTS = [
    1, 2, 3, 7, 15, 16, 17, 31, 32, 33,
    63, 64, 65, 127, 128, 129, 130,
    191, 192, 193, 255, 256, 257, 258,
    383, 384, 385, 511, 512, 513, 1000,
]
THRESHOLD = 0.9995
ENV = dict(os.environ)
ENV["ASCEND_RT_VISIBLE_DEVICES"] = ENV.get("ASCEND_RT_VISIBLE_DEVICES", "5")

failures = []
print("%6s %10s %10s %10s %10s" % ("T", "query", "q_rope", "kv_cache", "kr_cache"))
for t in TESTS:
    env = dict(ENV)
    env["T"] = str(t)
    # A subprocess can transiently fail to acquire the device when many are spawned in a
    # row ("rtSetDevice ... device retain error").  Retry only on that, never on a kernel
    # fault (507015) or a numerical failure.
    for attempt in range(3):
        out = subprocess.run([sys.executable, "csrc/mla_prolog_v3/tests/test_glm_mla_prolog_precision.py"],
                             capture_output=True, text=True, env=env)
        text = out.stdout + out.stderr
        device_flake = ("rtSetDevice" in text or "device retain" in text) and "507015" not in text
        if out.returncode == 0 or not device_flake:
            break
        print("  (retry T=%d, device busy)" % t)
    if out.returncode != 0:
        print("%6d FAILED (rc=%d)" % (t, out.returncode))
        failures.append((t, "rc"))
        tail = [ln for ln in text.splitlines() if "Error" in ln or "error" in ln][-2:]
        for ln in tail:
            print("        " + ln[:160])
        continue
    cos = {}
    for line in text.splitlines():
        for key in ("query", "query_rope", "kv_cache", "kr_cache"):
            if line.strip().startswith(key + " ") and "cosine=" in line:
                cos[key] = float(line.split("cosine=")[1].split()[0])
    row = [t] + [cos.get(k, float("nan")) for k in ("query", "query_rope", "kv_cache", "kr_cache")]
    print("%6d %10.6f %10.6f %10.6f %10.6f" % tuple(row))
    bad = [k for k, v in cos.items() if not (v >= THRESHOLD)]
    if len(cos) != 4 or bad:
        failures.append((t, bad))

print()
if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("all %d token sizes >= %.4f on all five outputs" % (len(TESTS), THRESHOLD))
