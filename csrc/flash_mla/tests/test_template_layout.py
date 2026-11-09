"""Static guard: the direct-invoke explicit specializations match the upstream tiling-key selection."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEY = (ROOT / "op_kernel" / "arch35" / "flash_mla_template_tiling_key.h").read_text()
DEFS = (ROOT / "op_kernel" / "utils" / "flash_mla_common_def.h").read_text()
ENTRY = (ROOT / "op_kernel" / "flash_mla.cpp").read_text()
CMAKE = (ROOT / "CMakeLists.txt").read_text()


def resolve(token):
    token = token.strip()
    if token.isdigit():
        return int(token)
    for text in (DEFS, KEY):
        match = re.search(rf"#define\s+{re.escape(token)}\s+(\d+)", text)
        if match:
            return int(match.group(1))
    assert False, f"unresolved tiling-key token {token}"


def sel_values(name, macro):
    values = []
    for match in re.finditer(rf"{macro}\({name},\s*ASCENDC_TPL_UI_LIST,\s*([^)]*)\)", KEY):
        values.extend(resolve(x) for x in match.group(1).split(",") if x.strip())
    return values


layouts = set(sel_values("InOutLayoutType", "ASCENDC_TPL_UINT_SEL"))
kvs = set(sel_values("KvLayoutType", "ASCENDC_TPL_UINT_SEL"))
masks = {resolve(x) for m in re.finditer(r"ASCENDC_TPL_BOOL_SEL\(HasAttenMask,\s*([^)]*)\)", KEY)
         for x in m.group(1).split(",") if x.strip()}
configs = set(sel_values("Config", "ASCENDC_TPL_UINT_SEL"))
explicit = {(int(l), int(k), m == "true")
            for l, k, m in re.findall(r"FLASH_MLA_INSTANTIATE\((\d+),\s*(\d+),\s*(false|true)\)", ENTRY)}
expected = {(l, k, bool(m)) for l in layouts for k in kvs for m in masks}


def test_entry_specializations_cover_upstream_selection():
    assert layouts == {0, 1, 2, 4}
    assert kvs == {1, 2, 3}
    assert masks == {0, 1}
    assert configs == {0}
    assert explicit == expected
    assert len(explicit) == 24


def test_one_template_entry_and_two_dtype_compile_variants():
    assert not (ROOT / "vendor").exists()
    assert not list((ROOT / "op_kernel").glob("sgl_flash_mla_*.cpp"))
    assert not (ROOT / "op_kernel" / "flash_mla_impl.h").exists()
    assert ENTRY.count("__global__ __aicore__ void FLASH_MLA_ENTRY") == 1
    sources = re.findall(r"ascendc_library\((flash_mla_\w+_kernel)\s+STATIC\s+\$\{FLASH_MLA_KERNEL_SRC\}\)", CMAKE)
    assert sources == ["flash_mla_bf16_kernel", "flash_mla_fp16_kernel"]
    assert CMAKE.count("${FLASH_MLA_KERNEL_SRC}") == 2
    assert "-DORIG_DTYPE_Q=DT_BF16" in CMAKE
    assert "-DORIG_DTYPE_Q=DT_FLOAT16" in CMAKE
    assert "-DFLASH_MLA_ENTRY=flash_mla_fp16" in CMAKE
