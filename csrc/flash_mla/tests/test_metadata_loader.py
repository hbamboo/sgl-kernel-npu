"""A serving process may load another vendor with the same op-api SONAME."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

def test_metadata_after_unrelated_vendor_loaded(tmp_path):
    compiler=shutil.which("cc")
    if not compiler:pytest.skip("A C compiler is needed for the loader fixture")
    library=tmp_path/"libcust_opapi.so"
    subprocess.run([compiler,"-shared","-fPIC","-Wl,-soname,libcust_opapi.so","-x","c","-o",str(library),"-"],input="int unrelated_vendor(void) { return 1; }",text=True,check=True)
    code="""
import ctypes,sys
foreign=ctypes.CDLL(sys.argv[1],mode=ctypes.RTLD_GLOBAL)
assert not hasattr(ctypes.CDLL('libcust_opapi.so'),'aclnnFlashMlaMetadata')
import torch,torch_npu
from sgl_kernel_npu import flash_mla_metadata
seq=torch.tensor([128],device='npu',dtype=torch.int32)
metadata=flash_mla_metadata(seq,20,1,max_seqlen_q=1,max_seqlen_kv=128)
torch.npu.synchronize()
assert metadata.dtype==torch.int32 and metadata.numel()==4096
print('metadata loader collision regression PASS')
"""
    subprocess.run([sys.executable,"-c",code,str(library)],check=True,timeout=120,env=os.environ.copy())
