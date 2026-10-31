"""Independent CPU goldens for source variants absent from the MLA workload."""
import os,itertools
import torch,torch_npu,pytest
torch.set_num_threads(4);torch.npu.set_device(0)
if os.environ.get('SGL_SCATTER_TEST_LIBRARY'):
    torch.ops.load_library(os.environ['SGL_SCATTER_TEST_LIBRARY'])
else:
    import sgl_kernel_npu
call=torch.ops.npu.scatter_pa_kv_cache
def bits(x):return x.view(torch.uint8).cpu().contiguous()

@pytest.mark.parametrize('idx',[torch.int32,torch.int64])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32,torch.int8,torch.int16,torch.int32,torch.uint8,torch.uint16,torch.uint32,torch.float8_e4m3fn,torch.float8_e5m2])
@pytest.mark.parametrize('d',[32,8192])
def test_rope_full_window_mean(dtype,idx,d):
    # Whole-window reduction has an independent unambiguous mathematical golden.
    b,s,h=1,8,1
    base=130 if dtype==torch.uint8 else 33000 if dtype==torch.uint16 else 130000000 if dtype==torch.uint32 else 2
    x=(torch.arange(s,dtype=torch.float64)*2+base).view(1,s,1,1).expand(b,s,h,d).contiguous().to(dtype)
    # Copy before casting reproduces the actual input values for FP8 exactly.
    expected=x.double().mean(dim=1).to(dtype)
    v=x.clone();kc=torch.full((2,128,1,d),1,dtype=dtype);vc=kc.clone()
    kn,vn,kcn,vcn=[t.npu() for t in [x,v,kc,vc]]
    call(kn,kcn,torch.tensor([[7]],dtype=idx,device='npu'),vn,vcn,
         compress_lens=torch.tensor([[s]],dtype=idx,device='npu'),
         compress_seq_offset=torch.tensor([0],dtype=idx,device='npu'),
         seq_lens=torch.tensor([s],dtype=idx,device='npu'),scatter_mode='Rope')
    kc.view(256,1,d)[7]=expected;vc.view(256,1,d)[7]=expected
    assert torch.equal(bits(kcn),bits(kc));assert torch.equal(bits(vcn),bits(vc))

@pytest.mark.parametrize('kdtype,vdtype',[(torch.bfloat16,torch.float32),(torch.float32,torch.int8),(torch.int8,torch.float16),(torch.uint8,torch.bfloat16),(torch.float8_e4m3fn,torch.float8_e5m2),(torch.float16,torch.uint16)])
@pytest.mark.parametrize('idx',[torch.int32,torch.int64])
def test_mixed_nz(kdtype,vdtype,idx):
    n,h,kd,vd,cap,page=129,3,64,128,384,128
    ku=32//torch.empty((),dtype=kdtype).element_size();vu=32//torch.empty((),dtype=vdtype).element_size()
    k=torch.randn(n,h,kd).to(kdtype);v=torch.randn(n,h,vd).to(vdtype)
    kc=torch.full((cap//page,h*kd//ku,page,ku),3,dtype=kdtype)
    vc=torch.full((cap//page,h*vd//vu,page,vu),5,dtype=vdtype)
    ek,ev=kc.clone(),vc.clone();slots=torch.randperm(cap)[:n].to(idx)
    for i,slot in enumerate(slots.tolist()):
        ek[slot//page,:,slot%page]=k[i].reshape(-1,ku)
        ev[slot//page,:,slot%page]=v[i].reshape(-1,vu)
    kn,vn,kcn,vcn,sn=[x.npu() for x in [k,v,kc,vc,slots]]
    call(kn,kcn,sn,vn,vcn,cache_mode='PA_NZ')
    assert torch.equal(bits(kcn),bits(ek));assert torch.equal(bits(vcn),bits(ev))

def test_packed_fp4_bits():
    n,h,d=129,3,64;cap=384
    raw=torch.randint(0,256,(n,h,d),dtype=torch.uint8);cache=torch.full((3,128,h,d),0xA7,dtype=torch.uint8)
    slots=torch.randperm(cap)[:n];expected=cache.clone();expected.view(cap,h,d)[slots]=raw
    x=raw.npu().view(torch.float4_e2m1fn_x2);kc=cache.npu().view(torch.float4_e2m1fn_x2);vc=cache.npu().view(torch.float4_e2m1fn_x2)
    call(x,kc,slots.npu(),x,vc)
    assert torch.equal(bits(kc),expected);assert torch.equal(bits(vc),expected)

@pytest.mark.parametrize('mode',['Alibi','Omni'])
@pytest.mark.parametrize('d',[32,8192])
@pytest.mark.parametrize('idx',[torch.int32,torch.int64])
def test_copy_compression_variants(mode,d,idx):
    # ALiBi keeps the last W tokens; zero-compression Omni is the identity.
    s,h,cap=8,2,256;keep=3 if mode=='Alibi' else s
    x=torch.randn(1,s,h,d,dtype=torch.bfloat16);v=x*2
    kc=torch.full((2,128,1,d),-3,dtype=x.dtype);vc=kc.clone();ek,ev=kc.clone(),vc.clone()
    slots=torch.tensor([[7,127]],dtype=idx)
    for head in range(h):
        slot=slots[0,head].item()
        ek.view(cap,1,d)[slot:slot+keep,0]=x[0,s-keep:,head]
        ev.view(cap,1,d)[slot:slot+keep,0]=v[0,s-keep:,head]
    kn,vn,kcn,vcn,sn=[t.npu() for t in [x,v,kc,vc,slots]]
    lens=torch.full((1,h),keep if mode=='Alibi' else 0,dtype=idx,device='npu')
    call(kn,kcn,sn,vn,vcn,compress_lens=lens,
         compress_seq_offset=None if mode=='Alibi' else torch.zeros(h,dtype=idx,device='npu'),
         seq_lens=torch.tensor([s],dtype=idx,device='npu'),scatter_mode=mode)
    assert torch.equal(bits(kcn),bits(ek));assert torch.equal(bits(vcn),bits(ev))

@pytest.mark.parametrize('d',[32,65536])
@pytest.mark.parametrize('idx',[torch.int32,torch.int64])
def test_nct_overlapping_source_rows(d,idx):
    n,h,cap=129,1,384;stride=d//2;ko,vo=7,5
    k=torch.randn(n,h,d,dtype=torch.float16);v=torch.randn_like(k)
    kc=torch.full((3,128,h,d),3,dtype=k.dtype);vc=kc.clone();ek,ev=kc.clone(),vc.clone()
    slots=torch.randperm(cap)[:n].to(idx)
    for i,slot in enumerate(slots.tolist()):
        ek.view(cap,h,d)[slot]=k.flatten()[i*stride+ko:i*stride+ko+d].reshape(h,d)
        ev.view(cap,h,d)[slot]=v.flatten()[i*stride+vo:i*stride+vo+d].reshape(h,d)
    kn,vn,kcn,vcn,sn=[t.npu() for t in [k,v,kc,vc,slots]]
    call(kn,kcn,sn,vn,vcn,scatter_mode='Nct',strides=[stride,stride],offsets=[ko,vo])
    assert torch.equal(bits(kcn),bits(ek));assert torch.equal(bits(vcn),bits(ev))

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
@pytest.mark.parametrize('idx',[torch.int32,torch.int64])
def test_nz_five_dim_noncontiguous(dtype,idx):
    n,h,d,cap,page=129,3,64,384,128;unit=32//torch.empty((),dtype=dtype).element_size()
    raw=torch.randn(n,h,d+16,dtype=dtype);key=raw[...,:d]
    cache=torch.full((3,h,d//unit,page,unit*2),3,dtype=dtype);expected=cache.clone()
    slots=torch.randperm(cap)[:n].to(idx)
    for i,slot in enumerate(slots.tolist()):expected[slot//page,:,:,slot%page,:unit]=key[i].reshape(h,d//unit,unit)
    xn=raw.npu();kn=xn[...,:d];kc=cache.npu();vc=cache.npu()
    call(kn,kc[...,:unit],slots.npu(),kn,vc[...,:unit],cache_mode='PA_NZ')
    assert torch.equal(bits(kc),bits(expected));assert torch.equal(bits(vc),bits(expected))
