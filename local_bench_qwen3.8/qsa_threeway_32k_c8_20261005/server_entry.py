import os, runpy, inspect, textwrap, hashlib
import sglang.srt.layers.attention.qwen_sparse_attn_backend as backend
arm = os.environ['QSA_EXPERIMENT_ARM']
assert arm in ('ck', 'triton', 'flydsl')
assert os.environ['SGLANG_AITER_QSA_PA_DECODE'] == ('1' if arm == 'flydsl' else '0')
cls = backend.QwenSparseAttnBackend
if arm == 'ck':
    source = textwrap.dedent(inspect.getsource(cls._forward_paged_attention))
    assert source.count('if is_hip():') == 1
    source = source.replace('if is_hip():', 'if False:  # Experiment: bypass ROCm Triton fallback and use CK below')
    exec(compile(source, __file__ + ':ck_override', 'exec'), backend.__dict__)
    cls._forward_paged_attention = backend._forward_paged_attention
    print('QSA_THREEWAY_CK_OVERRIDE sha256=' + hashlib.sha256(source.encode()).hexdigest(), flush=True)
    original_resolver = backend._resolve_flash_attn_varlen_func
    def resolve_ck_once():
        original_ck = original_resolver()
        def log_ck(*args, **kwargs):
            print('QSA_THREEWAY_EXECUTING ck_flash_attn_varlen_func', flush=True)
            backend._resolve_flash_attn_varlen_func = original_resolver
            return original_ck(*args, **kwargs)
        return log_ck
    backend._resolve_flash_attn_varlen_func = resolve_ck_once
original_triton = backend.sparse_gqa_packed_decode_triton
def log_triton(*args, **kwargs):
    assert arm == 'triton', 'Unexpected Triton dispatch in ' + arm
    print('QSA_THREEWAY_EXECUTING sparse_gqa_packed_decode_triton', flush=True)
    backend.sparse_gqa_packed_decode_triton = original_triton
    return original_triton(*args, **kwargs)
backend.sparse_gqa_packed_decode_triton = log_triton
original_flydsl = cls._forward_flydsl_sparse
def log_flydsl(self, *args, **kwargs):
    assert arm == 'flydsl', 'Unexpected FlyDSL dispatch in ' + arm
    print('QSA_THREEWAY_EXECUTING _forward_flydsl_sparse', flush=True)
    cls._forward_flydsl_sparse = original_flydsl
    return original_flydsl(self, *args, **kwargs)
cls._forward_flydsl_sparse = log_flydsl
if __name__ == '__main__':
    print('QSA_THREEWAY_ARM=' + arm + ' source=' + backend.__file__, flush=True)
    runpy.run_module('sglang.launch_server', run_name='__main__')
