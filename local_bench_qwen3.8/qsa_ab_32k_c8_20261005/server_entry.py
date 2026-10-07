import os, runpy
import sglang.srt.layers.attention.qwen_sparse_attn_backend as backend
original_support = backend.flydsl_qsa_pa_decode_supported
def logged_support(q, k, page):
    result = original_support(q, k, page)
    print(f"QSA_AB_DISPATCH flag={os.environ.get('SGLANG_AITER_QSA_PA_DECODE')} eligible={result} q_shape={tuple(q.shape)} q_dtype={q.dtype} kv_shape={tuple(k.shape)} page={page} source={backend.__file__}", flush=True)
    backend.flydsl_qsa_pa_decode_supported = original_support
    return result
backend.flydsl_qsa_pa_decode_supported = logged_support
cls = backend.QwenSparseAttnBackend
original_flydsl = cls._forward_flydsl_sparse
def logged_flydsl(self, *args, **kwargs):
    print("QSA_AB_EXECUTING _forward_flydsl_sparse", flush=True)
    cls._forward_flydsl_sparse = original_flydsl
    return original_flydsl(self, *args, **kwargs)
cls._forward_flydsl_sparse = logged_flydsl
if __name__ == '__main__':
    runpy.run_module('sglang.launch_server', run_name='__main__')
