import runpy
import flydsl.expr.primitive as primitive
original=primitive._infer_int_tuple_type
counter=0
def diagnostic(value):
 global counter
 counter+=1
 expanded=primitive._expand_int_tuple_leaves(value)
 print('INFER',counter,'input_type',type(value),'expanded_type',type(expanded),'expanded',repr(expanded),flush=True)
 return original(value)
primitive._infer_int_tuple_type=diagnostic
test=runpy.run_path('/var/home/my_aiter/aiter/op_tests/test_flydsl_hc_mix.py')
test['test_correctness'](1,10240,320)
