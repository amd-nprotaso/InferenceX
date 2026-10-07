import runpy
import flydsl.expr.primitive as p
from flydsl._mlir import ir
from flydsl.expr.numeric import Numeric
original=p._expand_int_tuple_leaves
count=0
def normalized(value):
 global count
 if isinstance(value,Numeric) and isinstance(value.value,ir.Value) and isinstance(value.value.type,ir.IndexType):
  count+=1
  return original(value.value)
 return original(value)
p._expand_int_tuple_leaves=normalized
t=runpy.run_path('/var/home/my_aiter/aiter/op_tests/test_flydsl_hc_mix.py')
for m in [1,4]:
 t['test_correctness'](m,10240,320)
 print('PASS HC correctness M=',m,'normalized index values=',count,flush=True)
