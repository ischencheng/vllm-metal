"""Where does mx.random.categorical spend its time on a full vocab row batch?"""
import time, mlx.core as mx

def bench(fn, iters=30, warm=5):
    for _ in range(warm): mx.eval(fn())
    ts=[]
    for _ in range(iters):
        t=time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter()-t)*1e3)
    ts.sort(); return "%.3f"%ts[len(ts)//2]

V=151936
key=mx.random.key(0)
for B in (1,16,64):
    x=(mx.random.normal((B,V))*2.5).astype(mx.float32); mx.eval(x)
    r={}
    r["argmax"]=bench(lambda: mx.argmax(x,axis=-1))
    r["uniform"]=bench(lambda: mx.random.uniform(shape=(B,V),key=key))
    r["gumbel"]=bench(lambda: mx.random.gumbel(shape=(B,V),key=key))
    r["categorical"]=bench(lambda: mx.random.categorical(x,axis=-1,key=key))
    r["argsort"]=bench(lambda: mx.argsort(x,axis=-1))
    def sorted_path():
        idx=mx.argsort(x,axis=-1)[..., ::-1]; s=mx.take_along_axis(x,idx,axis=-1)
        pos=mx.random.categorical(s,axis=-1,key=key); return mx.take_along_axis(idx,pos[:,None],axis=-1)
    r["main_temp_only"]=bench(sorted_path)
    print(f"B={B}", r, flush=True)
