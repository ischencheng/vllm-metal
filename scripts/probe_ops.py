"""Microbench building blocks for a sort-free top-p cutoff in MLX (M5 Pro)."""
import time, mlx.core as mx

def bench(fn, iters=30, warm=5):
    for _ in range(warm): mx.eval(fn())
    ts=[]
    for _ in range(iters):
        t=time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter()-t)*1e3)
    ts.sort(); return ts[len(ts)//2], ts[int(len(ts)*0.9)]

V=151936
for B in (1,16,64):
    mx.random.seed(0)
    x = (mx.random.normal((B,V))*2.5).astype(mx.float32)  # logit-like spread
    x = mx.where(mx.arange(V)[None,:] < 5, x+12, x)        # a few strong tokens
    mx.eval(x)
    argmax = lambda: mx.argmax(x,axis=-1)
    argsort = lambda: mx.argsort(x,axis=-1)
    def softmax_sum(): 
        m=mx.max(x,axis=-1,keepdims=True); return mx.sum(mx.exp(x-m),axis=-1)
    def hist256():
        m=mx.max(x,axis=-1,keepdims=True); d=m-x
        b=mx.minimum((d*8).astype(mx.int32),255)
        idx=(mx.arange(B,dtype=mx.int32)[:,None]*256+b).reshape(-1)
        return mx.zeros((B*256,),dtype=mx.float32).at[idx].add(mx.exp(-d).reshape(-1))
    def hist256_count():
        m=mx.max(x,axis=-1,keepdims=True); d=m-x
        b=mx.minimum((d*8).astype(mx.int32),255)
        idx=(mx.arange(B,dtype=mx.int32)[:,None]*256+b).reshape(-1)
        return mx.zeros((B*256,),dtype=mx.int32).at[idx].add(mx.ones(idx.shape,dtype=mx.int32))
    def masked_sum():  # one bisection step
        t=mx.max(x,axis=-1,keepdims=True)-3.0
        return mx.sum(mx.where(x>t, mx.exp(x-t), 0.0),axis=-1)
    def categorical(): return mx.random.categorical(x,axis=-1)
    print(f"B={B}", {n: "%.3f/%.3f"%bench(f) for n,f in [("argmax",argmax),("categorical",categorical),("softmax_sum",softmax_sum),("masked_sum",masked_sum),("hist256_f32",hist256),("hist256_i32",hist256_count),("argsort",argsort)]}, flush=True)
