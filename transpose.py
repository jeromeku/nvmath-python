import torch
from torch._inductor import config
from triton.testing import do_bench
config.trace.enabled = True

bs, seqlen, d = 8, 4096, 2048
D = torch.randn(bs, d, seqlen, dtype=torch.bfloat16, device="cuda:0")

def transpose(x: torch.Tensor):
    return x.transpose(1, 2).contiguous()

compiled_transpose = torch.compile(transpose)
ref_t = do_bench(lambda : transpose(D))
test_t = do_bench(lambda: compiled_transpose(D))
print(f"{ref_t:.4f} vs {test_t:.4f}")