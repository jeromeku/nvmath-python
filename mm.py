import torch
import numpy as np
import nvmath.bindings.cublasLt as lt
import torch
import numpy as np

import nvmath.bindings.cublasLt as lt
from nvmath.bindings.cublasLt import MatrixLayoutAttribute as MLA, Order, MatmulDescAttribute
from nvmath.bindings.cublas import Operation  # <-- use cuBLAS Operation enum

def batched_matmul_row_to_rowDL_torch(A: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
    """
    A:    [B, L, D]  row-major (contiguous), CUDA
    Bmat: [D, D]     row-major (contiguous), CUDA
    return D_out: [B, D, L] row-major (no transpose kernel)
    """
    assert A.is_cuda and Bmat.is_cuda
    assert A.ndim == 3 and Bmat.ndim == 2
    B, L, D = A.shape
    assert Bmat.shape == (D, D)
    assert A.dtype in (torch.float16, torch.bfloat16) and Bmat.dtype == A.dtype

    D_out = torch.empty((B, D, L), device=A.device, dtype=A.dtype)

    # raw ptrs
    ptrA, ptrB, ptrD = A.data_ptr(), Bmat.data_ptr(), D_out.data_ptr()

    # leading dims (elements)
    ldA, ldB, ldD = D, D, L

    # batch strides (elements)
    strideA = L * D
    strideB = 0                # broadcast B across batches
    strideD = L * D

    # handle + matmul descriptor
    handle = lt.create()
    desc = lt.matmul_desc_create(lt.CudaDataType.R_32F)  # scales in FP32, compute FP32
    lt.matmul_desc_set_attribute(desc, MatmulDescAttribute.COMPUTE_TYPE,
                                 np.int32(lt.MatmulComputeType.COMPUTE_32F))
    # transpose flags belong to the descriptor; use cuBLAS Operation enum values
    lt.matmul_desc_set_attribute(desc, MatmulDescAttribute.TRANSA,
                                 np.int32(int(Operation.NON_TRANSPOSE)))
    lt.matmul_desc_set_attribute(desc, MatmulDescAttribute.TRANSB,
                                 np.int32(int(Operation.NON_TRANSPOSE)))

    # dtype enum
    cdtype = {torch.float16: lt.CudaDataType.R_16F,
              torch.bfloat16: lt.CudaDataType.R_16BF}[A.dtype]

    # A: ROW-major [L x D], batched
    layoutA = lt.matrix_layout_create(cdtype, L, D, ldA)
    lt.matrix_layout_set_attribute(layoutA, MLA.ORDER, np.int32(Order.ROW))
    lt.matrix_layout_set_attribute(layoutA, MLA.BATCH_COUNT, np.int32(B))
    lt.matrix_layout_set_attribute(layoutA, MLA.BATCH_STRIDE, np.int64(strideA))

    # B: ROW-major [D x D], broadcast
    layoutB = lt.matrix_layout_create(cdtype, D, D, ldB)
    lt.matrix_layout_set_attribute(layoutB, MLA.ORDER, np.int32(Order.ROW))
    lt.matrix_layout_set_attribute(layoutB, MLA.BATCH_COUNT, np.int32(1))
    lt.matrix_layout_set_attribute(layoutB, MLA.BATCH_STRIDE, np.int64(strideB))

    # D: describe as COL-major [L x D] (== row-major [D x L] byte layout)
    layoutD = lt.matrix_layout_create(cdtype, L, D, ldD)
    lt.matrix_layout_set_attribute(layoutD, MLA.ORDER, np.int32(Order.COL))
    lt.matrix_layout_set_attribute(layoutD, MLA.BATCH_COUNT, np.int32(B))
    lt.matrix_layout_set_attribute(layoutD, MLA.BATCH_STRIDE, np.int64(strideD))

    alpha, beta = np.float32(1.0), np.float32(0.0)
    stream = torch.cuda.current_stream(device=A.device).cuda_stream

    lt.matmul(handle, desc,
              alpha, ptrA, layoutA,
                     ptrB, layoutB,
              beta,  0,     None,          # no explicit C
                     ptrD, layoutD,
              None,  stream, 0, 0)

    torch.cuda.synchronize(device=A.device)
    return D_out

if __name__ == "__main__":
    bs = 8
    seqlen = 4096
    hidden_dim = 1024
    dtype = torch.bfloat16
    device = "cuda:0"
    A = torch.randn(bs, seqlen, hidden_dim, dtype=dtype, device=device)
    B = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device)
    D = A @ B
    D = D.transpose(2, 1).contiguous()
    EXPECTED_SHAPE = torch.Size([bs, hidden_dim, seqlen])
    assert D.shape == EXPECTED_SHAPE, f"expected {EXPECTED_SHAPE}, got {D.shape}"

    cublas_out = batched_matmul_row_to_rowDL_torch(A, B)
    assert cublas_out.shape == EXPECTED_SHAPE 