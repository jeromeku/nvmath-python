import torch
import numpy as np
import sys
import nvmath as nv
from nvmath.bindings import cublasLt as lt
from nvmath.bindings.cublas import Operation, ComputeType  # Operation comes from cublas
from nvmath.linalg._internal.matmul_desc_ifc import DESC_ENUM_SCALAR_ATTR, DESC_ENUM_SCALAR_ATTR_INFO

# ----- problem -----
B, L, D = 8, 128, 512
device = "cuda"
dtype = torch.float16  # example; change as needed

# A: [B, L, D] row-major
A = torch.randn(B, L, D, device=device, dtype=dtype)
# B: [D, D] row-major (broadcast across batches)
Bmat = torch.randn(D, D, device=device, dtype=dtype)
# D: [B, D, L] row-major target (note axes)
Dout = torch.empty(B, D, L, device=device, dtype=dtype)

# raw pointers
ptrA = A.data_ptr()
ptrB = Bmat.data_ptr()
ptrD = Dout.data_ptr()

# leading dims and strides (in elements, then converted to bytes internally)
# row-major: ld = number of columns of the 2D view
ldA = D           # A[b] is (L x D)
ldB = D           # B is (D x D)
ldD = L           # D[b] is (D x L), row-major -> ld = L

# strided batched offsets (in bytes)
elem_size = A.element_size()
strideA = (A.stride(0) * elem_size)  # bytes to next batch of A
# Broadcast B across batches => stride 0 with batch_count = 1
strideB = 0
strideD = (Dout.stride(0) * elem_size)

# 1) create handle
handle = lt.create()
def get_np_data(x, dtype: np.dtype):
    buf = np.zeros((1,), dtype=dtype)
    buf[0] = x
    return buf, buf.ctypes.data, buf.nbytes
    
# 2) layouts (note: data type enum is top-level nv.CudaDataType)
layoutA = lt.matrix_layout_create(int(nv.CudaDataType.CUDA_R_16F), L, D, ldA)

_, row_major, row_major_size = get_np_data(lt.Order.ROW, np.int32)
_, batch_count, batch_count_size = get_np_data(B, np.int32)
_, batch_stride, batch_stride_size = get_np_data(strideA, np.int64)
lt.matrix_layout_set_attribute(layoutA, lt.MatrixLayoutAttribute.ORDER, row_major, row_major_size)
lt.matrix_layout_set_attribute(layoutA, lt.MatrixLayoutAttribute.BATCH_COUNT, batch_count, batch_count_size)
lt.matrix_layout_set_attribute(layoutA, lt.MatrixLayoutAttribute.STRIDED_BATCH_OFFSET, batch_stride, batch_stride_size)

_, batch_countB, batch_count_sizeB = get_np_data(8, np.int32)
_, batch_strideB, batch_stride_sizeB = get_np_data(strideB, np.int64)
layoutB = lt.matrix_layout_create(int(nv.CudaDataType.CUDA_R_16F), D, D, ldB)
lt.matrix_layout_set_attribute(layoutB, lt.MatrixLayoutAttribute.ORDER, row_major, row_major_size)
lt.matrix_layout_set_attribute(layoutB, lt.MatrixLayoutAttribute.BATCH_COUNT, batch_countB, batch_count_sizeB)
lt.matrix_layout_set_attribute(layoutB, lt.MatrixLayoutAttribute.STRIDED_BATCH_OFFSET, batch_strideB, batch_stride_sizeB)

# D is (rows=D, cols=L) in row-major -> B x D x L without a post-transpose
_, col_major, col_major_size = get_np_data(lt.Order.COL, np.int32)
_, batch_countD, batch_count_sizeD = get_np_data(B, np.int32)
_, batch_strideD, batch_stride_sizeD = get_np_data(strideD, np.int64)
layoutD = lt.matrix_layout_create(int(nv.CudaDataType.CUDA_R_16F), L, D, ldD)
lt.matrix_layout_set_attribute(layoutD, lt.MatrixLayoutAttribute.ORDER, col_major, col_major_size)
lt.matrix_layout_set_attribute(layoutD, lt.MatrixLayoutAttribute.BATCH_COUNT, batch_countD, batch_count_sizeD)
lt.matrix_layout_set_attribute(layoutD, lt.MatrixLayoutAttribute.STRIDED_BATCH_OFFSET, batch_strideD, batch_stride_sizeD)

# 3) matmul descriptor
# compute_type: cublasComputeType_t; scale_type: cudaDataType_t for alpha/beta
desc = lt.matmul_desc_create(int(ComputeType.COMPUTE_32F), int(nv.CudaDataType.CUDA_R_32F))  # FP32 accumulate, FP32 scales

def set_desc_attr(desc, _attr: lt.MatmulDescAttribute, value):
    import ctypes
    info = DESC_ENUM_SCALAR_ATTR_INFO.get(_attr.name)
    enum_value, ctype = info
    ctypes_value = ctype(value)
    lt.matmul_desc_set_attribute(
        desc, enum_value, ctypes.addressof(ctypes_value), ctypes.sizeof(ctypes_value)
    )

set_desc_attr(desc, lt.MatmulDescAttribute.TRANSA, Operation.N)
set_desc_attr(desc, lt.MatmulDescAttribute.TRANSB, Operation.N)

# 4) scalars and stream
alpha = torch.tensor(1.0, dtype=torch.float32, device=device)
beta  = torch.tensor(0.0, dtype=torch.float32, device=device)
stream = torch.cuda.current_stream().cuda_stream
# 5) run
# Note: c argument is unused when writing into D directly; pass ptrD for both c and d with same layout if you don't need C separate
lt.matmul(
    int(handle),
    int(desc),
    alpha.data_ptr(),  # alpha pointer
    ptrA, int(layoutA),
    ptrB, int(layoutB),
    beta.data_ptr(),   # beta pointer
    ptrD, int(layoutD),   # C
    ptrD, int(layoutD),   # D (output)
    0,                   # algo (0 => let library pick via heuristics unless you pre-plan)
    0, 0,                # workspace ptr & size
    int(stream),
)

# sanity check against PyTorch (A[B,L,D] @ B[D,D] -> [B,L,D], then view as [B,D,L] w/transpose)
ref = (A @ Bmat)          # [B, L, D]
refDL = ref.transpose(1, 2).contiguous()  # [B, D, L]
assert torch.allclose(Dout, refDL, atol=5e-3, rtol=5e-3)

# cleanup
lt.destroy(handle)
lt.matrix_layout_destroy(layoutA)
lt.matrix_layout_destroy(layoutB)
lt.matrix_layout_destroy(layoutD)
lt.matmul_desc_destroy(desc)
