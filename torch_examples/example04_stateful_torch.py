# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example illustrates the use of stateful matrix multiplication objects. Stateful objects
amortize the cost of preparation across multiple executions.

The inputs as well as the result are PyTorch tensors on the GPU.
"""

import torch
import sys
import nvmath
import logging
from cuda.core.experimental import Device
from nvmath.bindings import cublasLt as cublaslt
from nvmath.linalg._internal import matmul_desc_ifc, matmul_pref_ifc, matrix_layout_ifc
from nvmath.linalg._internal.utils import get_handle, pointer_aligned_to
from nvmath.internal.utils import package_wrapper, get_memory_limit_from_device_id
from nvmath.bindings.cublas import Operation, ComputeType
from nvmath.linalg._internal.typemaps import cudaDataType, SCALE_TYPE_TO_DEFAULT_COMPUTE_TYPE
from nvmath.linalg.advanced import MatmulPlanPreferences, MatmulOptions
from nvmath.linalg._internal import algo_cap_ifc, algo_config_ifc
from nvmath.linalg.advanced import _algorithmmod

import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s::%(levelname)s:: %(pathname)s:%(lineno)d: %(message)s",
    datefmt="%H:%M:%S",
)
current_stream = torch.cuda.current_stream()

def create_mm_desc(compute_type: ComputeType, scale_type: cudaDataType):
    mm_desc = cublaslt.matmul_desc_create(compute_type, scale_type)

    mm_desc_ifc = matmul_desc_ifc.MatmulDescInterface(mm_desc)
    mm_desc_ifc.compute_type = compute_type
    mm_desc_ifc.scale_type = scale_type
    
    return mm_desc, mm_desc_ifc


# Create a wrapper class that implements __cuda_stream__
class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python


# Prepare sample input data
device_id = 0
bs, seqlen, d = 2, 4096, 2048
dtype = torch.bfloat16
A = torch.rand(bs, seqlen, d, device=device_id, dtype=dtype)
B = torch.rand(d, d, device=device_id, dtype=dtype)
D = torch.empty(bs, d, seqlen, dtype=dtype, device=device_id)
alpha = np.zeros((1,), dtype=np.float32)
alpha[0] = 1
beta = np.zeros((1,), dtype=np.float32)

_dtype = cudaDataType.CUDA_R_16BF
ALIGNMENT_BYTES = 256

# Use the stateful object as a context manager to automatically release resources.
mm = nvmath.linalg.advanced.Matmul(A, B)

assert mm.compute_type == ComputeType.COMPUTE_32F
assert mm.scale_type == cudaDataType.CUDA_R_32F

# desc_ifc = mm.mm_desc_ifc
# Create handle
lt_handle = get_handle(device_id=device_id)
desc, desc_ifc = create_mm_desc(mm.compute_type, mm.scale_type)
assert desc_ifc.matmul_desc == desc
desc_ifc.TRANSA = Operation.N
desc_ifc.TRANSB = Operation.N

ldA = A.shape[-1]
batch_offset_A = A.stride(0)
assert ldA == d

ldB = B.shape[-1]
batch_offset_B = 0

ldD = D.shape[-1]
assert ldD == seqlen
batch_offset_D = D.stride(0)

a_layout_ptr = cublaslt.matrix_layout_create(_dtype, rows=seqlen, cols=d, ld=ldA)
b_layout_ptr = cublaslt.matrix_layout_create(_dtype, rows=d, cols=d, ld=ldB)

d_layout_ptr = cublaslt.matrix_layout_create(_dtype, rows=seqlen, cols=d, ld=ldD) # Note D will be COL_MAJOR hence will be transposed

c_layout_ptr = cublaslt.matrix_layout_create(_dtype, rows=seqlen, cols=d, ld=ldD)
# c_layout_ptr = d_layout_ptr # reuse since no C

layout_a_ifc = matrix_layout_ifc.MatrixLayoutInterface(a_layout_ptr)
layout_a_ifc.order = cublaslt.Order.ROW
layout_a_ifc.batch_count = bs
layout_a_ifc.strided_batch_offset = batch_offset_A

layout_b_ifc = matrix_layout_ifc.MatrixLayoutInterface(b_layout_ptr)
layout_b_ifc.order = cublaslt.Order.ROW
layout_b_ifc.batch_count = bs
layout_b_ifc.strided_batch_offset = batch_offset_B

layout_d_ifc = matrix_layout_ifc.MatrixLayoutInterface(d_layout_ptr)
layout_d_ifc.order = cublaslt.Order.COL
layout_d_ifc.batch_count = bs
layout_d_ifc.strided_batch_offset = batch_offset_D

layout_c_ifc = matrix_layout_ifc.MatrixLayoutInterface(c_layout_ptr)
layout_c_ifc.order = cublaslt.Order.COL
layout_c_ifc.batch_count = bs
layout_c_ifc.strided_batch_offset = batch_offset_D

breakpoint()
options = MatmulOptions()
limit = 8
preferences = MatmulPlanPreferences(limit=limit)
algorithms_buffer = cublaslt.MatmulHeuristicResult(limit)
num_algorithms = np.empty((1,), dtype=np.int32)

preference_ptr = cublaslt.matmul_preference_create()
preference_ifc = matmul_pref_ifc.MatmulPreferenceInterface(preference_ptr)
memory_limit = r"80%"
preference_ifc.max_workspace_bytes = get_memory_limit_from_device_id(memory_limit, device_id)
preference_ifc.reduction_scheme_mask = preferences.reduction_scheme_mask
preference_ifc.max_waves_count = preferences.max_waves_count
preference_ifc.impl_mask = preferences.numerical_impl_mask
ALIGNMENT_A = min(ALIGNMENT_BYTES, pointer_aligned_to(A.data_ptr()))
ALIGNMENT_B = min(ALIGNMENT_BYTES, pointer_aligned_to(B.data_ptr()))

cublaslt.matmul_algo_get_heuristic(
    lt_handle,
    desc,
    a_layout_ptr,
    b_layout_ptr,
    c_layout_ptr,
    d_layout_ptr,
    preference_ptr,
    limit,
    algorithms_buffer.ptr,
    num_algorithms.ctypes.data,
)
num_algorithms = num_algorithms[0]

assert num_algorithms > 0, "No valid algos"
algorithms_buffer = algorithms_buffer[:num_algorithms]
best_algorithm_struct = algorithms_buffer[0]["algo"]
workspace_size = int(np.max(algorithms_buffer["workspace_size"]))
algorithm_objects = tuple(_algorithmmod.Algorithm(a) for a in algorithms_buffer)
workspace = torch.empty((workspace_size,), dtype=torch.uint8, device=device_id)

cublaslt.matmul(
    lt_handle,
    desc,
    alpha.ctypes.data,
    A.data_ptr(),
    a_layout_ptr,
    B.data_ptr(),
    b_layout_ptr,
    beta.ctypes.data,
    0,
    c_layout_ptr,
    D.data_ptr(),
    d_layout_ptr,
    best_algorithm_struct.ctypes.data,
    workspace.data_ptr(),
    workspace_size,
    current_stream.cuda_stream,
)
current_stream.synchronize()
ref = (A @ B).transpose(1, 2).contiguous()
EXPECTED_SHAPE = torch.Size([bs, d, seqlen])
assert ref.shape == EXPECTED_SHAPE
assert D.shape == EXPECTED_SHAPE
diff = (ref - D).abs().max()

print(f"{diff.item():.4f}")
print(ref.view(-1)[:10], ref.view(-1)[-10:])
print(D.view(-1)[:10], D.view(-1)[-10:])
