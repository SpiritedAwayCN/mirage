#define USE_NVSHMEM 0
#include "runtime.h"

__device__ void smem_input_loader_func_0(
    int i,
    cutlass::half_t *dtensor10000008,
    cutlass::half_t *dtensor10000009,
    cutlass::half_t *dtensor10000010) {
  extern __shared__ char smem_buffer[];
  // Load input tensor from device to shared memory
  {
    int tb_offset_row = 64 * blockIdx.z;
    int tb_offset_column = 0;
    int global_offset = 16384 * blockIdx.x;
    cutlass::half_t *stensor_ptr =
        (cutlass::half_t*)(smem_buffer + (i % 2) * 49408 + 0);
    int base_offset = global_offset + tb_offset_row * 64 + tb_offset_column;
    for (int _idx = threadIdx.x * 8; _idx < 4096; _idx += 8 * blockDim.x) {
      unsigned stensor_int_ptr =
          cutlass::arch::cutlass_get_smem_pointer(stensor_ptr + _idx);
      cutlass::half_t *_dtensor_ptr = dtensor10000008 + base_offset + _idx;
      asm volatile(
          "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"::
          "r"(stensor_int_ptr),
          "l"(_dtensor_ptr),
          "n"(16),
          "r"(16));
    } // end of for-loop
  }
  // Load input tensor from device to shared memory
  {
    int tb_offset_row = 0;
    int tb_offset_column = 256 * blockIdx.y + 64 * i;
    int global_offset = 262144 * blockIdx.x;
    cutlass::half_t *stensor_ptr =
        (cutlass::half_t*)(smem_buffer + (i % 2) * 49408 + 8192);
    int base_offset = global_offset + tb_offset_column * 64 + tb_offset_row;
    for (int _idx = threadIdx.x * 8; _idx < 4096; _idx += 8 * blockDim.x) {
      unsigned stensor_int_ptr =
          cutlass::arch::cutlass_get_smem_pointer(stensor_ptr + _idx);
      cutlass::half_t *_dtensor_ptr = dtensor10000009 + base_offset + _idx;
      asm volatile(
          "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"::
          "r"(stensor_int_ptr),
          "l"(_dtensor_ptr),
          "n"(16),
          "r"(16));
    } // end of for-loop
  }
  // Load input tensor from device to shared memory
  {
    int tb_offset_row = 256 * blockIdx.y + 64 * i;
    int tb_offset_column = 0;
    int global_offset = 262144 * blockIdx.x;
    cutlass::half_t *stensor_ptr =
        (cutlass::half_t*)(smem_buffer + (i % 2) * 49408 + 16384);
    int base_offset = global_offset + tb_offset_row * 64 + tb_offset_column;
    for (int _idx = threadIdx.x * 8; _idx < 4096; _idx += 8 * blockDim.x) {
      unsigned stensor_int_ptr =
          cutlass::arch::cutlass_get_smem_pointer(stensor_ptr + _idx);
      cutlass::half_t *_dtensor_ptr = dtensor10000010 + base_offset + _idx;
      asm volatile(
          "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"::
          "r"(stensor_int_ptr),
          "l"(_dtensor_ptr),
          "n"(16),
          "r"(16));
    } // end of for-loop
  }
  asm volatile("cp.async.commit_group;\n" ::);
} // end of smem_input_loader_func_0

__global__ void graphdef_kernel_0(
    cutlass::half_t *dtensor10000008,
    cutlass::half_t *dtensor10000009,
    cutlass::half_t *dtensor10000010,
    cutlass::half_t *dtensor10000011,
    cutlass::half_t *dtensor10000012) {
  extern __shared__ char smem_buffer[];
  smem_input_loader_func_0(
      0,
      dtensor10000008,
      dtensor10000009,
      dtensor10000010);
  for (int i = 0; i < 4; i++) {
    // launch cp.async operators
    if (i + 1 < 4) {
      smem_input_loader_func_0(
          i + 1,
          dtensor10000008,
          dtensor10000009,
          dtensor10000010);
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    // Perform thread-block matmul
    {
      cutlass::half_t *_A_ptr =
          (cutlass::half_t *)(smem_buffer + (i % 2) * 49408 + 0);
      cutlass::half_t *_B_ptr =
          (cutlass::half_t *)(smem_buffer + (i % 2) * 49408 + 8192);
      cutlass::half_t *_C_ptr =
          (cutlass::half_t *)(smem_buffer + 24576);
      //mirage::threadblock::GenericMatmulExecutor<mirage::type::ACT_EXP> executor(
      //_A_ptr, _B_ptr, _C_ptr, 64, 64, 64, threadIdx.x);
    }
    // Perform thread-block matmul
    {
      cutlass::half_t *_A_ptr =
          (cutlass::half_t *)(smem_buffer + 24576);
      cutlass::half_t *_B_ptr =
          (cutlass::half_t *)(smem_buffer + (i % 2) * 49408 + 16384);
      cutlass::half_t *_C_ptr =
          (cutlass::half_t *)(smem_buffer + 32768);
      //mirage::threadblock::GenericMatmulExecutor<mirage::type::ACT_NONE> executor(
      //_A_ptr, _B_ptr, _C_ptr, 64, 64, 64, threadIdx.x);
    }
    // Perform thread-block elementwise reduction
    {
    }
  } // end of for-loop
  // Save output tensor from shared to device memory
  {
    int tb_offset_row = 64 * blockIdx.z;
    int tb_offset_column = 64 * blockIdx.y;
    int global_offset = 262144 * blockIdx.x;
    cutlass::half_t *stensor_ptr =
        (cutlass::half_t*)(smem_buffer + 32768);
    int base_offset = global_offset + tb_offset_row * 1024 + tb_offset_column;
    cutlass::half_t *dtensor_ptr = dtensor10000011 + base_offset;
    int _col = threadIdx.x % 64;
    for (int _row = threadIdx.x / 2; _row < 64; _row += 2) {
      dtensor_ptr[_row * 1024 + _col] = stensor_ptr[_row * 64 + _col];
    } // end of for-loop
  }
  // Save output tensor from shared to device memory
  {
    int tb_offset_row = 64 * blockIdx.z;
    int tb_offset_column = 1 * blockIdx.y;
    int global_offset = 4096 * blockIdx.x;
    cutlass::half_t *stensor_ptr =
        (cutlass::half_t*)(smem_buffer + 40960);
    int base_offset = global_offset + tb_offset_row * 16 + tb_offset_column;
    cutlass::half_t *dtensor_ptr = dtensor10000012 + base_offset;
    int _col = threadIdx.x % 1;
    for (int _row = threadIdx.x / 128; _row < 64; _row += 128) {
      dtensor_ptr[_row * 16 + _col] = stensor_ptr[_row * 1 + _col];
    } // end of for-loop
  }
}

__device__ void smem_input_loader_func_1(
    int i,
    cutlass::half_t *dtensor10000011,
    cutlass::half_t *dtensor10000012) {
  extern __shared__ char smem_buffer[];
  // Load input tensor from device to shared memory
  {
    int tb_offset_row = 16 * blockIdx.y;
    int tb_offset_column = 0;
    int global_offset = 262144 * blockIdx.x;
    cutlass::half_t *stensor_ptr =
        (cutlass::half_t*)(smem_buffer + (i % 2) * 39456 + 0);
    int base_offset = global_offset + tb_offset_row * 1024 + tb_offset_column;
    for (int _idx = threadIdx.x * 8; _idx < 16384; _idx += 8 * blockDim.x) {
      unsigned stensor_int_ptr =
          cutlass::arch::cutlass_get_smem_pointer(stensor_ptr + _idx);
      cutlass::half_t *_dtensor_ptr = dtensor10000011 + base_offset + _idx;
      asm volatile(
          "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"::
          "r"(stensor_int_ptr),
          "l"(_dtensor_ptr),
          "n"(16),
          "r"(16));
    } // end of for-loop
  }
  // Load input tensor from device to shared memory
  {
    int tb_offset_row = 16 * blockIdx.y;
    int tb_offset_column = 0;
    int global_offset = 4096 * blockIdx.x;
    cutlass::half_t *stensor_ptr =
        (cutlass::half_t*)(smem_buffer + (i % 2) * 39456 + 32768);
    int base_offset = global_offset + tb_offset_row * 16 + tb_offset_column;
    for (int _idx = threadIdx.x * 8; _idx < 256; _idx += 8 * blockDim.x) {
      unsigned stensor_int_ptr =
          cutlass::arch::cutlass_get_smem_pointer(stensor_ptr + _idx);
      cutlass::half_t *_dtensor_ptr = dtensor10000012 + base_offset + _idx;
      asm volatile(
          "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"::
          "r"(stensor_int_ptr),
          "l"(_dtensor_ptr),
          "n"(16),
          "r"(16));
    } // end of for-loop
  }
  asm volatile("cp.async.commit_group;\n" ::);
} // end of smem_input_loader_func_1

__global__ void graphdef_kernel_1(
    cutlass::half_t *dtensor10000011,
    cutlass::half_t *dtensor10000012,
    cutlass::half_t *dtensor10000013) {
  extern __shared__ char smem_buffer[];
  smem_input_loader_func_1(
      0,
      dtensor10000011,
      dtensor10000012);
  for (int i = 0; i < 1; i++) {
    // launch cp.async operators
    if (i + 1 < 1) {
      smem_input_loader_func_1(
          i + 1,
          dtensor10000011,
          dtensor10000012);
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    // Perform thread-block elementwise reduction
    {
    }
    // Perform thread-block elementwise reduction
    {
    }
    // Perform thread-block elementwise div
    {
      cutlass::half_t *_in1_ptr =
          (cutlass::half_t *)(smem_buffer + 33280);
      cutlass::half_t *_in2_ptr =
          (cutlass::half_t *)(smem_buffer + 35328);
      cutlass::half_t *_out_ptr =
          (cutlass::half_t *)(smem_buffer + 35360);
      for (int i = 0; i < 1024; i += blockDim.x) {
        _out_ptr[i] = _in1_ptr[i / 1] / _in2_ptr[i / 64];
      }
    }
  } // end of for-loop
  // Save output tensor from shared to device memory
  {
    int tb_offset_row = 16 * blockIdx.y;
    int tb_offset_column = 0;
    int global_offset = 16384 * blockIdx.x;
    cutlass::half_t *stensor_ptr =
        (cutlass::half_t*)(smem_buffer + 35360);
    int base_offset = global_offset + tb_offset_row * 64 + tb_offset_column;
    cutlass::half_t *dtensor_ptr = dtensor10000013 + base_offset;
    int _col = threadIdx.x % 64;
    for (int _row = threadIdx.x / 2; _row < 16; _row += 2) {
      dtensor_ptr[_row * 64 + _col] = stensor_ptr[_row * 64 + _col];
    } // end of for-loop
  }
}

void mugraph_executor(char *gpu_base_ptr) {
  // launching kernel: graphdef_kernel_0
  {
    cutlass::half_t *dtensor10000008 = (cutlass::half_t*)(gpu_base_ptr + 0);
    cutlass::half_t *dtensor10000009 = (cutlass::half_t*)(gpu_base_ptr + 65536);
    cutlass::half_t *dtensor10000010 = (cutlass::half_t*)(gpu_base_ptr + 1114112);
    cutlass::half_t *dtensor10000011 = (cutlass::half_t*)(gpu_base_ptr + 2162688);
    cutlass::half_t *dtensor10000012 = (cutlass::half_t*)(gpu_base_ptr + 3211264);
    dim3 grid_dim = {2, 16, 4}
    dim3 block_dim = {128, 1, 1}
    graphdef_kernel_0<<<grid_dim, block_dim, 73984>>>(
      dtensor10000008, dtensor10000009, dtensor10000010
      , dtensor10000011, dtensor10000012
    );
  }
  // launching kernel: graphdef_kernel_1
  {
    cutlass::half_t *dtensor10000011 = (cutlass::half_t*)(gpu_base_ptr + 2162688);
    cutlass::half_t *dtensor10000012 = (cutlass::half_t*)(gpu_base_ptr + 3211264);
    cutlass::half_t *dtensor10000013 = (cutlass::half_t*)(gpu_base_ptr + 3227648);
    dim3 grid_dim = {2, 16, 1}
    dim3 block_dim = {128, 1, 1}
    graphdef_kernel_1<<<grid_dim, block_dim, 72736>>>(
      dtensor10000011, dtensor10000012
      , dtensor10000013
    );
  }
} // end of mugraph_executer

int main() {
  char *gpu_base_ptrs[16];
  for (int i = 0; i < 1; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&gpu_base_ptrs[i], 2147483648));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  CHECK_CUDA(cudaEventCreate(&events[0]));
  CHECK_CUDA(cudaEventCreate(&events[1]));
  for (int i = 0; i < 1024; i++) {
    mugraph_executor(gpu_base_ptrs[0]);
  }
  CHECK_CUDA(cudaEventRecord(events[0]));
  for (int i = 0; i < 1024; i++) {
    mugraph_executor(gpu_base_ptrs[0]);
  }
  CHECK_CUDA(cudaEventRecord(events[1]));
  CHECK_CUDA(cudaEventSynchronize(events[1]));
  float runtime_ms;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  printf("Mugraph runtime = %.8lfms\n", runtime_ms / 1024);
}
