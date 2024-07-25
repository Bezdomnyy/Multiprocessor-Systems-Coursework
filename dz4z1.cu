#include <stdio.h>
#include <stdlib.h>

__host__ __device__ void divisor_count_and_sum(unsigned int n, unsigned int* pcount,
                           unsigned int* psum) {
    unsigned int divisor_count = 1;
    unsigned int divisor_sum = 1;
    unsigned int power = 2;
    for (; (n & 1) == 0; power <<= 1, n >>= 1) {
        ++divisor_count;
        divisor_sum += power;
    }
    for (unsigned int p = 3; p * p <= n; p += 2) {
        unsigned int count = 1, sum = 1;
        for (power = p; n % p == 0; power *= p, n /= p) {
            ++count;
            sum += power;
        }
        divisor_count *= count;
        divisor_sum *= sum;
    }
    if (n > 1) {
        divisor_count *= 2;
        divisor_sum *= n + 1;
    }
    *pcount = divisor_count;
    *psum = divisor_sum;
}

#define BLOCK_SIZE 1024

__shared__ unsigned int arithmetic_count[BLOCK_SIZE];
__shared__ unsigned int composite_count[BLOCK_SIZE];

__global__ void arithmetic_kernel(unsigned int* gpu_arithmetic_count, unsigned int* gpu_composite_count)
{

    unsigned int divisor_count;
    unsigned int divisor_sum;

    // calculate my id
    unsigned int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
    // find the divisor count and the divisor sum
    divisor_count_and_sum(1 + myIdx, &divisor_count, &divisor_sum);
    // initialize the result variable to 0
    unsigned int set = 0;
    // determine whether the number is arithmetic and whether to record the result
    set = divisor_sum % divisor_count == 0 ? 1 : 0;
    // set = myIdx <= n ? set : 0; 
    arithmetic_count[threadIdx.x] = set;
    composite_count [threadIdx.x] = divisor_count > 2 ? set : 0;

    // // reduce the number of arithmetic numbers found per block
    // // sync threads
    __syncthreads();

    // // reduce the results in shared memory
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            arithmetic_count[threadIdx.x] = arithmetic_count[threadIdx.x] +
                                            arithmetic_count[threadIdx.x + s];
            composite_count [threadIdx.x] = composite_count [threadIdx.x] +
                                            composite_count [threadIdx.x + s];
        }
        __syncthreads();
    }

    // put the reduction result in global memory
    if (threadIdx.x == 0) {
        gpu_arithmetic_count[blockIdx.x] = arithmetic_count[0]; //arithmetic_count[0];
        gpu_composite_count[blockIdx.x] = composite_count[0]; // composite_count[0];
    }

}

int main(int argc, char** argv) {
    int num = atoi(argv[1]);
    unsigned int arithmetic_count = 0;
    unsigned int composite_count = 0;
    unsigned int n;

    // // calculate gpu grid and block dimensions
    // dim3 dimGrid((num * 1.23 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // dim3 dimBlock(BLOCK_SIZE);

    // calculate gpu grid and block dimensions
    dim3 dimGrid((num * 1.25 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE);

    // declare pointers to cpu and gpu buffers
    unsigned int *cpu_arithmetic_count, *cpu_composite_count;
    unsigned int *gpu_arithmetic_count, *gpu_composite_count;

    // allocate cpu buffers
    unsigned int bytes = dimGrid.x * sizeof(unsigned int);
    cpu_arithmetic_count = (unsigned int*)malloc(bytes);
    cpu_composite_count  = (unsigned int*)malloc(bytes);

    // allocate gpu buffers
    cudaMalloc(&gpu_arithmetic_count, bytes);
    cudaMalloc(&gpu_composite_count , bytes);

    // run kernel
    arithmetic_kernel
    <<<dimGrid, dimBlock>>>
    (gpu_arithmetic_count, gpu_composite_count);

    // copy the results
    cudaMemcpy(cpu_arithmetic_count, gpu_arithmetic_count, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_composite_count , gpu_composite_count , bytes, cudaMemcpyDeviceToHost);

    // tally the amounts
    n = 0;
    for (int i = 0; i < dimGrid.x && arithmetic_count + BLOCK_SIZE <= num; i++) {
        //printf("[%d] arithmetic count: %u, composite count: %u\n", i, cpu_arithmetic_count[i], cpu_composite_count[i]);
        arithmetic_count += cpu_arithmetic_count[i];
        composite_count  += cpu_composite_count [i];
        n += BLOCK_SIZE;
    }

    ++n;

    for (; arithmetic_count <= num; ++n) {
        unsigned int divisor_count;
        unsigned int divisor_sum;
        divisor_count_and_sum(n, &divisor_count, &divisor_sum);
        if (divisor_sum % divisor_count != 0)
            continue;
        ++arithmetic_count;
        if (divisor_count > 2)
            ++composite_count;
    }

    // free allocated memory
    free    (cpu_arithmetic_count); free    (cpu_composite_count);
    cudaFree(gpu_arithmetic_count); cudaFree(gpu_composite_count);

    // print the results
    printf("\n%uth arithmetic number is %u\n", arithmetic_count, n);
    printf("Number of composite arithmetic numbers <= %u: %u\n", n, composite_count);
    
    return 0;
}