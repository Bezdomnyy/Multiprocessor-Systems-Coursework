#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void divisor_count_and_sum(unsigned int n, unsigned int* pcount,
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

int main(int argc, char** argv) {
    int num = atoi(argv[1]) + 1;
    unsigned int arithmetic_count = 0;
    unsigned int composite_count = 0;
    unsigned int n;
    unsigned int end = 0;
    unsigned int start = 0;
    int nthreads;

    // double start_time, end_time, elapsed_time;
    // start_time = omp_get_wtime();

#pragma omp parallel\
    default(none)\
    private(n)\
    shared(start, end, arithmetic_count, composite_count, num, nthreads)
{   /* start parallel */


#pragma omp single
{   /* single start */
    nthreads = omp_get_num_threads();
    while (arithmetic_count < num)
    {
        start = end + 1;
        end += num - arithmetic_count;

        n = end - start + 1;
        int ntasks = (n < nthreads ? n : nthreads);
        int taskChunk = (n + ntasks - 1) / ntasks;
        
        for (int i = 0; i < ntasks; i++) {
            int taskStart = start + i*taskChunk;
            int taskEnd = taskStart + taskChunk - 1 < end ? taskStart + taskChunk - 1 : end;

#pragma omp task
{   /* task start */
            unsigned int my_arithmetic_count = 0;
            unsigned int my_composite_count = 0;
            for (n = taskStart; n <= taskEnd; ++n)
            {
                unsigned int divisor_count;
                unsigned int divisor_sum;
                divisor_count_and_sum(n, &divisor_count, &divisor_sum);
                if (divisor_sum % divisor_count != 0)
                    continue;
                ++my_arithmetic_count;
                if (divisor_count > 2)
                    ++my_composite_count;
            }
        
#pragma omp atomic
            arithmetic_count += my_arithmetic_count;
#pragma omp atomic
            composite_count += my_composite_count;
}   /* task end */

        }

#pragma omp taskwait

    }

}   /* single end*/

}   /* parallel end */
    n = end + 1; // THIS IS INCORRECT

    // end_time = omp_get_wtime();
    // elapsed_time = end_time - start_time;

    printf("\n%uth arithmetic number is %u\n", arithmetic_count, n);
    printf("Number of composite arithmetic numbers <= %u: %u\n", n, composite_count);
    // printf("\nParallelised section execution time: %f\n\n", elapsed_time);
    
    return 0;
}