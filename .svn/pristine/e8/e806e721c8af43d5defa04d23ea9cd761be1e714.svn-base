#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

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

// define mpi master's rank
#define MASTER 0

int main(int argc, char* argv[]) {

    // init mpi world
    MPI_Init(&argc, &argv);

    // program arg
    int num = atoi(argv[1]) + 1;

    // arithmetic search result vars
    unsigned int arithmetic_count = 0;
    unsigned int composite_count = 0;
    unsigned int master_arithmetic_count = 0;
    unsigned int master_composite_count = 0;
    unsigned int m_ar_cnt_buf;
    unsigned int m_cp_cnt_buf;
    unsigned int n;

    // search chunk bounds vars
    unsigned int end = 0;
    unsigned int start = 0;

    // mpi info vars
    int rank;
    int size;

    // init mpi info vars
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    while (1) {
        // reset counter vars
        arithmetic_count = 0;
        composite_count = 0;

        // bcast loop break condition
        MPI_Bcast(&master_arithmetic_count, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        if (master_arithmetic_count >= num) break;

        // calc new chunk bounds
        start = end + 1;
        end += num - master_arithmetic_count;

        // calc how many processes need to work
        int searchChunk = end - start + 1;
        int ntasks = (searchChunk < size ? searchChunk : size);

        // check if my work is needed :(
        if (rank >= ntasks) {
            MPI_Reduce(&arithmetic_count,&m_ar_cnt_buf,1,MPI_INT,MPI_SUM,MASTER,MPI_COMM_WORLD);
            MPI_Reduce(&composite_count,&m_cp_cnt_buf,1,MPI_INT,MPI_SUM,MASTER,MPI_COMM_WORLD);
            continue;
        }

        // do my chunk of work
        for (int i = start + rank; i <= end; i+=ntasks) {
            unsigned int divisor_count;
            unsigned int divisor_sum;
            divisor_count_and_sum(i, &divisor_count, &divisor_sum);
            if (divisor_sum % divisor_count != 0)
                continue;
            ++arithmetic_count;
            if (divisor_count > 2)
                ++composite_count;
        }

        // reduce counter vars
        MPI_Reduce(&arithmetic_count,&m_ar_cnt_buf,1,MPI_INT,MPI_SUM,MASTER,MPI_COMM_WORLD);
        MPI_Reduce(&composite_count,&m_cp_cnt_buf,1,MPI_INT,MPI_SUM,MASTER,MPI_COMM_WORLD);

        // update master's count vars
        if (rank == MASTER) {
            master_arithmetic_count += m_ar_cnt_buf;
            master_composite_count += m_cp_cnt_buf;
        }
    }

    // master outputs the final results
    if (rank == MASTER) {

        // set the result
        n = end + 1;

        // program output
        printf("\n%uth arithmetic number is %u\n", master_arithmetic_count, n);
        printf("Number of composite arithmetic numbers <= %u: %u\n", n, master_composite_count);

    }
    
    
    // finalize mpi world
    MPI_Finalize();

    return 0;
}