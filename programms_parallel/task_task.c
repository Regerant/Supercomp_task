#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <stdalign.h>
#include <stdint.h> 

static inline unsigned long long rdtsc(void)
{
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}

#define Max(a,b) ((a)>(b)?(a):(b))
#define BLOCK_SIZE 128

#define N (2*2*2*2*2*2*2*2*2*2*2*2+2)

static alignas(64) double A[N][N];
static alignas(64) double B[N][N];

static const double maxeps = 0.1e-7;
static int itmax = 100;
static double eps;
void init_data(void);
void verify(void);

int main(void)
{
    unsigned long long start_ticks = rdtsc();
    double t_start = omp_get_wtime();

    init_data();

    #pragma omp parallel
    {
        for(int it = 1; it <= itmax; it++)
        {
            #pragma omp single
            {
                eps = 0.0;

                int blocksCount1 = 0;
                for(int i = 2; i < N - 2; i += BLOCK_SIZE)
                {
                    for(int j = 2; j < N - 2; j += BLOCK_SIZE)
                    {
                        blocksCount1++;
                    }
                }

                double *max_per_task1 = (double*)calloc(blocksCount1, sizeof(double));

                int task_id1 = 0;
                #pragma omp taskgroup
                {
                    for(int i = 2; i < N - 2; i += BLOCK_SIZE)
                    {
                        int i2 = (i + BLOCK_SIZE < N - 2) ? (i + BLOCK_SIZE) : (N - 2);

                        for(int j = 2; j < N - 2; j += BLOCK_SIZE)
                        {
                            int j2 = (j + BLOCK_SIZE < N - 2) ? (j + BLOCK_SIZE) : (N - 2);

                            int local_id = task_id1++;
                            #pragma omp task firstprivate(i, j, i2, j2, local_id) shared(A, B, max_per_task1)
                            {
                                for(int ii = i; ii < i2; ii++)
                                {
                                    for(int jj = j; jj < j2; jj++)
                                    {
                                        B[ii][jj] = (
                                              A[ii-2][jj] + A[ii-1][jj] + A[ii+1][jj] + A[ii+2][jj]
                                            + A[ii][jj-2] + A[ii][jj-1] + A[ii][jj+1] + A[ii][jj+2]
                                        ) / 8.0;
                                    }
                                }
                                max_per_task1[local_id] = 0.0; 
                            }
                        }
                    }
                }
                free(max_per_task1);

                int blocksCount2 = 0;
                for(int i = 1; i < N - 1; i += BLOCK_SIZE)
                {
                    for(int j = 1; j < N - 1; j += BLOCK_SIZE)
                    {
                        blocksCount2++;
                    }
                }
                double *max_per_task2 = (double*)calloc(blocksCount2, sizeof(double));

                int task_id2 = 0;
                #pragma omp taskgroup
                {
                    for(int i = 1; i < N - 1; i += BLOCK_SIZE)
                    {
                        int i2 = (i + BLOCK_SIZE < N - 1) ? (i + BLOCK_SIZE) : (N - 1);

                        for(int j = 1; j < N - 1; j += BLOCK_SIZE)
                        {
                            int j2 = (j + BLOCK_SIZE < N - 1) ? (j + BLOCK_SIZE) : (N - 1);

                            int local_id = task_id2++;
                            #pragma omp task firstprivate(i, j, i2, j2, local_id) shared(A, B, max_per_task2)
                            {
                                double local_max = 0.0;
                                for(int ii = i; ii < i2; ii++)
                                {
                                    for(int jj = j; jj < j2; jj++)
                                    {
                                        double e = fabs(A[ii][jj] - B[ii][jj]);
                                        A[ii][jj] = B[ii][jj];
                                        if (e > local_max) 
                                        {
                                            local_max = e;
                                        }
                                    }
                                }
                                max_per_task2[local_id] = local_max;
                            }
                        }
                    }
                }

                double global_max = 0.0;
                for(int i = 0; i < blocksCount2; i++)
                {
                    global_max = Max(global_max, max_per_task2[i]);
                }
                free(max_per_task2);

                eps = global_max;
                // printf("it = %4d   eps = %e\n", it, eps);
                if (eps < maxeps)
                {
                    itmax = it - 1;
                }
            }
        }
    }

    verify();

    unsigned long long end_ticks = rdtsc();
    double t_end = omp_get_wtime();

    unsigned long long diff = end_ticks - start_ticks;
    printf("ticks: %llu\n", diff);
    printf("OMP_WTIME: %f sec\n", (t_end - t_start));

    return 0;
}

void init_data(void)
{
    #pragma omp parallel for
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if (i == 0 || i == N-1 || j == 0 || j == N-1)
                A[i][j] = 0.0;
            else
                A[i][j] = 1.0 + i + j;
        }
    }
}

void verify(void)
{
    double s = 0.0;
    #pragma omp parallel
    {
        double s_local = 0.0;
        #pragma omp for nowait
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                s_local += A[i][j] * (i + 1) * (j + 1) / (double)(N*N);
            }
        }
        #pragma omp atomic
        s += s_local;
    }
    printf("S = %f\n", s);
}
