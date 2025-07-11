#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <stdalign.h>

static inline unsigned long long rdtsc(void)
{
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}

#define Max(a,b) ((a)>(b)?(a):(b))
#define N (2*2*2*2*2*2*2*2*2*2*2*2+2)

static alignas(64) double A[N][N];
static alignas(64) double B[N][N];

static const double maxeps = 0.1e-7;
static int itmax = 100;

static double eps;
void init_data(void);
void relax(void);
void resid(void);
void verify(void);

static int get_chunk_size(void)
{
    int nThreads = omp_get_max_threads();
    int chunk = 8 * nThreads;
    if (chunk < 1) chunk = 1;
    return chunk;
}

int main(void)
{
    unsigned long long start_ticks = rdtsc();
    double t_start = omp_get_wtime();

    init_data();

    #pragma omp parallel
    {
        int chunk = get_chunk_size();

        for(int it = 1; it <= itmax; it++)
        {
            #pragma omp single
            {
                eps = 0.0;
            }
            #pragma omp for schedule(static, chunk) nowait
            for(int i = 2; i < N - 2; i++)
            {
                for(int j = 2; j < N - 2; j++)
                {
                    B[i][j] = (
                          A[i-2][j] + A[i-1][j] + A[i+1][j] + A[i+2][j]
                        + A[i][j-2] + A[i][j-1] + A[i][j+1] + A[i][j+2]
                    ) / 8.0;
                }
            }
            #pragma omp barrier
            #pragma omp for schedule(static, chunk) reduction(max:eps)
            for(int i = 1; i < N - 1; i++)
            {
                for(int j = 1; j < N - 1; j++)
                {
                    double e = fabs(A[i][j] - B[i][j]);
                    A[i][j]   = B[i][j];
                    eps       = Max(eps, e);
                }
            }
            #pragma omp single
            {
               // printf("it=%4d   eps=%e\n", it, eps);
                if (eps < maxeps)
                {
                    itmax = it - 1;
                }
            }
        }
    }
    verify();

    unsigned long long end_ticks = rdtsc();
    unsigned long long diff = end_ticks - start_ticks;
    double t_end = omp_get_wtime();

    printf("ticks: %llu\n", diff);
    printf("OMP_WTIME: %f sec\n", (t_end - t_start));

    return 0;
}
void init_data(void)
{
    int chunk = get_chunk_size();
    #pragma omp parallel for schedule(static, chunk)
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
void relax(void){}
void resid(void){}
void verify(void)
{
    double s = 0.0;
    int chunk = get_chunk_size();

    #pragma omp parallel for schedule(static, chunk) reduction(+:s)
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            s += A[i][j] * (i + 1) * (j + 1) / (double)(N*N);
        }
    }
    printf("S = %f\n", s);
}