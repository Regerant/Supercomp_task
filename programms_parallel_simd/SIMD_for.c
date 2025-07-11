#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))
//#define N (2 * 2 * 2 * 2 * 2 * 2 + 2)
#define  N   (2*2*2*2*2*2*2*2*2*2*2*2+2)

double maxeps = 0.1e-7;
int itmax = 100;
double eps;
double A[N][N], B[N][N];

void relax();
void resid();
void init();
void verify();

int main(int argc, char **argv)
{
    int it;
    double time_start, time_fin;

    init();

    time_start = omp_get_wtime();

    for (it = 1; it <= itmax; it++)
    {
        eps = 0.0;

        #pragma omp parallel
        {
            relax();
            resid();
        }
        printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps) break;
    }

    verify();

    time_fin = omp_get_wtime();
    printf("Time: %g sec\n", time_fin - time_start);
    printf("NxN: %i\nN: %i\n", N*N, N);

    return 0;
}

void init()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (i == 0 || i == N - 1 || j == 0 || j == N - 1) ? 0.0 : (1.0 + i + j);
}

void relax()
{
    #pragma omp for collapse(2)
    for (int i = 2; i < N - 2; i++)
    {
        for (int j = 2; j < N - 2; j += 4)
        {
            __m256d ai_2j = _mm256_loadu_pd(&A[i - 2][j]);
            __m256d ai_1j = _mm256_loadu_pd(&A[i - 1][j]);
            __m256d ai_2jp2 = _mm256_loadu_pd(&A[i + 2][j]);
            __m256d ai_1jp1 = _mm256_loadu_pd(&A[i + 1][j]);

            __m256d aijm2 = _mm256_loadu_pd(&A[i][j - 2]);
            __m256d aijm1 = _mm256_loadu_pd(&A[i][j - 1]);
            __m256d aijp2 = _mm256_loadu_pd(&A[i][j + 2]);
            __m256d aijp1 = _mm256_loadu_pd(&A[i][j + 1]);

            __m256d sum = _mm256_add_pd(ai_2j, ai_1j);
            sum = _mm256_add_pd(sum, ai_2jp2);
            sum = _mm256_add_pd(sum, ai_1jp1);
            sum = _mm256_add_pd(sum, aijm2);
            sum = _mm256_add_pd(sum, aijm1);
            sum = _mm256_add_pd(sum, aijp2);
            sum = _mm256_add_pd(sum, aijp1);

            __m256d avg = _mm256_div_pd(sum, _mm256_set1_pd(8.0));
            _mm256_storeu_pd(&B[i][j], avg);
        }
    }
}

void resid()
{
    eps = 0.0;
    __m256d max_eps = _mm256_set1_pd(0.0);

    #pragma omp for collapse(2)
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j += 4)
        {
            __m256d aij = _mm256_loadu_pd(&A[i][j]);
            __m256d bij = _mm256_loadu_pd(&B[i][j]);

            __m256d diff = _mm256_sub_pd(aij, bij);
            __m256d abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff);

            max_eps = _mm256_max_pd(max_eps, abs_diff);

            _mm256_storeu_pd(&A[i][j], bij);
        }
    }
    double temp_eps[4];
    _mm256_storeu_pd(temp_eps, max_eps);

    for (int idx = 0; idx < 4; idx++)
    {
        eps = Max(eps, temp_eps[idx]);
    }
    #pragma omp barrier
}

void verify()
{
    double s = 0.0;
    double norm_factor = 1.0 / (N * N);

    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j += 4)
        {
            __m256d aij = _mm256_loadu_pd(&A[i][j]);
            __m256d indices = _mm256_set_pd((i + 1) * (j + 4), (i + 1) * (j + 3), (i + 1) * (j + 2), (i + 1) * (j + 1));
            __m256d prod = _mm256_mul_pd(aij, indices);
            __m256d scaled_prod = _mm256_mul_pd(prod, _mm256_set1_pd(norm_factor));

            double temp_s[4];
            _mm256_storeu_pd(temp_s, scaled_prod);
            s += temp_s[0] + temp_s[1] + temp_s[2] + temp_s[3];
        }
    }
    printf("  S = %f\n", s);
}
