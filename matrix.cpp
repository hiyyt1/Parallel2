#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>  // Для Intrinsics
#include <omp.h>        // Для OpenMP

// Исходная программа (без оптимизации)
void matrix_mult_naive(const double* A, const double* B, double* C, int N, int K, int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

// Полуавтоматическая векторизация с OpenMP
void matrix_mult_vectorized(const double* A, const double* B, double* C, int N, int K, int M) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

// Векторизация с Intrinsics (AVX2)
void matrix_mult_intrinsics(const double* A, const double* B, double* C, int N, int K, int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; j += 4) {  // Обрабатываем по 4 элемента за раз
            __m256d c = _mm256_setzero_pd();  // Инициализируем вектор нулями
            for (int k = 0; k < K; ++k) {
                __m256d a = _mm256_broadcast_sd(&A[i * K + k]);  // Загружаем элемент из A
                __m256d b = _mm256_loadu_pd(&B[k * M + j]);      // Загружаем 4 элемента из B
                c = _mm256_fmadd_pd(a, b, c);                   // Fused multiply-add
            }
            _mm256_storeu_pd(&C[i * M + j], c);  // Сохраняем результат
        }
    }
}

// Функция для замера времени
template <typename Func>
void measure_time(Func func, const char* label, const double* A, const double* B, double* C, int N, int K, int M) {
    auto start = std::chrono::high_resolution_clock::now();
    func(A, B, C, N, K, M);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << label << ": " << elapsed.count() << " секунд" << std::endl;
}

// Главная функция
int main() {
    const int N = 1024, K = 1024, M = 1024;

    std::vector<double> A(N * K, 1.0);
    std::vector<double> B(K * M, 1.0);
    std::vector<double> C(N * M, 0.0);

    std::cout << "Умножение матриц " << N << "x" << K << " и " << K << "x" << M << ":\n";

    // Исходная программа
    measure_time(matrix_mult_naive, "Исходная программа", A.data(), B.data(), C.data(), N, K, M);

    // Полуавтоматическая векторизация с OpenMP
    measure_time(matrix_mult_vectorized, "Полуавтоматическая векторизация (OpenMP)", A.data(), B.data(), C.data(), N, K, M);

    // Векторизация с Intrinsics
    measure_time(matrix_mult_intrinsics, "Ручная векторизация с Intrinsics (AVX2)", A.data(), B.data(), C.data(), N, K, M);

    return 0;
}
