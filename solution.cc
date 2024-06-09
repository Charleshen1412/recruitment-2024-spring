#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>

void optimized_pre_phase1(size_t) {}

void optimized_post_phase1() {}

void optimized_pre_phase2(size_t) {}

void optimized_post_phase2() {}

size_t Partition(float *data, size_t i, size_t j)
{
    // size_t randomIndex = i + rand() % (j - i + 1);
    // float temp = data[randomIndex];
    // data[randomIndex] = data[i];
    // data[i] = temp;
    float temp = data[i];
    while (i < j)
    {
        while (i < j && data[j] >= temp)
        {
            j--;
        }
        data[i] = data[j];
        while (i < j && data[i] <= temp)
        {
            i++;
        }
        data[j] = data[i];
    }
    data[i] = temp;
    return i;
}

void quicksort_omp(float *data, size_t i, size_t j, size_t cutoff)
{
    if (i < j)
    {
        size_t pos = Partition(data, i, j);
        if (j - i < cutoff)
        {
            if (pos > i + 1)
            {
                quicksort_omp(data, i, pos - 1, cutoff);
            }
            if (pos < j - 1)
            {
                quicksort_omp(data, pos + 1, j, cutoff);
            }
        }
        else
        {
            if (pos > i + 1)
            {
#pragma omp task
                quicksort_omp(data, i, pos - 1, cutoff);
            }
            if (pos < j - 1)
            {
#pragma omp task
                quicksort_omp(data, pos + 1, j, cutoff);
            }
        }
    }
}

void optimized_do_phase1(float *data, size_t size)
{
    size_t cutoff = 1000;
#pragma omp parallel num_threads(20)
    {
#pragma omp single nowait
        {
            quicksort_omp(data, 0, size - 1, cutoff);
        }
    }
}

// void optimized_do_phase1(float *data, size_t size)
// {
//     std::sort(data, data + size);
// }

// void optimized_do_phase1(float* data, size_t size) {
//     size_t i, phase;
//     float temp;
//     #pragma omp parallel num_threads(128) default(none) shared(data, size) private(i, temp, phase)
//     for(phase = 0; phase < size; phase++){
//         if(phase % 2 == 0){
//             #pragma omp for
//             for(i = 1; i < size; i += 2){
//                 if(data[i-1]>data[i]){
//                     temp = data[i-1];
//                     data[i-1] = data[i];
//                     data[i] = temp;
//                 }
//             }
//         }else{
//             #pragma omp for
//             for(i = 1; i < size - 1; i += 2){
//                 if(data[i]>data[i+1]){
//                     temp = data[i+1];
//                     data[i+1] = data[i];
//                     data[i] = temp;
//                 }
//             }
//         }
//     }
// }

// void multi_binary_search(size_t* result, float* data, float* query, size_t size) {
//     int my_rank = omp_get_thread_num();
//     int thread_count = omp_get_num_threads();
//     size_t local_n = size / thread_count;
//     size_t local_end = my_rank == thread_count - 1 ? size : (my_rank + 1) * local_n;
//     size_t i = my_rank * local_n;
//
//     for (; i + 15 < local_end; i += 16)
//     {
//         __m512 query_vec = _mm512_loadu_ps(&query[i]);
//         __m512i l_vec = _mm512_setzero_si512();
//         __m512i r_vec = _mm512_set1_epi32(size);
//         while (true)
//         {
//             __m512i m_vec = _mm512_srli_epi32(_mm512_add_epi32(l_vec, r_vec), 1);
//             __m512 data_vec = _mm512_i32gather_ps(m_vec, data, 4);
//             __mmask16 mask = _mm512_cmp_ps_mask(data_vec, query_vec, _CMP_LT_OQ);
//             l_vec = _mm512_mask_add_epi32(l_vec, mask, m_vec, _mm512_set1_epi32(1));
//             r_vec = _mm512_mask_mov_epi32(r_vec, mask, m_vec);
//             __mmask16 done_mask = _mm512_cmpeq_epi32_mask(l_vec, r_vec);
//             // alignas(64) int mask_array[16];
//             // _mm512_store_epi32(mask_array, _mm512_maskz_set1_epi32(done_mask, 1));
//             // std::cout << "Done mask array: ";
//             // for (int k = 0; k < 16; ++k) {
//             //     std::cout << mask_array[k] << " ";
//             // }
//             // std::cout << std::endl;
//             if (done_mask == 0x0000)
//             {
//                 // std::cout<<"yes";
//                 break;
//             }
//         }
//         _mm512_store_epi32(&result[i], l_vec);
//     }
//     for (; i < local_end; ++i)
//     {
//         size_t l = 0, r = size;
//         while (l < r)
//         {
//             size_t m = l + (r - l) / 2;
//             if (data[m] < query[i])
//             {
//                 l = m + 1;
//             }
//             else
//             {
//                 r = m;
//             }
//         }
//         result[i] = l;
//     }
// }

// void optimized_do_phase2(size_t* result, float* data, float* query, size_t size) {
//     #pragma omp parallel num_threads(128)
//     multi_binary_search(result, data, query, size);
// }

void multi_binary_search(size_t *result, float *data, float *query, size_t size)
{
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    size_t local_n = size / thread_count;
    size_t local_end = my_rank == thread_count - 1 ? size : (my_rank + 1) * local_n;
    size_t i = my_rank * local_n;
    for (; i + 7 < local_end; i += 8)
    {
        __m256 query_vec = _mm256_loadu_ps(&query[i]);
        __m256i l_vec = _mm256_setzero_si256();
        __m256i r_vec = _mm256_set1_epi32(size);

        while (true)
        {
            __m256i m_vec = _mm256_srli_epi32(_mm256_add_epi32(l_vec, r_vec), 1);
            __m256 data_vec = _mm256_i32gather_ps(data, m_vec, 4);
            __m256 mask = _mm256_cmp_ps(data_vec, query_vec, _CMP_LT_OQ);
            l_vec = _mm256_blendv_epi8(l_vec, _mm256_add_epi32(m_vec, _mm256_set1_epi32(1)), _mm256_castps_si256(mask));
            r_vec = _mm256_blendv_epi8(m_vec, r_vec, _mm256_castps_si256(mask));
            __m256i done_mask = _mm256_cmpeq_epi32(l_vec, r_vec);
            if (_mm256_testc_si256(done_mask, _mm256_set1_epi32(-1)))
            {
                break;
            }
        }

        alignas(32) int l_array[8];
        _mm256_store_si256(reinterpret_cast<__m256i *>(l_array), l_vec);
        for (int j = 0; j < 8; ++j)
        {
            result[i + j] = l_array[j];
        }
    }
    for (; i < local_end; ++i)
    {
        size_t l = 0, r = size;
        while (l < r)
        {
            size_t m = l + (r - l) / 2;
            if (data[m] < query[i])
            {
                l = m + 1;
            }
            else
            {
                r = m;
            }
        }
        result[i] = l;
    }
}

void optimized_do_phase2(size_t *result, float *data, float *query, size_t size)
{
#pragma omp parallel num_threads(128)
    multi_binary_search(result, data, query, size);
}

// void multi_binary_search(size_t* result, float* data, float* query, size_t size) {
//     int my_rank = omp_get_thread_num();
//     int thread_count = omp_get_num_threads();
//     size_t local_n = size / thread_count;
//     size_t local_a = my_rank * local_n;
//     size_t local_b = my_rank == thread_count - 1 ? size : (my_rank + 1) * local_n;
//     for (size_t i = local_a; i < local_b; ++i) {
//         size_t l = 0, r = size;
//         while (l < r) {
//             size_t m = l + (r - l) / 2;
//             if (data[m] < query[i]) {
//                 l = m + 1;
//             } else {
//                 r = m;
//             }
//         }
//         result[i] = l;
//     }
// }

// void optimized_do_phase2(size_t* result, float* data, float* query, size_t size) {
//     #pragma omp parallel num_threads(128)
//     multi_binary_search(result, data, query, size);
// }

// void optimized_do_phase2(size_t* result, float* data, float* query, size_t size) {
//     for (size_t i = 0; i < size; ++i) {
//         size_t l = 0, r = size;
//         while (l < r) {
//             size_t m = l + (r - l) / 2;
//             if (data[m] < query[i]) {
//                 l = m + 1;
//             } else {
//                 r = m;
//             }
//         }
//         result[i] = l;
//     }
// }