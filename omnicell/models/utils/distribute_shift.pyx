# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: language_level=3

# Add at the top of the file
cdef extern from "omp.h":
    int omp_get_thread_num() nogil
    int omp_get_num_threads() nogil
    void omp_set_num_threads(int)
    void omp_set_schedule(int, int)

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport round
from numpy.random import Generator, PCG64
from cython.parallel cimport parallel, prange
from openmp cimport omp_get_thread_num


# Define C types for better performance
ctypedef np.float32_t F32_t
ctypedef np.float64_t F64_t
ctypedef np.int64_t I64_t

cimport cython
from cython cimport floating

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def sample_multinomial_batch(np.ndarray[F32_t, ndim=2] probs,
                           np.ndarray[F32_t, ndim=1] counts,
                           np.ndarray[F32_t, ndim=2] max_count=None,
                           int max_rejections=100,
                           int num_threads=8):
    if num_threads > 1:
        omp_set_num_threads(num_threads)
    # Transpose input matrices
    probs = np.ascontiguousarray(probs.T)  # Now [n_cols, n_rows]
    if max_count is not None:
        max_count = np.ascontiguousarray(max_count.T)
    counts = np.ascontiguousarray(counts)
    
    cdef int n_cols = probs.shape[0]  # Swapped dimensions
    cdef int n_rows = probs.shape[1]
    cdef np.ndarray[F32_t, ndim=2] result = np.zeros((n_cols, n_rows), dtype=np.float32)
    cdef bint has_max_count = max_count is not None
    
    # Create thread-local buffers
    cdef np.ndarray[F64_t, ndim=2] p_tmp_all = np.zeros((num_threads, n_rows), dtype=np.float64)
    cdef np.ndarray[I64_t, ndim=2] sample_all = np.zeros((num_threads, n_rows), dtype=np.int64)
    
    # Create RNGs outside parallel region
    rng_list = [Generator(PCG64()) for _ in range(num_threads)]
    
    cdef int thread_id, start, end, block_size
    
    # Calculate block size based on number of columns and threads
    
    
    # Process blocks of columns in parallel
    if num_threads > 1:
        block_size = n_cols // num_threads + 1
        for start in prange(0, n_cols, block_size, nogil=True, num_threads=num_threads, schedule='static'):
            thread_id = omp_get_thread_num()
            end = min(start + block_size, n_cols)
            with gil:
                _process_column_block(
                    start, end, probs, counts, max_count, result,
                    p_tmp_all[thread_id],
                    sample_all[thread_id],
                    rng_list[thread_id],
                    max_rejections,
                    has_max_count
                )
    else:
        # Use single thread for small matrices
        _process_column_block(
            0, n_cols, probs, counts, max_count, result,
            p_tmp_all[0],
            sample_all[0],
            rng_list[0],
            max_rejections,
            has_max_count
        )
    
    return result.T  # Transpose back before returning

cdef _process_column_block(int start,
                          int end,
                          F32_t[:, :] probs,
                          F32_t[:] counts,
                          F32_t[:, :] max_count,
                          F32_t[:, :] result,
                          F64_t[:] p_tmp,
                          I64_t[:] sample,
                          object rng,
                          int max_rejections,
                          bint has_max_count):
    """Process a block of columns together"""
    cdef int n_rows = probs.shape[1]  # Note: dimensions swapped
    cdef int i, j, k
    cdef F64_t p_sum
    cdef int n_count
    cdef np.ndarray[I64_t, ndim=1] temp_sample
    cdef np.ndarray[F64_t, ndim=1] p_tmp_arr
    cdef np.ndarray[I64_t, ndim=1] sample_arr
    cdef int remaining
    
    # Process each column in the block
    for j in range(start, end):
        n_count = int(round(counts[j]))
        if n_count == 0:
            continue
            
        # Copy and normalize probabilities with float64 precision
        p_sum = 0.0
        for i in range(n_rows):
            p_tmp[i] = probs[j, i]  # Note: indices swapped
            p_sum += p_tmp[i]
            
        if p_sum == 0:
            continue
            
        for i in range(n_rows):
            p_tmp[i] /= p_sum
        
        # Create numpy arrays from memoryviews and sample
        p_tmp_arr = np.asarray(p_tmp)
        temp_sample = rng.multinomial(n=n_count, pvals=p_tmp_arr)
        
        # Copy values individually
        for i in range(n_rows):
            sample[i] = temp_sample[i]
        
        if has_max_count:
            for k in range(max_rejections):
                over_max = False
                for i in range(n_rows):
                    if sample[i] > max_count[j, i]:  # Note: indices swapped
                        over_max = True
                        break
                        
                if not over_max:
                    break
                    
                # Resample if needed
                p_sum = 0.0
                for i in range(n_rows):
                    if sample[i] > max_count[j, i]:  # Note: indices swapped
                        sample[i] = int(max_count[j, i])  # Note: indices swapped
                    p_tmp[i] = probs[j, i] if sample[i] < max_count[j, i] else 0  # Note: indices swapped
                    p_sum += p_tmp[i]
                    
                if p_sum > 0:
                    for i in range(n_rows):
                        p_tmp[i] /= p_sum
                    
                    sample_arr = np.asarray(sample)
                    remaining = n_count - int(np.sum(sample_arr))
                    
                    if remaining > 0:
                        p_tmp_arr = np.asarray(p_tmp)
                        temp_sample = rng.multinomial(n=remaining, pvals=p_tmp_arr)
                        for i in range(n_rows):
                            sample[i] += temp_sample[i]
        
        # Copy results back to float32
        for i in range(n_rows):
            result[j, i] = float(sample[i])  # Note: indices swapped


def sample_pert(np.ndarray[F32_t, ndim=2] ctrl,
                np.ndarray[F32_t, ndim=2] weighted_dist,
                np.ndarray[F32_t, ndim=1] mean_shift,
                int max_rejections=100,
                int num_threads=1012):
    # Ensure contiguous arrays
    ctrl = np.ascontiguousarray(ctrl)
    weighted_dist = np.ascontiguousarray(weighted_dist)
    mean_shift = np.ascontiguousarray(mean_shift)
    
    cdef int n_rows = ctrl.shape[0]
    cdef int n_cols = ctrl.shape[1]
    cdef np.ndarray[F32_t, ndim=1] count_shift = np.round(mean_shift * n_rows).astype(np.float32)
    cdef np.ndarray[F32_t, ndim=2] max_count = np.copy(ctrl)
    cdef np.ndarray[F32_t, ndim=2] samples
    cdef np.ndarray[F32_t, ndim=2] sampled_pert
    cdef int i, j
    cdef float sign
    
    # Set max_count to large value where count_shift > 0
    for j in range(n_cols):
        if count_shift[j] > 0:
            for i in range(n_rows):
                max_count[i, j] = 1e10
    
    samples = sample_multinomial_batch(
        weighted_dist, 
        np.abs(count_shift), 
        max_count=max_count, 
        max_rejections=max_rejections,
        num_threads=num_threads
    )
    
    sampled_pert = np.zeros((n_rows, n_cols), dtype=np.float32)
    
    # Apply shifts with proper sign
    for j in range(n_cols):
        sign = 1.0 if count_shift[j] > 0 else -1.0
        for i in range(n_rows):
            sampled_pert[i, j] = max(0.0, ctrl[i, j] + sign * samples[i, j])
    
    return sampled_pert

def get_proportional_weighted_dist(X):
    # Convert input to float32 and ensure contiguous
    X = np.ascontiguousarray(X, dtype=np.float32)
    
    cdef np.ndarray[F32_t, ndim=2] X_arr = X
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[F32_t, ndim=2] weighted_dist = np.zeros((n_rows, n_cols), dtype=np.float32)
    cdef float col_sum
    cdef int i, j
    
    # Process columns serially (parallel version was causing issues)
    for j in range(n_cols):
        col_sum = 0.0
        for i in range(n_rows):
            col_sum += X_arr[i, j]
            
        if col_sum > 0:
            for i in range(n_rows):
                weighted_dist[i, j] = X_arr[i, j] / col_sum
                    
    return weighted_dist