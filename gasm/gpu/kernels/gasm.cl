// OpenCL kernels for the GASM GPU back-end.
//
// Score matrices are stored dense, row-major, in single precision (matching the
// article's GPU implementation). Incidence matrices are stored as CSR. All the
// heavy iteration work (sparse-times-dense products) runs here on the device.

// Sparse (CSR, M x K) times dense (K x N, row-major) -> dense (M x N).
__kernel void spmm(
    const int M, const int N,
    __global const int*   row_ptr,
    __global const int*   col_idx,
    __global const float* vals,
    __global const float* D,
    __global float*       C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= M || j >= N) return;
    float acc = 0.0f;
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    for (int p = start; p < end; ++p) {
        acc += vals[p] * D[col_idx[p] * N + j];
    }
    C[i * N + j] = acc;
}

// Dense transpose: A (M x N) -> At (N x M).
__kernel void transpose(
    const int M, const int N,
    __global const float* A,
    __global float*       At)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= M || j >= N) return;
    At[j * M + i] = A[i * N + j];
}

// In-place division by a scalar.
__kernel void scale(const int n, __global float* A, const float f)
{
    int i = get_global_id(0);
    if (i >= n) return;
    A[i] /= f;
}

// In-place elementwise product A *= B.
__kernel void hadamard(const int n, __global float* A, __global const float* B)
{
    int i = get_global_id(0);
    if (i >= n) return;
    A[i] *= B[i];
}

// In-place elementwise sum A += B.
__kernel void add(const int n, __global float* A, __global const float* B)
{
    int i = get_global_id(0);
    if (i >= n) return;
    A[i] += B[i];
}

// Row-wise argmax of a dense (M x N) matrix -> int[M].
__kernel void row_argmax(
    const int M, const int N,
    __global const float* X,
    __global int*         out)
{
    int i = get_global_id(0);
    if (i >= M) return;
    int   best = 0;
    float bv   = X[i * N];
    for (int j = 1; j < N; ++j) {
        float v = X[i * N + j];
        if (v > bv) { bv = v; best = j; }
    }
    out[i] = best;
}
