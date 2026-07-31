// TemperEngine Phase 4 compute kernels (MSL 2.x). Compiled to a .metallib at
// build time and embedded into the static library as a byte array.
#include <metal_stdlib>
using namespace metal;

// --- element-wise ------------------------------------------------------------

kernel void kernel_add(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       uint index [[thread_position_in_grid]])
{
    out[index] = a[index] + b[index];
}

kernel void kernel_sub(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       uint index [[thread_position_in_grid]])
{
    out[index] = a[index] - b[index];
}

kernel void kernel_mul(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       uint index [[thread_position_in_grid]])
{
    out[index] = a[index] * b[index];
}

kernel void kernel_div(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       uint index [[thread_position_in_grid]])
{
    out[index] = a[index] / b[index];
}

kernel void kernel_relu(device const float *in [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        uint index [[thread_position_in_grid]])
{
    float v = in[index];
    out[index] = v > 0.0f ? v : 0.0f;
}

kernel void kernel_gelu(device const float *in [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        uint index [[thread_position_in_grid]])
{
    float v = in[index];
    out[index] = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
}

kernel void kernel_silu(device const float *in [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        uint index [[thread_position_in_grid]])
{
    float v = in[index];
    out[index] = v / (1.0f + exp(-v));
}

// --- matmul: tiled, threadgroup-shared A/B tiles, bounds-guarded ------------

constant uint MATMUL_TILE = 16;

kernel void kernel_matmul(device const float *A [[buffer(0)]],
                          device const float *B [[buffer(1)]],
                          device float *C [[buffer(2)]],
                          constant uint &M [[buffer(3)]],
                          constant uint &K [[buffer(4)]],
                          constant uint &N [[buffer(5)]],
                          uint2 gid [[threadgroup_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float As[MATMUL_TILE][MATMUL_TILE];
    threadgroup float Bs[MATMUL_TILE][MATMUL_TILE];

    uint row = gid.y * MATMUL_TILE + lid.y;
    uint col = gid.x * MATMUL_TILE + lid.x;
    float acc = 0.0f;

    uint num_tiles = (K + MATMUL_TILE - 1) / MATMUL_TILE;
    for (uint t = 0; t < num_tiles; t++)
    {
        uint a_row = gid.y * MATMUL_TILE + lid.y;
        uint a_col = t * MATMUL_TILE + lid.x;
        As[lid.y][lid.x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;

        uint b_row = t * MATMUL_TILE + lid.y;
        uint b_col = gid.x * MATMUL_TILE + lid.x;
        Bs[lid.y][lid.x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < MATMUL_TILE; i++)
        {
            acc += As[lid.y][i] * Bs[i][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
    {
        C[row * N + col] = acc;
    }
}
