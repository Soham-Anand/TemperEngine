#include "temper/metal/metal.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include "temper/core/device.h"
#include "temper/core/logger.h"
#include "temper/core/runtime.h"
#include "temper/compute/kernel.h"
#include "temper/math/tensor.h"
#include "temper/math/shape.h"
#include <string.h>
#include <stdlib.h>

// Embedded .metallib produced at build time by bin2c (see CMakeLists.txt).
extern const unsigned char temper_metal_library_bytes[];
extern const size_t temper_metal_library_bytes_len;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

static id<MTLDevice> s_device = nil;
static id<MTLCommandQueue> s_queue = nil;
static id<MTLLibrary> s_library = nil;
static TemperRuntime s_metal_runtime;
static bool s_metal_initialized = false;

// host_ptr -> MTLBuffer registry. The core only holds the shared-memory
// pointer (buffer.contents); kernels need the MTLBuffer object, so the runtime
// keeps the mapping between the two.
#define METAL_MAX_BUFS 4096

typedef struct MetalBufEntry
{
    void *host_ptr;
    id<MTLBuffer> buffer;
} MetalBufEntry;

static MetalBufEntry s_bufs[METAL_MAX_BUFS];
static uint32_t s_buf_count = 0;

// Pipeline cache: one compute pipeline state per kernel function name.
#define METAL_MAX_PIPELINES 16

typedef struct MetalPipelineEntry
{
    const char *name;
    id<MTLComputePipelineState> state;
} MetalPipelineEntry;

static MetalPipelineEntry s_pipelines[METAL_MAX_PIPELINES];
static uint32_t s_pipeline_count = 0;

// ---------------------------------------------------------------------------
// Buffer registry
// ---------------------------------------------------------------------------

static id<MTLBuffer> metal_buffer_for_host(const void *host_ptr)
{
    if (!host_ptr)
    {
        return nil;
    }
    for (uint32_t i = 0; i < s_buf_count; i++)
    {
        if (s_bufs[i].host_ptr == host_ptr)
        {
            return s_bufs[i].buffer;
        }
    }
    return nil;
}

static void *metal_alloc_host(size_t bytes)
{
    if (bytes == 0)
    {
        return NULL;
    }
    if (s_buf_count >= METAL_MAX_BUFS)
    {
        temper_error("Metal buffer registry full (%d entries)", METAL_MAX_BUFS);
        return NULL;
    }
    id<MTLBuffer> buffer = [s_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer)
    {
        temper_error("Metal buffer allocation failed (%zu bytes)", bytes);
        return NULL;
    }
    // Preserve the calloc-like contract the core expects: memory is zeroed.
    memset(buffer.contents, 0, bytes);
    s_bufs[s_buf_count].host_ptr = buffer.contents;
    s_bufs[s_buf_count].buffer = buffer;
    s_buf_count++;
    return buffer.contents;
}

static void metal_free_host(void *ptr)
{
    if (!ptr)
    {
        return;
    }
    for (uint32_t i = 0; i < s_buf_count; i++)
    {
        if (s_bufs[i].host_ptr == ptr)
        {
            s_bufs[i] = s_bufs[s_buf_count - 1];
            s_buf_count--;
            return;
        }
    }
    temper_warn("metal_free_host: pointer not owned by this runtime");
}

// ---------------------------------------------------------------------------
// Pipeline cache
// ---------------------------------------------------------------------------

static id<MTLComputePipelineState> metal_pipeline_for(const char *name)
{
    for (uint32_t i = 0; i < s_pipeline_count; i++)
    {
        if (strcmp(s_pipelines[i].name, name) == 0)
        {
            return s_pipelines[i].state;
        }
    }
    if (s_pipeline_count >= METAL_MAX_PIPELINES)
    {
        temper_error("Metal pipeline cache full");
        return nil;
    }
    NSString *fn = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [s_library newFunctionWithName:fn];
    if (!function)
    {
        temper_error("Missing Metal kernel function: %s", name);
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> state =
        [s_device newComputePipelineStateWithFunction:function error:&error];
    if (!state)
    {
        temper_error("Failed to build compute pipeline for %s", name);
        return nil;
    }
    s_pipelines[s_pipeline_count].name = strdup(name);
    s_pipelines[s_pipeline_count].state = state;
    s_pipeline_count++;
    return state;
}

static const char *metal_kernel_fn(TemperKernelType type)
{
    switch (type)
    {
    case TEMPER_KERNEL_ADD:
        return "kernel_add";
    case TEMPER_KERNEL_SUB:
        return "kernel_sub";
    case TEMPER_KERNEL_MUL:
        return "kernel_mul";
    case TEMPER_KERNEL_DIV:
        return "kernel_div";
    case TEMPER_KERNEL_RELU:
        return "kernel_relu";
    case TEMPER_KERNEL_GELU:
        return "kernel_gelu";
    case TEMPER_KERNEL_SILU:
        return "kernel_silu";
    case TEMPER_KERNEL_MATMUL:
        return "kernel_matmul";
    default:
        return NULL;
    }
}

// ---------------------------------------------------------------------------
// Kernel launches
// ---------------------------------------------------------------------------

static bool is_unary(TemperKernelType type)
{
    return type == TEMPER_KERNEL_RELU || type == TEMPER_KERNEL_GELU ||
           type == TEMPER_KERNEL_SILU;
}

static int metal_elementwise_launch(const TemperKernelImpl *impl,
                                    const TemperDispatchArgs *args,
                                    TemperKernelReport *report)
{
    (void)report;
    if (!args || !args->output || !args->inputs || args->input_count < 1)
    {
        return -1;
    }
    const TemperTensor *in0 = args->inputs[0];
    if (!in0->resource)
    {
        return -1;
    }
    size_t n = temper_shape_count(&in0->shape);

    id<MTLBuffer> b0 = metal_buffer_for_host(in0->resource->host_ptr);
    if (!b0)
    {
        return -1;
    }

    id<MTLBuffer> b1 = nil;
    if (!is_unary(impl->type))
    {
        if (args->input_count < 2 || !args->inputs[1] || !args->inputs[1]->resource)
        {
            return -1;
        }
        const TemperTensor *in1 = args->inputs[1];
        if (temper_shape_count(&in1->shape) != n)
        {
            // Broadcasting is not supported on GPU in Phase 4 (dispatch falls
            // back to CPU via Phase 5). Reject loudly rather than corrupt.
            temper_warn("Metal element-wise kernels require equal shapes");
            return -1;
        }
        b1 = metal_buffer_for_host(in1->resource->host_ptr);
        if (!b1)
        {
            return -1;
        }
    }

    TemperTensor out;
    memset(&out, 0, sizeof(out));
    out = temper_tensor_create_on_device(in0->shape, in0->dtype, TEMPER_DEVICE_GPU_0);
    if (!out.resource)
    {
        return -1;
    }
    id<MTLBuffer> bo = metal_buffer_for_host(out.resource->host_ptr);
    if (!bo)
    {
        temper_tensor_destroy(&out);
        return -1;
    }

    const char *fn = metal_kernel_fn(impl->type);
    id<MTLComputePipelineState> ps = metal_pipeline_for(fn);
    if (!ps)
    {
        temper_tensor_destroy(&out);
        return -1;
    }

    id<MTLCommandBuffer> cb = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:b0 offset:0 atIndex:0];
    if (b1)
    {
        [enc setBuffer:b1 offset:0 atIndex:1];
        [enc setBuffer:bo offset:0 atIndex:2];
    }
    else
    {
        [enc setBuffer:bo offset:0 atIndex:1];
    }
    NSUInteger tg = MIN((NSUInteger)256, ps.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
    {
        temper_error("Metal %s kernel failed", fn);
        temper_tensor_destroy(&out);
        return -1;
    }
    *args->output = out;
    return 0;
}

static int metal_matmul_launch(const TemperKernelImpl *impl,
                               const TemperDispatchArgs *args,
                               TemperKernelReport *report)
{
    (void)impl;
    (void)report;
    if (!args || !args->output || !args->inputs || args->input_count != 2)
    {
        return -1;
    }
    const TemperTensor *a = args->inputs[0];
    const TemperTensor *b = args->inputs[1];
    if (!a->resource || !b->resource || a->shape.ndim != 2 || b->shape.ndim != 2)
    {
        return -1;
    }
    if (a->shape.dims[1] != b->shape.dims[0])
    {
        temper_error("Metal matmul shape mismatch: %lld vs %lld",
                     a->shape.dims[1], b->shape.dims[0]);
        return -1;
    }
    uint32_t M = (uint32_t)a->shape.dims[0];
    uint32_t K = (uint32_t)a->shape.dims[1];
    uint32_t N = (uint32_t)b->shape.dims[1];

    id<MTLBuffer> ba = metal_buffer_for_host(a->resource->host_ptr);
    id<MTLBuffer> bb = metal_buffer_for_host(b->resource->host_ptr);
    if (!ba || !bb)
    {
        return -1;
    }

    TemperShape out_shape = temper_shape_2d(M, N);
    TemperTensor out;
    memset(&out, 0, sizeof(out));
    out = temper_tensor_create_on_device(out_shape, a->dtype, TEMPER_DEVICE_GPU_0);
    if (!out.resource)
    {
        return -1;
    }
    id<MTLBuffer> bc = metal_buffer_for_host(out.resource->host_ptr);
    if (!bc)
    {
        temper_tensor_destroy(&out);
        return -1;
    }

    id<MTLComputePipelineState> ps = metal_pipeline_for("kernel_matmul");
    if (!ps)
    {
        temper_tensor_destroy(&out);
        return -1;
    }

    id<MTLCommandBuffer> cb = [s_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:ba offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:bc offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(M) atIndex:3];
    [enc setBytes:&K length:sizeof(K) atIndex:4];
    [enc setBytes:&N length:sizeof(N) atIndex:5];
    const uint32_t TILE = 16;
    uint32_t gw = ((N + TILE - 1) / TILE) * TILE;
    uint32_t gh = ((M + TILE - 1) / TILE) * TILE;
    [enc dispatchThreads:MTLSizeMake(gw, gh, 1)
        threadsPerThreadgroup:MTLSizeMake(TILE, TILE, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
    {
        temper_error("Metal matmul kernel failed");
        temper_tensor_destroy(&out);
        return -1;
    }
    *args->output = out;
    return 0;
}

// ---------------------------------------------------------------------------
// Kernel implementations
// ---------------------------------------------------------------------------

static TemperKernelImpl g_metal_impls[TEMPER_KERNEL_TYPE_COUNT];

static void register_metal_impl(TemperKernelType type, const char *name,
                                TemperKernelLaunchFn launch)
{
    TemperKernelImpl *impl = &g_metal_impls[type];
    impl->name = name;
    impl->type = type;
    impl->device_type = TEMPER_DEVICE_GPU;
    impl->launch = launch;
    if (temper_kernel_register(impl) != 0)
    {
        temper_error("Failed to register Metal kernel %s", name);
    }
}

static void register_metal_kernels(void)
{
    register_metal_impl(TEMPER_KERNEL_ADD, "metal_add", metal_elementwise_launch);
    register_metal_impl(TEMPER_KERNEL_SUB, "metal_sub", metal_elementwise_launch);
    register_metal_impl(TEMPER_KERNEL_MUL, "metal_mul", metal_elementwise_launch);
    register_metal_impl(TEMPER_KERNEL_DIV, "metal_div", metal_elementwise_launch);
    register_metal_impl(TEMPER_KERNEL_RELU, "metal_relu", metal_elementwise_launch);
    register_metal_impl(TEMPER_KERNEL_GELU, "metal_gelu", metal_elementwise_launch);
    register_metal_impl(TEMPER_KERNEL_SILU, "metal_silu", metal_elementwise_launch);
    register_metal_impl(TEMPER_KERNEL_MATMUL, "metal_matmul_tiled", metal_matmul_launch);
}

// ---------------------------------------------------------------------------
// Runtime plumbing
// ---------------------------------------------------------------------------

static int metal_init(void)
{
    return s_metal_initialized ? 0 : -1;
}

static void metal_shutdown(void)
{
    if (!s_metal_initialized)
    {
        return;
    }
    for (uint32_t i = 0; i < s_pipeline_count; i++)
    {
        free((void *)s_pipelines[i].name);
    }
    s_pipeline_count = 0;
    s_buf_count = 0;
    s_library = nil;
    s_queue = nil;
    s_device = nil;
    s_metal_initialized = false;
}

static void metal_noop(void)
{
}

bool temper_metal_is_available(void)
{
    @autoreleasepool
    {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev != nil;
    }
}

int temper_metal_maybe_init(void)
{
    if (s_metal_initialized)
    {
        return 0;
    }

    s_device = MTLCreateSystemDefaultDevice();
    if (!s_device)
    {
        temper_warn("Metal unavailable: no default device");
        return -1;
    }
    s_queue = [s_device newCommandQueue];
    if (!s_queue)
    {
        temper_error("Metal command queue creation failed");
        return -1;
    }

    dispatch_data_t libdata = dispatch_data_create(temper_metal_library_bytes,
                                                   temper_metal_library_bytes_len,
                                                   NULL, NULL);
    NSError *error = nil;
    s_library = [s_device newLibraryWithData:libdata error:&error];
    if (!s_library)
    {
        temper_error("Failed to load embedded Metal library (%s)",
                     error ? error.localizedDescription.UTF8String : "unknown error");
        return -1;
    }

    TemperDeviceCaps caps = {0};
    caps.supports_fp16 = true;
    caps.supports_bf16 = false;
    caps.supports_int8 = true;
    caps.total_memory = (size_t)s_device.recommendedMaxWorkingSetSize;
    caps.compute_threads = (size_t)[[NSProcessInfo processInfo] activeProcessorCount];
    caps.tflops = 0.0f;
    if (temper_device_register(TEMPER_DEVICE_GPU_0, caps) < 0)
    {
        temper_warn("GPU_0 already registered; caps updated by Metal init");
    }

    s_metal_runtime = (TemperRuntime){
        .name = "metal",
        .device = TEMPER_DEVICE_GPU_0,
        .alloc = metal_alloc_host,
        .free = metal_free_host,
        .alloc_host = metal_alloc_host,
        .free_host = metal_free_host,
        .init = metal_init,
        .shutdown = metal_shutdown,
        .synchronize = metal_noop,
        .wait_idle = metal_noop,
    };
    if (temper_runtime_register(&s_metal_runtime) != 0)
    {
        return -1;
    }

    register_metal_kernels();
    s_metal_initialized = true;
    temper_info("Metal backend initialized: %s", s_device.name.UTF8String);
    return 0;
}

void temper_metal_shutdown(void)
{
    metal_shutdown();
}

#endif // __APPLE__
