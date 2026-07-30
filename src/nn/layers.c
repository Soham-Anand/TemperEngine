#include "temper/nn/layers.h"
#include "temper/utils/assert.h"
#include "temper/core/logger.h"
#include <stdlib.h>
#include <math.h>

static float rand_float(void)
{
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

TemperLayer temper_layer_dense(uint32_t in_features, uint32_t out_features)
{
    TemperLayer layer = {0};
    TemperShape w_shape = temper_shape_2d(in_features, out_features);
    layer.weights = temper_tensor_create(w_shape, TEMPER_DTYPE_F32);
    layer.grad_weights = temper_tensor_create(w_shape, TEMPER_DTYPE_F32);
    float scale = sqrtf(2.0f / (float)in_features);
    size_t count = temper_shape_count(&w_shape);
    float *w_data = temper_tensor_data(&layer.weights);
    for (size_t i = 0; i < count; i++)
    {
        w_data[i] = rand_float() * scale;
    }
    TemperShape b_shape = temper_shape_1d(out_features);
    layer.bias = temper_tensor_create(b_shape, TEMPER_DTYPE_F32);
    layer.grad_bias = temper_tensor_create(b_shape, TEMPER_DTYPE_F32);
    layer.name = "Dense";
    return layer;
}

TemperTensor temper_layer_dense_forward(const TemperLayer *layer, const TemperTensor *input)
{
    TEMPER_ASSERT(input->shape.ndim == 2);
    TEMPER_ASSERT(input->shape.dims[1] == layer->weights.shape.dims[0]);

    // Matmul: output = input @ weights
    TemperTensor output = temper_tensor_matmul(input, &layer->weights);

    // Add bias: output += bias (broadcast across batch dimension)
    int64_t batch = output.shape.dims[0];
    int64_t out_features = output.shape.dims[1];
    float *out_data = temper_tensor_data(&output);
    float *b_data = temper_tensor_data(&layer->bias);

    for (int64_t i = 0; i < batch; i++)
    {
        for (int64_t j = 0; j < out_features; j++)
        {
            out_data[i * out_features + j] += b_data[j];
        }
    }

    return output;
}

void temper_layer_destroy(TemperLayer *layer)
{
    temper_tensor_destroy(&layer->weights);
    temper_tensor_destroy(&layer->bias);
    temper_tensor_destroy(&layer->grad_weights);
    temper_tensor_destroy(&layer->grad_bias);
}

TemperLayerNorm temper_layer_norm_create(uint32_t normalized_shape, float epsilon)
{
    TemperLayerNorm ln = {0};
    TemperShape s = temper_shape_1d(normalized_shape);
    ln.gamma = temper_tensor_create(s, TEMPER_DTYPE_F32);
    ln.beta = temper_tensor_create(s, TEMPER_DTYPE_F32);
    size_t count = temper_shape_count(&s);
    float *g_data = temper_tensor_data(&ln.gamma);
    for (size_t i = 0; i < count; i++)
    {
        g_data[i] = 1.0f;
    }
    ln.epsilon = epsilon;
    return ln;
}

TemperTensor temper_layer_norm_forward(const TemperLayerNorm *ln, const TemperTensor *input)
{
    TEMPER_ASSERT(input->shape.ndim == 2);

    int64_t batch = input->shape.dims[0];
    int64_t normalized_shape = input->shape.dims[1];

    TemperTensor result = temper_tensor_create(input->shape, input->dtype);

    float *in_data = temper_tensor_data(input);
    float *res_data = temper_tensor_data(&result);
    float *gamma_data = temper_tensor_data(&ln->gamma);
    float *beta_data = temper_tensor_data(&ln->beta);

    for (int64_t i = 0; i < batch; i++)
    {
        // 1. Compute mean
        float mean = 0.0f;
        for (int64_t j = 0; j < normalized_shape; j++)
        {
            mean += in_data[i * normalized_shape + j];
        }
        mean /= (float)normalized_shape;

        // 2. Compute variance
        float variance = 0.0f;
        for (int64_t j = 0; j < normalized_shape; j++)
        {
            float diff = in_data[i * normalized_shape + j] - mean;
            variance += diff * diff;
        }
        variance /= (float)normalized_shape;

        // 3. Normalize & Affine: (x - mean) / sqrt(variance + epsilon) * gamma + beta
        float inv_std = 1.0f / sqrtf(variance + ln->epsilon);
        for (int64_t j = 0; j < normalized_shape; j++)
        {
            float normalized = (in_data[i * normalized_shape + j] - mean) * inv_std;
            res_data[i * normalized_shape + j] =
                gamma_data[j] * normalized + beta_data[j];
        }
    }

    return result;
}

void temper_layer_norm_destroy(TemperLayerNorm *ln)
{
    temper_tensor_destroy(&ln->gamma);
    temper_tensor_destroy(&ln->beta);
}

TemperEmbedding temper_embedding_create(uint32_t vocab_size, uint32_t embed_dim)
{
    TemperEmbedding emb = {0};
    emb.vocab_size = vocab_size;
    emb.embed_dim = embed_dim;
    TemperShape s = temper_shape_2d(vocab_size, embed_dim);
    emb.table = temper_tensor_create(s, TEMPER_DTYPE_F32);
    size_t count = temper_shape_count(&s);
    float *t_data = temper_tensor_data(&emb.table);
    for (size_t i = 0; i < count; i++)
    {
        t_data[i] = rand_float() * 0.1f;
    }
    return emb;
}

TemperTensor temper_embedding_forward(const TemperEmbedding *emb, const int64_t *indices,
                                      uint32_t count)
{
    TemperShape s = temper_shape_2d(count, emb->embed_dim);
    TemperTensor result = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *res_data = temper_tensor_data(&result);
    float *tbl_data = temper_tensor_data(&emb->table);

    for (uint32_t i = 0; i < count; i++)
    {
        int64_t idx = indices[i];
        TEMPER_ASSERT_MSG(idx >= 0 && (uint32_t)idx < emb->vocab_size, "Embedding index out of bounds");
        for (uint32_t j = 0; j < emb->embed_dim; j++)
        {
            res_data[i * emb->embed_dim + j] = tbl_data[idx * emb->embed_dim + j];
        }
    }
    return result;
}

void temper_embedding_destroy(TemperEmbedding *emb)
{
    temper_tensor_destroy(&emb->table);
}
