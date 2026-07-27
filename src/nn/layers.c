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
    for (size_t i = 0; i < count; i++)
    {
        layer.weights.data[i] = rand_float() * scale;
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
    return temper_tensor_matmul(input, &layer->weights);
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
    for (size_t i = 0; i < count; i++)
    {
        ln.gamma.data[i] = 1.0f;
    }
    ln.epsilon = epsilon;
    return ln;
}

TemperTensor temper_layer_norm_forward(const TemperLayerNorm *ln, const TemperTensor *input)
{
    (void)ln;
    TemperTensor result = temper_tensor_create(input->shape, input->dtype);
    size_t count = temper_shape_count(&input->shape);
    for (size_t i = 0; i < count; i++)
    {
        result.data[i] = input->data[i];
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
    for (size_t i = 0; i < count; i++)
    {
        emb.table.data[i] = rand_float() * 0.1f;
    }
    return emb;
}

TemperTensor temper_embedding_forward(const TemperEmbedding *emb, const int64_t *indices,
                                      uint32_t count)
{
    TemperShape s = temper_shape_2d(count, emb->embed_dim);
    TemperTensor result = temper_tensor_create(s, TEMPER_DTYPE_F32);
    for (uint32_t i = 0; i < count; i++)
    {
        int64_t idx = indices[i];
        for (uint32_t j = 0; j < emb->embed_dim; j++)
        {
            result.data[i * emb->embed_dim + j] = emb->table.data[idx * emb->embed_dim + j];
        }
    }
    return result;
}

void temper_embedding_destroy(TemperEmbedding *emb)
{
    temper_tensor_destroy(&emb->table);
}
