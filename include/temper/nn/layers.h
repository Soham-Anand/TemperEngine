#ifndef TEMPER_LAYERS_H
#define TEMPER_LAYERS_H

#include "temper/math/tensor.h"

typedef struct TemperLayer
{
    TemperTensor weights;
    TemperTensor bias;
    TemperTensor grad_weights;
    TemperTensor grad_bias;
    const char *name;
} TemperLayer;

TemperLayer temper_layer_dense(uint32_t in_features, uint32_t out_features);
TemperTensor temper_layer_dense_forward(const TemperLayer *layer, const TemperTensor *input);
void temper_layer_destroy(TemperLayer *layer);

typedef struct TemperLayerNorm
{
    TemperTensor gamma;
    TemperTensor beta;
    float epsilon;
} TemperLayerNorm;

TemperLayerNorm temper_layer_norm_create(uint32_t normalized_shape, float epsilon);
TemperTensor temper_layer_norm_forward(const TemperLayerNorm *ln, const TemperTensor *input);
void temper_layer_norm_destroy(TemperLayerNorm *ln);

typedef struct TemperEmbedding
{
    TemperTensor table;
    uint32_t vocab_size;
    uint32_t embed_dim;
} TemperEmbedding;

TemperEmbedding temper_embedding_create(uint32_t vocab_size, uint32_t embed_dim);
TemperTensor temper_embedding_forward(const TemperEmbedding *emb, const int64_t *indices,
                                      uint32_t count);
void temper_embedding_destroy(TemperEmbedding *emb);

#endif
