#include "temper/nn/layers.h"
#include "temper/nn/activations.h"
#include "temper/nn/losses.h"
#include "temper/training/optimizer.h"
#include "temper/training/scheduler.h"
#include "temper/training/checkpoint.h"
#include <stdio.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;
static int test_failed = 0;

#define TEST(name) static void name(void)
#define RUN(name)                                                             \
    do                                                                        \
    {                                                                         \
        tests_run++;                                                          \
        test_failed = 0;                                                      \
        printf("  %-40s", #name);                                             \
        name();                                                               \
        if (test_failed)                                                      \
            printf("FAIL\n");                                                 \
        else                                                                  \
        {                                                                     \
            tests_passed++;                                                   \
            printf("PASS\n");                                                 \
        }                                                                     \
    } while (0)

#define ASSERT(cond)                                                          \
    do                                                                        \
    {                                                                         \
        if (!(cond))                                                          \
        {                                                                     \
            printf("\n    %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            test_failed = 1;                                                  \
            return;                                                           \
        }                                                                     \
    } while (0)

TEST(test_dense_layer)
{
    TemperLayer layer = temper_layer_dense(8, 4);
    ASSERT(temper_tensor_data(&layer.weights) != NULL);
    ASSERT(layer.weights.shape.dims[0] == 8);
    ASSERT(layer.weights.shape.dims[1] == 4);
    temper_layer_destroy(&layer);
}

TEST(test_layer_norm)
{
    TemperLayerNorm ln = temper_layer_norm_create(16, 1e-5f);
    ASSERT(temper_tensor_data(&ln.gamma) != NULL);
    ASSERT(ln.epsilon == 1e-5f);
    temper_layer_norm_destroy(&ln);
}

TEST(test_layer_norm_forward)
{
    TemperLayerNorm ln = temper_layer_norm_create(4, 1e-5f);
    TemperShape s = temper_shape_2d(1, 4);
    TemperTensor input = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *idata = temper_tensor_data(&input);
    idata[0] = 1.0f; idata[1] = 2.0f; idata[2] = 3.0f; idata[3] = 4.0f;
    TemperTensor out = temper_layer_norm_forward(&ln, &input);
    float *odata = temper_tensor_data(&out);
    float mean = (odata[0] + odata[1] + odata[2] + odata[3]) / 4.0f;
    ASSERT(fabsf(mean) < 0.01f);
    temper_tensor_destroy(&input);
    temper_tensor_destroy(&out);
    temper_layer_norm_destroy(&ln);
}

TEST(test_embedding)
{
    TemperEmbedding emb = temper_embedding_create(100, 32);
    ASSERT(temper_tensor_data(&emb.table) != NULL);
    ASSERT(emb.vocab_size == 100);
    ASSERT(emb.embed_dim == 32);
    int64_t indices[] = {0, 1, 2};
    TemperTensor out = temper_embedding_forward(&emb, indices, 3);
    ASSERT(out.shape.dims[0] == 3);
    ASSERT(out.shape.dims[1] == 32);
    temper_tensor_destroy(&out);
    temper_embedding_destroy(&emb);
}

TEST(test_relu)
{
    TemperShape s = temper_shape_1d(4);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *tdata = temper_tensor_data(&t);
    tdata[0] = -1.0f; tdata[1] = 2.0f; tdata[2] = -3.0f; tdata[3] = 4.0f;
    TemperTensor r = temper_relu(&t);
    float *rdata = temper_tensor_data(&r);
    ASSERT(rdata[0] == 0.0f);
    ASSERT(rdata[1] == 2.0f);
    ASSERT(rdata[2] == 0.0f);
    ASSERT(rdata[3] == 4.0f);
    temper_tensor_destroy(&t);
    temper_tensor_destroy(&r);
}

TEST(test_gelu)
{
    TemperShape s = temper_shape_1d(1);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *tdata = temper_tensor_data(&t);
    tdata[0] = 0.0f;
    TemperTensor r = temper_gelu(&t);
    float *rdata = temper_tensor_data(&r);
    ASSERT(fabsf(rdata[0]) < 0.01f);
    temper_tensor_destroy(&t);
    temper_tensor_destroy(&r);
}

TEST(test_softmax)
{
    TemperShape s = temper_shape_2d(1, 3);
    TemperTensor t = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *tdata = temper_tensor_data(&t);
    tdata[0] = 1.0f; tdata[1] = 2.0f; tdata[2] = 3.0f;
    TemperTensor r = temper_softmax(&t, 1);
    float *rdata = temper_tensor_data(&r);
    float sum = rdata[0] + rdata[1] + rdata[2];
    ASSERT(fabsf(sum - 1.0f) < 1e-5f);
    temper_tensor_destroy(&t);
    temper_tensor_destroy(&r);
}

TEST(test_cross_entropy)
{
    TemperShape s = temper_shape_2d(1, 3);
    TemperTensor pred = temper_tensor_create(s, TEMPER_DTYPE_F32);
    TemperTensor target = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *pdata = temper_tensor_data(&pred);
    float *tdata = temper_tensor_data(&target);
    pdata[0] = 0.2f; pdata[1] = 0.3f; pdata[2] = 0.5f;
    tdata[2] = 1.0f;
    float loss = temper_cross_entropy(&pred, &target);
    ASSERT(loss > 0.0f);
    temper_tensor_destroy(&pred);
    temper_tensor_destroy(&target);
}

TEST(test_mse)
{
    TemperShape s = temper_shape_1d(2);
    TemperTensor pred = temper_tensor_create(s, TEMPER_DTYPE_F32);
    TemperTensor target = temper_tensor_create(s, TEMPER_DTYPE_F32);
    float *pdata = temper_tensor_data(&pred);
    float *tdata = temper_tensor_data(&target);
    pdata[0] = 1.0f; pdata[1] = 2.0f;
    tdata[0] = 1.5f; tdata[1] = 2.5f;
    float loss = temper_mse(&pred, &target);
    ASSERT(fabsf(loss - 0.25f) < 1e-6f);
    temper_tensor_destroy(&pred);
    temper_tensor_destroy(&target);
}

TEST(test_sgd_optimizer)
{
    TemperOptimizer opt = temper_optimizer_sgd(0.01f);
    ASSERT(opt.learning_rate == 0.01f);
    ASSERT(opt.step == 0);
}

TEST(test_adam_optimizer)
{
    TemperOptimizer opt = temper_optimizer_adam(0.001f, 0.9f, 0.999f, 1e-8f);
    ASSERT(opt.learning_rate == 0.001f);
    ASSERT(opt.beta1 == 0.9f);
}

TEST(test_cosine_scheduler)
{
    TemperScheduler sched = temper_scheduler_cosine(0.01f, 0.001f, 100, 1000);
    float lr0 = temper_scheduler_get_lr(&sched, 0);
    ASSERT(fabsf(lr0) < 1e-6f);
    float lr50 = temper_scheduler_get_lr(&sched, 50);
    ASSERT(lr50 > 0.0f);
    ASSERT(lr50 < 0.01f);
}

TEST(test_checkpoint)
{
    TemperCheckpoint cp = {.epoch = 5, .step = 100, .loss = 0.5f};
    const char *path = "test_temper.cp";
    int ret = temper_checkpoint_save(path, &cp);
    ASSERT(ret == 0);
    TemperCheckpoint loaded = {0};
    ret = temper_checkpoint_load(path, &loaded);
    ASSERT(ret == 0);
    ASSERT(loaded.epoch == 5);
    ASSERT(loaded.step == 100);
    ASSERT(fabsf(loaded.loss - 0.5f) < 1e-6f);
    remove(path);
}

int main(void)
{
    printf("=== NN Tests ===\n");
    RUN(test_dense_layer);
    RUN(test_layer_norm);
    RUN(test_layer_norm_forward);
    RUN(test_embedding);
    RUN(test_relu);
    RUN(test_gelu);
    RUN(test_softmax);
    RUN(test_cross_entropy);
    RUN(test_mse);
    RUN(test_sgd_optimizer);
    RUN(test_adam_optimizer);
    RUN(test_cosine_scheduler);
    RUN(test_checkpoint);
    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
