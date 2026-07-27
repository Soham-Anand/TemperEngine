#include "temper/temper.h"
#include "temper/math/tensor.h"
#include "temper/nn/layers.h"
#include "temper/nn/activations.h"
#include <stdio.h>

int main(void)
{
    temper_init();
    temper_info("TemperEngine %s", TEMPER_VERSION_STRING);

    TemperShape input_shape = temper_shape_2d(1, 8);
    TemperTensor input = temper_tensor_create(input_shape, TEMPER_DTYPE_F32);
    for (int i = 0; i < 8; i++)
    {
        input.data[i] = (float)i;
    }

    TemperLayer dense = temper_layer_dense(8, 4);
    TemperTensor out = temper_layer_dense_forward(&dense, &input);
    TemperTensor activated = temper_relu(&out);

    printf("Dense output: ");
    for (int i = 0; i < 4; i++)
    {
        printf("%.4f ", activated.data[i]);
    }
    printf("\n");

    temper_tensor_destroy(&input);
    temper_tensor_destroy(&out);
    temper_tensor_destroy(&activated);
    temper_layer_destroy(&dense);
    temper_shutdown();
    return 0;
}
