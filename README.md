# TfLite-vx-delegate
TfLite-vx-delegate constructed with TIM-VX as an openvx delegate for tensorflow lite. Before vx-delegate, you may have nnapi-linux version from Verisilicon, we suggest you move to this new delegate because:

    1. without nnapi, it's flexiable to enable more AI operators.
    2. vx-delegate is opensourced, and will promised compatible with latest tensorflow release.

# Examples

examples/minimal
modified based on [offical minimal](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/lite/examples/minimal/)

```sh
minimal libvx_delegate.so mobilenet_v2_1.0_224_quant.tflite
```
