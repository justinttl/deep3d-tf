# Tensorflow VGG16 and VGG19

This is a Tensorflow implemention of VGG 16 and VGG 19 based on [tensorflow-vgg16](https://github.com/ry/tensorflow-vgg16) and [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow). Original Caffe implementation can be found in [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) and [here](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77).

We have modified the implementation of <a href="https://github.com/ry/tensorflow-vgg16">tensorflow-vgg16</a> to use numpy loading instead of default tensorflow model loading in order to speed up the initialisation and reduce the overall memory usage. This implementation enable further modify the network, e.g. remove the FC layers, or increase the batch size.

>To use the VGG networks, the npy files for <a href="https://dl.dropboxusercontent.com/u/50333326/vgg16.npy">VGG16</a> or <a href="https://dl.dropboxusercontent.com/u/50333326/vgg19.npy">VGG19</a> has to be downloaded.

##Usage
Use this to build the VGG object
```
vgg = vgg19.Vgg19()
vgg.build(images)
```
or
```
vgg = vgg16.Vgg16()
vgg.build(images)
```
The `images` is a tensor with shape `[None, 224, 224, 3]`. 
>Trick: the tensor can be a placeholder, a variable or even a constant.

All the VGG layers (tensors) can then be accessed using the vgg object. For example, `vgg.conv1_1`, `vgg.conv1_2`, `vgg.pool5`, `vgg.prob`, ...

`test_vgg16.py` and `test_vgg19.py` contain the sample usage.

##Extra
This library has been used in my another Tensorflow image style synethesis project: [stylenet](https://github.com/machrisaa/stylenet)
