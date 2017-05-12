# Tensorflow VGG16 and VGG19

This is a Tensorflow implemention of Deep3D (The original MXNet implementation can be found in the github repository: <a href="https://github.com/piiswrong/deep3d">Deep3D-MXNet</a>. The <a href="https://arxiv.org/abs/1604.03650">published paper</a> describes the development and performance of this network. 


We have modified the implementation of <a href="https://github.com/ry/tensorflow-vgg16">tensorflow-vgg19</a> to use numpy loading instead of default tensorflow model loading in order to speed up the initialisation and reduce the overall memory usage. This implementation enable further modify the network, e.g. remove the FC layers, or increase the batch size.

>To use the VGG networks, the npy files for [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) or [VGG19 NPY](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) has to be downloaded.

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


##Update 1: Trainable VGG:
Added a trainable version of the VGG19 `vgg19_trainable`. It support train from existing vaiables or from scratch. (But the trainer is not included)

A very simple testing is added `test_vgg19_trainable`, switch has demo about how to train, switch off train mode for verification, and how to save.

A seperated file is added (instead of changing existing one) because I want to keep the simplicity of the original VGG networks.


##Update 2: Tensorflow v1.0.0:
All the source code has been upgraded to [v1.0.0](https://github.com/tensorflow/tensorflow/blob/v1.0.0-rc1/RELEASE.md).

The conversion is done by my another project [tf0to1](https://github.com/machrisaa/tf0to1)

