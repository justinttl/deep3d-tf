# Tensorflow Deep3D
This is a Tensorflow implemention of Deep3D (The original MXNet implementation can be found in the github repository: <a href="https://github.com/piiswrong/deep3d">Deep3D-MXNet</a>. The <a href="https://arxiv.org/abs/1604.03650">published paper</a> describes the development and performance of the original network. There is a lot of work to be done for this ported model! 
* Refactor and port over NumPy weights to Tensorflow loading methods
* Add training persistance
* Train WAY longer on hollywood Dataset
* Explore TensorServe API for web hosting + experiment with Ruby API

**Presentation Slides:**  https://goo.gl/iijL3X

<img src="https://github.com/JustinTTL/Deep3D_TF/blob/master/viz/graph_run.png" width="700">

The backbone of this network is built from the github repository of an <a href="https://github.com/machrisaa/tensorflow-vgg">implementation of VGG19</a>.

## Some Results
<img src="https://github.com/JustinTTL/Deep3D_TF/blob/master/viz/dancegirl.gif" width="700">
<img src="https://github.com/JustinTTL/Deep3D_TF/blob/master/viz/horse.png" width="700">
<img src="https://github.com/JustinTTL/Deep3D_TF/blob/master/viz/depth.gif" width="700">
<img src="https://github.com/JustinTTL/Deep3D_TF/blob/master/viz/frodo.gif" width="700">

 
