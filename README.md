# FaceBoxes: A CPU Real-time Face Detector with High Accuracy

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Shifeng Zhang](http://www.cbsr.ia.ac.cn/users/sfzhang/)

### Introduction
We propose a novel face detector, named FaceBoxes, with superior performance on both speed and accuracy. Moreover, the speed of FaceBoxes is invariant to the number of faces. You can use the code to train/evaluate the FaceBoxes method for face detection. For more details, please refer to our [paper](https://arxiv.org/pdf/1708.05234.pdf).

<p align="left">
<img src="https://github.com/sfzhang15/FaceBoxes/blob/master/faceboxes_framework.jpg" alt="FaceBoxes Framework" width="777px">
</p>

<p align="left">
<img src="https://github.com/sfzhang15/FaceBoxes/blob/master/faceboxes_performance.jpg" alt="FaceBoxes Performance" width="770px">
</p>

_Note: The performance of FDDB is the true positive rate (TPR) at 1000 false postives. The speed is for VGA-resolution images._

### Citing FaceBoxes

Please cite our paper in your publications if it helps your research:

    @inproceedings{zhang2017faceboxes,
      title = {Faceboxes: A CPU Real-time Face Detector with High Accuracy},
      author = {Zhang, Shifeng and Zhu, Xiangyu and Lei, Zhen and Shi, Hailin and Wang, Xiaobo and Li, Stan Z.},
      booktitle = {IJCB},
      year = {2017}
    }

### Contents
1. [Installation](#installation)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [Others](#others)

### Installation
1. Get the code. We will call the cloned directory as `$FaceBoxes_ROOT`.
  ```Shell
  git clone https://github.com/sfzhang15/FaceBoxes.git
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  cd $FaceBoxes_ROOT
  # Modify Makefile.config according to your Caffe installation.
  # Make sure to include $FaceBoxes_ROOT/python to your PYTHONPATH.
  cp Makefile.config.example Makefile.config
  make all -j && make py
  ```


### Training
1. Download the [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html) dataset, convert it to VOC format and create the LMDB file. Or you can directly download our created [LMDB](https://drive.google.com/open?id=15-G-wyql5d8f4YFxU-o6cE_-kx9H_3MT) of WIDER FACE to `$FaceBoxes_ROOT/examples/`.
  ```Shell
  # You can modify create_list.sh and create_data.sh if needed.
  cd $FaceBoxes_ROOT
  ./data/WIDER_FACE/create_list.sh
  ./data/WIDER_FACE/create_data.sh
  ```

2. Train your model on WIDER FACE.
  ```Shell
  cd $FaceBoxes_ROOT/models/faceboxes
  sh train.sh
  ```

### Evaluation
1. Download the images of [AFW](https://drive.google.com/open?id=1Kl2Cjy8IwrkYDwMbe_9DVuAwTHJ8fjev), [PASCAL Face](https://drive.google.com/open?id=1p7dDQgYh2RBPUZSlOQVU4PgaSKlq64ik) and [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to `$FaceBoxes_ROOT/examples/images/`.

2. If you do not train the model by yourself, you can download our [trained model](https://drive.google.com/open?id=1eyqFViMoBlN8JokGRHxbnJ8D4o0pTWac).

3. Check out [`test/demo.py`](https://github.com/sfzhang15/FaceBoxes/blob/master/test/demo.py) on how to detect faces using the FaceBoxes model and how to plot detection results.

4. Evaluate the trained model via [`test/afw_test.py`](https://github.com/sfzhang15/FaceBoxes/blob/master/test/afw_test.py) on AFW.

5. Evaluate the trained model via [`test/pascal_test.py`](https://github.com/sfzhang15/FaceBoxes/blob/master/test/pascal_test.py) on PASCAL Face.

6. Evaluate the trained model via [`test/fddb_test.py`](https://github.com/sfzhang15/FaceBoxes/blob/master/test/fddb_test.py) on FDDB.

7. Download the [eval_tool](https://bitbucket.org/marcopede/face-eval) to show the performance.

### Others

1. We reimplement the FaceBoxes with PyTorch as [FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch).

2. We will release a trained model of the imporved version of FaceBoxes, which jonitly performs face detection and alignment (5 landmarks).

_Note: If you can not download the created LMDB, the provided images and the trained model through the above links, you can download them through [BaiduYun](https://pan.baidu.com/s/187ktF3lJXkEl6OpoTAu9YQ)._
