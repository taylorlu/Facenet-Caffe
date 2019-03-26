## Facenet-Caffe
This project contains:
* A web app based on flask to serve face register and face retrieve.
* Convert Inception_resnet_v1 model to Caffemodel, and to CoreML model in iOS.
* The iOS App can be found in [face_recognition_ios](https://github.com/taylorlu/face_recognition_ios)
## Prerequisites
1. Flask
2. pyw_hnswlib [fast approximate nearest neighbors](https://github.com/nmslib/hnswlib)
3. pyCaffe
4. Tensorflow (Only use to convert model)

## Web App
* ### Running
`python faceServer.py 8000`
#### The model path and save path is defined in the top of faceServer.py
`caffe_model_path = "./mtcnn"`</br>
`imgSavePath = "static/uploadImages"`</br>
`faceSearchData = "./faceSearchData"`</br>
`caffePrototxt = 'InceptionResnet_Model/resnetInception.prototxt'`</br>
`caffemodel = 'InceptionResnet_Model/inception_resnet_v1_conv1x1.caffemodel'`
#### The pretrained model of facenet based on caffe can be downloaded in</br> https://pan.baidu.com/s/11_nNcdE-fHuPuL3AWGcgOw</br>
#### which convert from tfmodel</br>
https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view
#### **warning**
**It seems that the older pretrained model with 128D embedding output has a higher accuray than new model with 512D output, refer to https://github.com/davidsandberg/facenet/issues/948. So here use pretrained model `20170512-110547` as default. You can change to `20180402-114759` in `tf2caffe.py`, remember to change `EMBEDDING_SIZE = 512` as well, and also `hnswlib.Index(space='l2', dim=512)` in `faceServer.py`.**

* ### Outline
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/web.png)

* ### register html
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/urlregister.png)
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/localregister.png)

* ### retrieve html
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/retrieve.png)
* #### API Response
1. POST /faceRegister

`{"faces": [{"BoxsPoints": [{"box": [639, 83, 824, 316], "pt": [671, 757, 712, 693, 764, 171, 157, 214, 263, 252]}, {"box": [252, 85, 456, 357], "pt": [309, 402, 353, 304, 395, 192, 197, 250, 285, 291]}], "name": "5252ab74-de19-4a35-ba2b-b96379c18055.jpg"}]}`

2. POST /faceRegisterBatch

`{"faces": [{"BoxsPoints": [{"box": [104, 146, 145, 196], "pt": [111, 127, 117, 118, 132, 167, 163, 178, 188, 184]}, {"box": [279, 80, 327, 144], "pt": [287, 307, 295, 292, 310, 107, 105, 123, 132, 130]}, {"box": [0, 152, 38, 211], "pt": [12, 32, 22, 10, 26, 174, 177, 185, 194, 197]}, {"box": [161, 99, 207, 159], "pt": [172, 193, 183, 174, 194, 123, 122, 134, 143, 143]}], "name": "http://img1.imgtn.bdimg.com/it/u=1446904722,2747785645&fm=26&gp=0.jpg"}, {"BoxsPoints": [{"box": [188, 37, 264, 137], "pt": [216, 250, 241, 219, 250, 80, 77, 102, 118, 116]}], "name": "http://img3.imgtn.bdimg.com/it/u=2577494811,3655771084&fm=26&gp=0.jpg"}]}`

3. POST /faceRetrieve

`{"faces": [{"distance": 0.23119516670703888, "path": "static/uploadImages/1550042580.7593167.jpg"}, {"distance": 0.31492435932159424, "path": "static/uploadImages/1550042916.6169314.jpg"}, {"distance": 0.46090781688690186, "path": "static/uploadImages/1550042916.6169314.jpg"}, {"distance": 0.48163485527038574, "path": "static/uploadImages/1550042916.206.jpg"}]}`

## Convert tensorflow model to caffemodel:
* #### Not all layers in tf can be converted to other framework model successfully.
* #### A conv layer can contain bias or sometimes not, the following map shows no bias in conv operation.
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/batchnorm1.png)
* #### The difference of batch normalization between tensorflow and caffe.
The expression of batch norm: <img src="https://github.com/taylorlu/FaceAll/blob/master/resource/bn1.png" alt="failed" width="120"/>
which has 2 steps:
1. <img src="https://github.com/taylorlu/FaceAll/blob/master/resource/bn3.png" alt="failed" width="70"/>.  Calculate the mean and variance of vectors in the layer of whole batch. This step is to do normalization. The 2 parameters of this expression is **Not Trainable**.
2. <img src="https://github.com/taylorlu/FaceAll/blob/master/resource/bn2.png" alt="failed" width="100"/>.  Scale the normalized vectors and shift to new region. The 2 parameters of this expression is **Trainable**.

In tensorflow, parameter gamma is fixed to 1.0, so the only trainable parameter is **beta**, `moving_mean` and `moving_variance` are calculated batch by batch.

![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/batchnorm2.png)

But in Caffe, batchnorm layer only do normalization, without rescale and shift, so we must put a scale layer on the top of each batchnorm layer. And also need to add `bias_term: true` in prototxt, or there will be no beta value.

![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/batchnorm3.png)

The shortcut of code is showing as follows:

![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/batchnorm4.png)
* #### other tips: 
1. `net.params['..'][1].data` should be assigned when there is biases in conv layer.
2. When input by 8x8 feature map, feed into pooling layer by size=3x3, stride=2, the output size of tensorflow is 3x3, but the caffe is 4x4, so it should do crop in caffe. The handy method is to replace crop layer by using kernel [[1,0], [0,0]] of conv2d.
3. Some versions of caffe has no batchnorm or scale layer, which will indicate some keys can not be found in the future, solution is to change to another version of caffe.
4. `Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz`, 8 Cores, 32 processors, retrieve speed of HSNW Algorithm is approximate 10ms-20ms, the amount of 100,000 faces, 512D embedding vector each face. Based on MKL BlAS library.

## Convert tensorflow model to CoreML:
1. InnerProduct warns shape not match, solved by using 1x1 conv2d replace.
2. Make sure the RGB, ARGB color space in iOS is all right.
