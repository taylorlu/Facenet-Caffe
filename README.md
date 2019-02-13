## FaceAll
This project contains:
* A web app based on flask to serve face register and face retrieve.
* Convert Inception_resnet_v1 model to Caffemodel, and to CoreML model in iOS.

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
#### The pretrained model of facenet based on caffe can be downloaded in</br> https://pan.baidu.com/s/11_nNcdE-fHuPuL3AWGcgOw

* ### Outline
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/web.png)

* ### register html
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/urlregister.png)
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/localregister.png)

* ### retrieve html
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/retrieve.png)

## Convert tensorflow model to caffemodel:
* #### Not all layers in tf can be convert to other framework model successfully.
* #### A conv layer can contain biases or sometimes not, the following map shows no biases in conv operation.
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/batchnorm1.png)
* #### The difference of batch normalization between tensorflow and caffe.
The expression of batch norm: <img src="https://github.com/taylorlu/FaceAll/blob/master/resource/bn1.png" alt="failed" width="120"/>
which has 2 steps:
1. <img src="https://github.com/taylorlu/FaceAll/blob/master/resource/bn3.png" alt="failed" width="70"/>.  Calculate the mean and variance of vectors in the layer of whole batch. This step is to do normalization. The 2 parameters of this expression is **Not Trainable**.
2. <img src="https://github.com/taylorlu/FaceAll/blob/master/resource/bn2.png" alt="failed" width="100"/>.  Scale the normalized vectors and shift to new region. The 2 parameters of this expression is **Trainable**.

In tensorflow, parameter gamma is fixed to 1.0, so the only trainable parameter is beta, moving_mean and moving_variance are calculated batch by batch.

![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/batchnorm2.png)

But in Caffe, batchnorm layer only do normalization, without rescale and shift, so we must put a scale layer on the top of each batchnorm layer. And also need to add `bias_term: true` in prototxt, or there will be no beta value.

![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/batchnorm3.png)

The shortcut of code is showing as follows:

![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/batchnorm4.png)
* #### other tips: 
1. `net.params['..'][1].data` should be assigned when there is biases in conv layer.
2. When input by 8x8 feature map, feed into pooling layer by size=3x3, stride=2, the output size of tensorflow is 3x3, but the caffe is 4x4, so should do crop in caffe. Crop layer can be replaced by using kernel [[1,0], [0,0]] of conv2d.
3. Some versions of caffe has no batchnorm or scale layer, which will indicate some keys can not be found in the future, solution is to change to another version of caffe.
4. `Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz`, 8 Cores, 32 processors, retrieve speed of HSNW Algorithm is approximate 10ms-20ms, the amount of 100,000 faces, 512D embedding vector each face. Based on MKL BlAS library.

## Convert tensorflow model to CoreML:
1. InnerProduct warns shape not match, solved by using 1x1 conv2d replace.
2. Make sure the RGB, ARGB color space in iOS is all right.
