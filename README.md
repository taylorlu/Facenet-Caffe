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

## Convert tensorflow model to caffemodel and CoreML:
* #### Not all layers in tf can be convert to other framework model successfully.
* #### A conv layer can contain biases or sometimes not, the following map shows no biases in conv operation.
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/batchnorm1.png)
* #### The difference of batch normalization between tensorflow and caffe.
The expression of batch norm: <img src="https://github.com/taylorlu/FaceAll/blob/master/resource/bn1.png" alt="failed" width="120"/>
which has 2 steps:
1. calculate the mean and variance of vectors of the layer in whole batch. This step is to do normalization. <img src="https://github.com/taylorlu/FaceAll/blob/master/resource/bn3.png" alt="failed" width="70"/>. The parameters of this expression is **not trainable**.
2. Scale the normalized vectors and shift to new region. <img src="https://github.com/taylorlu/FaceAll/blob/master/resource/bn2.png" alt="failed" width="50"/>. The parameters of this expression is **trainable**.

