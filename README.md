## FaceAll
This project contains:
* A web app based on flask to serve face register and face retrieve.
* Convert Inception_resnet_v1 model to Caffemodel, and to CoreML model in iOS.

## Prerequisites
1. Flask
2. pyw_hnswlib [fast approximate nearest neighbors](https://github.com/nmslib/hnswlib)
3. PyCaffe
4. tensorflow (Only use to convert model)

## Web App
* ### Running
`python faceServer.py 8000`
#### The model path and save path is defined in the top of faceServer.py
`caffe_model_path = "./mtcnn"  
imgSavePath = "static/uploadImages"  
faceSearchData = "./faceSearchData"  
caffePrototxt = 'InceptionResnet_Model/resnetInception.prototxt'  
caffemodel = 'InceptionResnet_Model/inception_resnet_v1_conv1x1.caffemodel'`
#### the pretrained model of facenet based on caffe can be downloaded in https://pan.baidu.com/s/11_nNcdE-fHuPuL3AWGcgOw

* ### Outline
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/web.png)

* ### register html
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/urlregister.png)
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/localretrieve.png)

* ### retrieve html
![failed](https://github.com/taylorlu/FaceAll/blob/master/resource/retrieve.png)

## Tips:

#1: BatchNorm, difference between caffe and tensorflow.

