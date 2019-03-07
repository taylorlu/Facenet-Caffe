from flask import Response, Flask, send_file, make_response, request
from flask import render_template, jsonify
from io import BytesIO
from skimage.io import imsave
import cv2
import numpy as np
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import sys
import json
import time
import pyw_hnswlib as hnswlib
from flask_cors import CORS, cross_origin
import urllib.request
import requests
import socket

sys.path.append("mtcnn")
import mtcnn

minsize = 40
threshold = [0.8, 0.8, 0.6]
factor = 0.709
caffe_model_path = "./mtcnn"
imgSavePath = "static/uploadImages"

max_elements = 10*10000
faceSearchData = "./faceSearchData"
hnswModel = hnswlib.Index(space='l2', dim=128)
dThreshold = 0.6
port = 9090

caffe.set_mode_cpu()
PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)

caffePrototxt = 'InceptionResnet_Model/resnetInception-128.prototxt'
caffemodel = 'InceptionResnet_Model/inception_resnet_v1_conv1x1.caffemodel'
net = caffe.Net(caffePrototxt, caffemodel, caffe.TEST)

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
 
    return ip

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def normL2Vector(bottleNeck):
    sum = 0
    for v in bottleNeck:
        sum += np.power(v, 2)
    sqrt = np.max([np.sqrt(sum), 0.0000000001])
    vector = np.zeros((bottleNeck.shape))
    for (i, v) in enumerate(bottleNeck):
        vector[i] = v/sqrt
    return vector.astype(np.float32)

def calcCaffeVector(image):
    image = cv2.resize(image, (160,160))
    prewhitened = prewhiten(image)[np.newaxis]
    inputCaffe = prewhitened.transpose((0,3,1,2)) #[1,3,160,160]

    net.blobs['data'].data[...] = inputCaffe
    net.forward()
    vector = normL2Vector(net.blobs['flatten'].data.squeeze())
    return vector

def mtcnnDetect(image):

    if(image.shape[2]!=3 and image.shape[2]!=4):
        return [],[],[]

    if(image.shape[2]==4):
        image = image[:,:,:-1]

    img_matlab = image.copy()
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp

    # boundingboxes: [None, 5] => the last dim is probability.
    boundingboxes, points = mtcnn.detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    boundingboxes = boundingboxes.astype(np.int32)
    vectors = []

    for i in range(boundingboxes.shape[0]):

        left = boundingboxes[i][0]
        right = boundingboxes[i][2]
        top = boundingboxes[i][1]
        bottom = boundingboxes[i][3]
        
        old_size = (right-left+bottom-top)/2.0
        centerX = right - (right-left)/2.0
        centerY = bottom - (bottom-top)/2 + old_size*0.1
        size = int(old_size*1.15)
        
        x1 = int(centerX-size/2)
        y1 = int(centerY-size/2)
        x2 = int(centerX+size/2)
        y2 = int(centerY+size/2)
        width = x2 - x1
        height = y2 - y1
        
        rectify_x1 = x1
        rectify_y1 = y1
        warped = img_matlab

        if(x2>img_matlab.shape[1]):
            warped = cv2.copyMakeBorder(img_matlab, 0, 0, 0, x2-img_matlab.shape[1], cv2.BORDER_CONSTANT)
        if(x1<0):
            warped = cv2.copyMakeBorder(img_matlab, 0, 0, -x1, 0, cv2.BORDER_CONSTANT)
            rectify_x1 = 0
        if(y2>img_matlab.shape[0]):
            warped = cv2.copyMakeBorder(img_matlab, 0, y2-img_matlab.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        if(y1<0):
            warped = cv2.copyMakeBorder(img_matlab, -y1, 0, 0, 0, cv2.BORDER_CONSTANT)
            rectify_y1 = 0

        warped = warped[rectify_y1:y2, rectify_x1:x2]

        vector = calcCaffeVector(warped)
        vectors.append(vector)

        if(left<0):
            boundingboxes[i][0] = 0
        if(top<0):
            boundingboxes[i][1] = 0
        if(right>img_matlab.shape[1]):
            boundingboxes[i][2] = img_matlab.shape[1]
        if(bottom>img_matlab.shape[0]):
            boundingboxes[i][3] = img_matlab.shape[0]

    return boundingboxes, points, vectors


app = Flask(__name__)

@app.route("/faceRegister", methods=['GET', 'POST'])
def faceRegister():

    if request.method == 'POST':    # post by browser

        names = request.form['names']
        faceArr = []

        for name in names[:-1].split("*"):

            file = request.files[name]
            img = np.fromstring(file.read(), np.uint8)
            filename = imgSavePath +"/"+ str(time.time()) +".jpg"
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(filename, img)
            boundingboxes, points, vectors = mtcnnDetect(img)
            
            filenames = []
            faceDict = {}
            boxPtArr = []
            for i in range(boundingboxes.shape[0]):
                filenames.append(filename)
                boxPtDict = {}
                box = []
                box.append(int(boundingboxes[i][0]))
                box.append(int(boundingboxes[i][1]))
                box.append(int(boundingboxes[i][2]))
                box.append(int(boundingboxes[i][3]))
                boxPtDict["box"] = box

                pts = []
                for j in range(10):
                    pts.append(int(points[i][j]))
                boxPtDict["pt"] = pts
                boxPtArr.append(boxPtDict)

            faceDict["BoxsPoints"] = boxPtArr
            faceDict["name"] = file.filename
            faceArr.append(faceDict)

            # Save face vector to hnswModel
            if(len(filenames)>0):
                hnswModel.add_items(vectors, filenames)

        hnswModel.save_index(faceSearchData)
        faceJson = {}
        faceJson["faces"] = faceArr

        response = make_response(json.dumps(faceJson))
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    else:
        # image = cv2.imread("38967138.jpg")
        # b,g,r = cv2.split(image)
        # image = cv2.merge([r,g,b])
        # arr = cv2.rectangle(image, (10, 10), (200, 200), (255, 0, 0), 2)
        # strIO = BytesIO()
        # imsave(strIO, arr, plugin='pil', format_str='png')
        # strIO.seek(0)

        # response = make_response(send_file(strIO, mimetype="image/jpeg"))
        response = make_response('{"faces":[{"name":"test1.jpg","box":[10,20,30,40],"pt":[10,20,30,40,50,10,20,30,40,50]},{"name":"test1.jpg","box":[101,20,30,40],"pt":[101,201,301,401,501,10,20,30,40,50]}]}')
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response


@app.route("/faceRegisterBatch", methods=['POST'])
@cross_origin()
def faceRegisterBatch():

    obj = request.get_json(force=True)
    urls = obj['urls']
    if(len(urls)>100):
        response = make_response("Limit Error: url items count > 100")
        return response

    faceArr = []
    for url in urls:
        try:
            headers = {"Referer":"https://www.fanfiction.net/s/4873155/1/Uchiha-Riz", "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36"}

            content = requests.get(url, timeout=5, headers=headers).content
            # content = urllib.request.urlopen(url, timeout=5, headers=headers).read()

            img = np.fromstring(content, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
            filename = imgSavePath +"/"+ str(time.time()) +".jpg"
            with open(filename, "wb") as fp:
                fp.write(content)

            boundingboxes, points, vectors = mtcnnDetect(img)
            
            filenames = []
            faceDict = {}
            boxPtArr = []
            for i in range(boundingboxes.shape[0]):
                filenames.append(filename)
                boxPtDict = {}
                box = []
                box.append(int(boundingboxes[i][0]))
                box.append(int(boundingboxes[i][1]))
                box.append(int(boundingboxes[i][2]))
                box.append(int(boundingboxes[i][3]))
                boxPtDict["box"] = box

                pts = []
                for j in range(10):
                    pts.append(int(points[i][j]))
                boxPtDict["pt"] = pts
                boxPtArr.append(boxPtDict)

            faceDict["BoxsPoints"] = boxPtArr
            faceDict["name"] = url
            faceArr.append(faceDict)

            # Save face vector to hnswModel
            if(len(filenames)>0):
                hnswModel.add_items(vectors, filenames)

        except:
            response = make_response("Unfortunitely -- An Unknow Error Happened")
            return response

    hnswModel.save_index(faceSearchData)
    faceJson = {}
    faceJson["faces"] = faceArr

    response = make_response(json.dumps(faceJson))
    return response


@app.route("/faceRetrieve", methods=['GET', 'POST'])
def faceRetrieve():

    if(hnswModel.cur_ind==0):
        response = make_response('Database is null now.')
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    if request.method == 'POST':
        name = request.form['name']
        file = request.files[name]

        img = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

        boundingboxes, points, vectors = mtcnnDetect(img)

        if(boundingboxes.shape[0]==0):
            response = make_response('Not detect any face in this picture, please choose another.')
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
        elif(boundingboxes.shape[0]>1):
            response = make_response('Detect more than 1 face in this picture, please choose another.')
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response

        topN = int(hnswModel.cur_ind/100)
        if(topN<=1):
            topN = hnswModel.cur_ind
        elif(topN<=10):
            topN = int(hnswModel.cur_ind/10)

        hnswModel.set_ef(topN)
        labels, distances = hnswModel.knn_query(vectors[0], k = topN)

        faceArr = []
        for i, label in enumerate(labels[0]):
            if(distances[0][i]>dThreshold):
                break
            faceDict = {}
            faceDict["distance"] = distances[0][i].tolist()
            faceDict["path"] = label
            faceArr.append(faceDict)

        faceJson = {}
        faceJson["faces"] = faceArr
        response = make_response(json.dumps(faceJson))
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    else:
        response = make_response('Not support Get method')
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

@app.route('/faceRegister.html')
def home():
    faceRegisterURL = "http://" +get_host_ip()+ ":" +str(port)+ "/faceRegister"
    faceRegisterBatchURL = "http://" +get_host_ip()+ ":" +str(port)+ "/faceRegisterBatch"
    return render_template("faceRegister.html", faceRegister=faceRegisterURL, faceRegisterBatch=faceRegisterBatchURL)

@app.route('/faceRetrieve.html')
def home2():
    faceRetrieveURL = "http://" +get_host_ip()+ ":" +str(port)+ "/faceRetrieve"
    return render_template("faceRetrieve.html", faceRetrieve=faceRetrieveURL)

def main(argv):
    global port
    port = argv[1]
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':

    if(os.path.exists(faceSearchData)):
        hnswModel.load_index(faceSearchData)
    else:
        hnswModel.init_index(max_elements = max_elements, ef_construction = 200, M = 16)

    main(sys.argv)
    

