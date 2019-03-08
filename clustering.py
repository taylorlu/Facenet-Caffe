""" Face Cluster """
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import importlib
import argparse
import math
from scipy import misc
import sys
sys.path.append("mtcnn")
import mtcnn
import time
import cv2
from random import shuffle
import networkx as nx
import uuid
# import sqlite3
from imutils import paths

minsize = 40
threshold = [0.8, 0.8, 0.6]
factor = 0.709
caffe_model_path = "./mtcnn"

caffe.set_mode_cpu()
PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a cos distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    import numpy as np
    if len(face_encodings) == 0:
        return np.empty((0))

    #return 1/np.linalg.norm(face_encodings - face_to_compare, axis=1)
    return np.sum(face_encodings*face_to_compare,axis=1)

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def calcCaffeVector(net,image):
    image = cv2.resize(image, (160,160))
    prewhitened = prewhiten(image)[np.newaxis]
    inputCaffe = prewhitened.transpose((0,3,1,2)) #[1,3,160,160]

    net.blobs['data'].data[...] = inputCaffe
    net.forward()
    vector = normL2Vector(net.blobs['flatten'].data.squeeze())
    return vector

def mtcnnDetect(image):

    try:
        if(image.shape[2]!=3 and image.shape[2]!=4):
            return [],[]

        if(image.shape[2]==4):
            image = image[:,:,:-1]

    except Exception as e:
        return [],[]

    img_matlab = image.copy()
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp

    # boundingboxes: [None, 5] => the last dim is probability.
    boundingboxes, points = mtcnn.detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    boundingboxes = boundingboxes.astype(np.int32)
    warpedFaces = []

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
        warpedFaces.append(warped)

        if(left<0):
            boundingboxes[i][0] = 0
        if(top<0):
            boundingboxes[i][1] = 0
        if(right>img_matlab.shape[1]):
            boundingboxes[i][2] = img_matlab.shape[1]
        if(bottom>img_matlab.shape[0]):
            boundingboxes[i][3] = img_matlab.shape[0]

    return boundingboxes, warpedFaces


def normL2Vector(bottleNeck):
    sum = 0
    for v in bottleNeck:
        sum += np.power(v, 2)
    sqrt = np.max([np.sqrt(sum), 0.0000000001])
    vector = np.zeros((bottleNeck.shape))
    for (i, v) in enumerate(bottleNeck):
        vector[i] = v/sqrt
    return vector.astype(np.float32)

def _chinese_whispers(encoding_list, threshold=0.55, iterations=20):
    """ Chinese Whispers Algorithm

    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/

    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate

    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of faceId,
            sorted by largest cluster to smallest
    """

    #from face_recognition.api import _face_distance

    G = nx.Graph()

    # Create graph
    nodes = []
    edges = []

    faceIds, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx+1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': faceIds[idx], 'path': faceIds[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                # Add edge if facial match
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(list(cluster_nodes))
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            #use the max sum of neighbor weights class as current node's class
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters

def cluster_facial_encodings(facial_encodings):
    """ Cluster facial encodings

        Intended to be an optional switch for different clustering algorithms, as of right now
        only chinese whispers is available.

        Input:
            facial_encodings: (faceId, facial_encoding) dictionary of facial encodings

        Output:
            sorted_clusters: a list of clusters, a cluster being a list of faceId,
                sorted by largest cluster to smallest

    """

    if len(facial_encodings) <= 1:
        print ("Number of facial encodings must be greater than one, can't cluster")
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings.items())
    return sorted_clusters

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

def compute_facial_encodings(net, image_paths):

    """ Compute Facial Encodings

        Given a set of images, compute the facial encodings of each face detected in the images and
        return them. If no faces, or more than one face found, return nothing for that image.

        Inputs:
            image_paths: a list of image paths

        Outputs:
            facial_encodings: (faceId, facial_encoding) dictionary of facial encodings

    """

    vectors = []
    features = []
    faceIds = []
    pts = []
    picPaths = []
    facial_encodings = {}

    for i in range(len(image_paths)):

        img = cv_imread(image_paths[i])   # BGR
        boundingboxes, warpedFaces = mtcnnDetect(img)

        for j in range(len(warpedFaces)):
            vector = calcCaffeVector(net, warpedFaces[j])
            vectors.append(vector)

            feature = ''
            for v in vector:    # float vector to str '1.0,2.2,3.2,...'
                feature.join(str(v))
                feature.join(',')
            feature = feature[:-1]
            features.append(feature)

            left = boundingboxes[j][0]
            right = boundingboxes[j][2]
            top = boundingboxes[j][1]
            bottom = boundingboxes[j][3]
            pt = '{},{},{},{}'.format(left,top,right,bottom)
            pts.append(pt)

            faceId = str(uuid.uuid1())
            faceIds.append(faceId)
            picPaths.append(image_paths[i])

            facial_encodings[faceId] = vector

    return facial_encodings, features, faceIds, pts, picPaths


def get_onedir(path):

    files = paths.list_files(r''.join(path), validExts=(".jpg", ".jpeg", ".png"))
    dataset = [it for it in files]
    return dataset 


def main(args):
    """ Main

    Given a list of images, save out facial encoding data files and copy
    images into folders of face clusters.

    """
    from os.path import join, basename, exists
    from os import makedirs
    import numpy as np
    import shutil
    import sys

    if not exists(args.output):
        makedirs(args.output)

    image_paths = get_onedir(args.input)

    image_size = 160
    embedding_size = 128
    caffePrototxt = os.path.join(args.model_dir, 'resnetInception-128.prototxt')

    caffemodel = os.path.join(args.model_dir, 'inception_resnet_v1_conv1x1.caffemodel')
    net = caffe.Net(caffePrototxt, caffemodel, caffe.TEST)

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on images') 

    facial_encodings, features, faceIds, pts, picPaths = compute_facial_encodings(net, image_paths)
    sorted_clusters = cluster_facial_encodings(facial_encodings)
    num_cluster = len(sorted_clusters)

    for idx, cluster in enumerate(sorted_clusters):
        #all the cluster
        cluster_dir = join(args.output, str(idx))
        if not exists(cluster_dir):
            makedirs(cluster_dir)
        for faceId in cluster:
            ii = faceIds.index(faceId)
            img = cv_imread(picPaths[ii])
            pt = np.array(pts[ii].split(',')).astype(np.int32)

            faceArea = img[pt[1]:pt[3], pt[0]:pt[2]]
            cv2.imwrite(join(cluster_dir, faceId+'.jpg'), faceArea)


def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--model_dir', type=str, help='model dir', required=True)
    parser.add_argument('--input', type=str, help='Input dir of images', required=True)
    parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    main(parse_args())


    # import uuid
    # import sqlite3

    # conn = sqlite3.connect('facelive.db')
    # print("Opened database successfully")
    # c = conn.cursor()
    # c.execute('''CREATE TABLE FACELIVE
    #        (ID INTEGER PRIMARY KEY  AUTOINCREMENT,
    #        FaceId           VARCHAR(40)     NOT NULL,
    #        Feature          TEXT            NOT NULL,
    #        PicPath          VARCHAR(255)    NOT NULL,
    #        ClusterId        INT             NOT NULL,
    #        PT               VARCHAR(255)    NOT NULL);''')
    # print("Table created successfully")
    # conn.commit()

    # query = "INSERT INTO FACELIVE (FaceId,Feature,PicPath,ClusterId,PT) VALUES (?,?,?,?,?)"
    # columns = ['FaceId', 'Feature', 'PicPath', 'ClusterId', 'PT']

    # a={}
    # b={}
    # a['FaceId'] =  str(uuid.uuid1())
    # a['Feature'] = '1,2,3,4,5'
    # a['PicPath'] = '/data/a.jpg'
    # a['ClusterId'] = 1
    # a['PT'] = '10,20,30,40'

    # b['FaceId'] =  str(uuid.uuid1())
    # b['Feature'] = '2,1,1,1,1'
    # b['PicPath'] = '/data/b.jpg'
    # b['ClusterId'] = 2
    # b['PT'] = '101,210,1,140'

    # items = [a,b]

    # for data in items:
    #     keys = tuple(data[c] for c in columns)
    #     c = conn.cursor()
    #     c.execute(query, keys)
    #     c.close()

    # conn.commit()


    # conn = sqlite3.connect('facelive.db')
    # print("Opened database successfully")
    # query = "select * from FACELIVE"
    # for row in conn.execute(query):
    #     print(row)
