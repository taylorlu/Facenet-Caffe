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
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
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

def mtcnnDetect(image, imgpath_list):

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

        warped = image[rectify_y1:y2, rectify_x1:x2]

        ct = time.time()
        local_time = time.localtime(ct)
        data_head = time.strftime("%Y%m%d%H%M%S", local_time)
        data_secs = (ct - int(ct)) * 1000
        time_stamp = "%s.%03d" % (data_head, data_secs)

        imgpath = os.path.join('face_dir', time_stamp+'.jpg')
        cv2.imwrite(imgpath, warped)
        imgpath_list.append(imgpath)

    return imgpath_list

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
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """

    #from face_recognition.api import _face_distance

    G = nx.Graph()

    # Create graph
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx+1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
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
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest

    """

    if len(facial_encodings) <= 1:
        print ("Number of facial encodings must be greater than one, can't cluster")
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings.items())
    return sorted_clusters

def compute_facial_encodings(net,image_size,
                    embedding_size,nrof_images,nrof_batches,emb_array,batch_size,paths):
    """ Compute Facial Encodings

        Given a set of images, compute the facial encodings of each face detected in the images and
        return them. If no faces, or more than one face found, return nothing for that image.

        Inputs:
            image_paths: a list of image paths

        Outputs:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

    """

    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        vectors = []

        for i in range(len(paths_batch)):
            img = misc.imread(paths_batch[i])   # BGR
            vector = calcCaffeVector(net,img)
            vectors.append(vector)

        emb_array[start_index:end_index,:] = np.array(vectors)

    facial_encodings = {}
    for x in range(nrof_images):
        facial_encodings[paths[x]] = emb_array[x,:]


    return facial_encodings

def get_onedir(paths):
    dataset = []
    path_exp = os.path.expanduser(paths)
    if os.path.isdir(path_exp):
        images = os.listdir(path_exp)
        image_paths = [os.path.join(path_exp,img) for img in images]

        for x in image_paths:
            if os.path.getsize(x)>0:
                dataset.append(x)
        
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
    imgpath_list = []
    for imgpath in image_paths:
        img = cv2.imread(imgpath)
        imgpath_list = mtcnnDetect(img, imgpath_list)

    image_size = 160
    embedding_size = 128
    caffePrototxt = os.path.join(args.model_dir, 'resnetInception-128.prototxt')

    caffemodel = os.path.join(args.model_dir, 'inception_resnet_v1_conv1x1.caffemodel')
    net = caffe.Net(caffePrototxt, caffemodel, caffe.TEST)

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on images') 

    nrof_images = len(imgpath_list)
    nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    facial_encodings = compute_facial_encodings(net,image_size,
        embedding_size,nrof_images,nrof_batches,emb_array,args.batch_size,imgpath_list)
    sorted_clusters = cluster_facial_encodings(facial_encodings)
    num_cluster = len(sorted_clusters)
        
    # Copy image files to cluster folders
    for idx, cluster in enumerate(sorted_clusters):
        #save all the cluster
        cluster_dir = join(args.output, str(idx))
        if not exists(cluster_dir):
            makedirs(cluster_dir)
        for path in cluster:
            shutil.copy(path, join(cluster_dir, basename(path)))


def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--model_dir', type=str, help='model dir', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', required=30)
    parser.add_argument('--input', type=str, help='Input dir of images', required=True)
    parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """ Entry point """
    main(parse_args())
