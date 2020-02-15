# implementation of Region-VLAD VPR framework
# Ahmad Khaliq
# ahmedest61@hotmail.com

from _collections import defaultdict
import caffe
import pickle
import numpy as np
from skimage.measure import regionprops,label
import itertools
import time


# Load pickle file
def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

# Read mean file
def binaryProto2npy(binaryProtoMeanFile):

     blob = caffe.proto.caffe_pb2.BlobProto()
     data = open( binaryProtoMeanFile, 'rb' ).read()
     blob.ParseFromString(data)
     data = np.array(blob.data)
     arr = np.array( caffe.io.blobproto_to_array(blob) )
     return arr[0]   

# Extract N ROIs from the conv layer
def getROIs(imgConvFeat,imgLocalConvFeat,img):
    
    clustersEnergies_Ej = []
    clustersBoxes  = []
    allROI_Box = [] 
    aggregatedROIs = []
    
    for featuremap in imgConvFeat:                         
        clusters = regionprops(label(featuremap),intensity_image=featuremap,cache=False);
        clustersBoxes.append(list(cluster.bbox for cluster in clusters))
        clustersEnergies_Ej.append(list(cluster.mean_intensity for cluster in clusters))
    # Make a list of ROIs with their bounded boxes
    clustersBoxes = list(itertools.chain.from_iterable(clustersBoxes))
    clustersEnergies_Ej = list(itertools.chain.from_iterable(clustersEnergies_Ej))
    # Sort the ROIs based on energies
    allROIs = sorted(clustersEnergies_Ej,reverse=True)
    # Pick up top N energetic ROIs with their bounding boxes
    allROIs = allROIs[:N]
    allROI_Box = [clustersBoxes[clustersEnergies_Ej.index(i)] for i in allROIs]
    clustersEnergies_Ej.clear()
    clustersBoxes.clear()
    aggregatedNROIs = np.zeros((N,imgLocalConvFeat.shape[2]))
    # Retreive the aggregated local descriptors lying under N ROIs
    for ROI in range(len(allROI_Box)):
  #      minRow, minCol, maxRow, maxCol = allROI_Box[ROI][0],allROI_Box[ROI][1],allROI_Box[ROI][2],allROI_Box[ROI][3] 
        aggregatedNROIs[ROI,:] = np.sum(imgLocalConvFeat[allROI_Box[ROI][0]:allROI_Box[ROI][2],allROI_Box[ROI][1]:allROI_Box[ROI][3]],axis=(0,1))                                         
    # NxK dimensional ROIs
    return np.asarray(aggregatedNROIs)

# Retreive the VLAD representation
def getVLAD(X,visualDictionary):
    
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters
   
    m,d = X.shape
    Vlad=np.zeros([k,d])
    #computing the differences
    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            Vlad[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)  
    Vlad = Vlad.flatten()
    Vlad = np.sign(Vlad)*np.sqrt(np.abs(Vlad))
    Vlad = Vlad/np.sqrt(np.dot(Vlad,Vlad))        
    Vlad = Vlad.reshape(k,d)    
    return Vlad 


# Load protext, model and mean file
protxt = "../AlexnetPlaces365/deploy_alexnet_places365.prototxt"
model = "../AlexnetPlaces365/alexnet_places365.caffemodel"
mean = "../AlexnetPlaces365/places365CNN_mean.binaryproto"

# Set ROIs
N = 400
# Set Layer
layer = 'conv3'
Features,StackedFeatures = defaultdict(list),defaultdict(list)
# Use GPU?
set_gpu =True
gpu_id = 0

totalT = 0

if set_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()

net = caffe.Net(protxt, model, caffe.TEST)
batch_size = 2
inputSize = net.blobs['data'].shape[2]
net.blobs['data'].reshape(batch_size,3,inputSize,inputSize)

# Test image
testImg = "../berlin_A100/berlin_A100_1/04FQ4NG5mYXOdbENCPup8w-640-00000.jpg"

# Ref image
refImg = "../berlin_A100/berlin_A100_2/HRdLsyO_U_0WmgrIZCKZ0g-640-00266.jpg"

image_paths_list = []
image_paths_list.append(testImg)
image_paths_list.append(refImg)

# Configuration 1
if N==200:
    V = 128
    vocab = load_obj("../Vocabulary_100_200_300.pkl")
# Configuration 2
else:
    V = 256
    vocab = load_obj("../Vocabulary_400.pkl")

# Load Images
images_loaded_by_caffe = [caffe.io.load_image(im) for im in image_paths_list] 
images_loaded_by_caffe = np.array(images_loaded_by_caffe)  

# Set Caffe
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
mean_file = binaryProto2npy(mean)
transformer.set_mean('data', mean_file.mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255)

data_blob_index = range(batch_size)
net.blobs['data'].data[data_blob_index] = \
[transformer.preprocess('data', img) for img in images_loaded_by_caffe]

print("*****Region-VLAD*****")
print("*****N: %d, V=%d*****" %(N,V))

ts=time.time()
# Forward Pass
res = net.forward()
te=time.time()
#print("Fwd Pass Time: %f ms" %((te-ts)*1000/float(batch_size)))
totalT +=(te-ts)*1000/float(batch_size)

ts=time.time()
# Stack the activations of feature maps to make local descriptors
Features[layer] = np.array(net.blobs[layer].data[data_blob_index].copy())
StackedFeatures[layer]=Features[layer].transpose(0,2,3,1)
# Retrieve N ROIs for test and ref images
testROIs= getROIs(Features[layer][0],StackedFeatures[layer][0],testImg)
refROIs= getROIs(Features[layer][1],StackedFeatures[layer][1],refImg)
te=time.time()

#print("Encoding Time: %f ms" %((te-ts)*1000/float(batch_size)))
totalT +=(te-ts)*1000/float(batch_size)

vocabulary = vocab[N][V][layer]

ts=time.time()  
# Retrieve VLAD descriptors using ROIs and vocabulary
testVLAD= getVLAD(testROIs,vocabulary)
refVLAD= getVLAD(refROIs,vocabulary)
te=time.time()

#print("VLAD Encoding Time: %f ms" %((te-ts)*1000/float(batch_size)))
totalT +=(te-ts)*1000/float(batch_size)

ts=time.time()
# Dot Prouct of test and ref VLADs
cosineMatchScore = np.sum(np.einsum('ij,ij->i', testVLAD, refVLAD))
te=time.time()
#print("VLAD Matching Time: %f ms" %((te-ts)*1000))
matchTime=(te-ts)*1000

print ("Score: %f" %cosineMatchScore)

#print("Query time til VLAD encoding :%f ms" %totalT)
#print("Retrieval Time 1 Test vs 1 Ref :%f ms" %(totalT+matchTime))
#print("Retrieval Time 1 Test vs 750 Ref :%f ms" %(totalT+750*matchTime))


    
