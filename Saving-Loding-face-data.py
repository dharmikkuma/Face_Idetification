from keras.models import load_model
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from os import listdir
from matplotlib import pyplot
import os
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import matplotlib
import matplotlib.patches as patches





# #############################################################################
# Extract face pixels from a given image.


def faceExt(imgpath, size=(160, 160)):
  img = Image.open(imgpath)
  img = img.convert('RGB')
  pix = np.asarray(img)
  detector = MTCNN()
  results = detector.detect_faces(pix)
  x1, y1, wd, ht = results[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1+wd, y1+ht
  # print('ordinates')
  #print(x1,y1,x2,y2)
  face = pix[y1:y2, x1:x2]
  img = Image.fromarray(face)
  img = img.resize(size)
  farray = np.asarray(img)
  return farray



#############################
  # Load faces from images stored in the given directory.
def load_faces(dir):
  faces = []
  for file in listdir(dir):
    path = dir+'\\'+file
    try:
      face = faceExt(path)
    except Exception as er:
      print(file)
    faces.append(face)
  return faces
  ###################################

  # Provide the path for the dataset directory to extract faces from the dataset using the aforementioned functions.


def load_dataset(directory):
  x = []
  y = []
  #print(listdir(directory))
  for subdir in listdir(directory):
    print(subdir)
    path = directory + '\\' + subdir
    print(path)
    if not os.path.isdir(path):
      continue
    try:
      faces = load_faces(path)
    except Exception as er:
      print(er)
    labels = [subdir for _ in range(len(faces))]
    print('>loaded %d examples for class: %s' % (len(faces), subdir))
    x.extend(faces)
    y.extend(labels)
  return np.asarray(x), np.asarray(y)
  ####################################

trainX, trainy = load_dataset(
      'E:\\Coursera\\Computer Vision\\OpenCV\\FACENET\\FaceNet-Implementation-master\\train')  # path for train dataset


print(trainX.shape, trainy.shape)
testX, testy = load_dataset(
    'E:\\Coursera\\Computer Vision\\OpenCV\\FACENET\\FaceNet-Implementation-master\\val')  # path for test dataset
# Save the extracted faces from the dataset as a compressed file.
savez_compressed('faces_dataset.npz', trainX, trainy, testX, testy)
################################
data = load('faces_dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
