# Process raw data and save them into pickle file.
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
from scipy import misc
import scipy.io
from skimage import io
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation
from constants import *
import imageio as imio

img_size = INPUT_SIZE
salmap_size = INPUT_SIZE


number = 1

for image, map in zip(sorted(os.listdir(pathToImages)), sorted(os.listdir(pathToMaps))):

    if number <= 800:
        os.rename(os.path.join(pathToImages, image), os.path.join(pathToImages, image[:4] + 'train' + image[-4:]))
        os.rename(os.path.join(pathToMaps, image[:-4] + '.png'), os.path.join(pathToMaps, image[:4] + 'train' + '.png'))
    elif number > 800 and number <= 1000:
        os.rename(os.path.join(pathToImages, image), os.path.join(pathToImages, image[:4] + 'val' + image[-4:]))
        os.rename(os.path.join(pathToMaps, image[:-4] + '.png'), os.path.join(pathToMaps, image[:4] + 'val' + '.png'))
    number += 1


# Resize train/validation files

listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToMaps, '*'))]
# listTestImages = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*test*'))]

for currFile in (listImgFiles):
    # # if tt.image.getImage().shape[:2] != (480, 640):
    # #    print 'Error:', currFile
    image = cv2.imread(pathToImages + currFile + '.jpg')
    imageResized = cv2.cvtColor(cv2.resize(image, img_size, interpolation=cv2.INTER_AREA),
                                cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(pathOutputImages, currFile + '.png'), imageResized)

for currFile in os.listdir(pathToMaps):
    map = cv2.imread(pathToMaps + currFile)
    saliencyResized = cv2.resize(map, salmap_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(pathOutputMaps, currFile[:-4] + '.png'), saliencyResized)

# LOAD DATA

# Train

listFilesTrain = [k for k in listImgFiles if 'train' in k]
trainData = []
for currFile in tqdm(listFilesTrain):
    trainData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                               os.path.join(pathOutputMaps, currFile + '.png'),
                                               # os.path.join(pathToFixationMaps, currFile + '.mat'),
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale))

with open(os.path.join(pathToPickle, 'trainData.pickle'), 'wb') as f:
    pickle.dump(trainData, f)

# Validation

listFilesValidation = [k for k in listImgFiles if 'val' in k]
validationData = []
for currFile in tqdm(listFilesValidation):
    validationData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                                    os.path.join(pathOutputMaps, currFile + '.png'),
                                                    # os.path.join(pathToFixationMaps, currFile + '.mat'),
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale))

with open(os.path.join(pathToPickle, 'validationData.pickle'), 'wb') as f:
    pickle.dump(validationData, f)
