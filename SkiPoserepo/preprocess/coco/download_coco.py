import os
from pycocotools.coco import COCO
import numpy as np
import cv2
# import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import requests
from pycocotools.coco import COCO
import requests
import csv

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

dataDir='.'
dataType='val2017'
annFile='%s/instances_%s.json'%(dataDir,dataType)

dataset_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/SportTech/COCO_Sport/Validation'
save_label_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/SportTech/COCO_Sport/Validation_Label'

coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person','tennis racket'])
imgIds = coco.getImgIds(catIds=catIds )
images = coco.loadImgs(imgIds)
print("imgIds: ", imgIds)
print("len imgIds: ", len(imgIds))
# print("images: ", images)

for im in images:
    print("im: ", im)
    img_data = requests.get(im['coco_url']).content
    with open(os.path.join(dataset_path,im['file_name']), 'wb') as handler:
        handler.write(img_data)

