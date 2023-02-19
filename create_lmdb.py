#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 18:44:21 2019

@author: fhk
"""

import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import re
from PIL import Image
import numpy as np
import imghdr


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            #            print(k)
            if isinstance(k, str):
                k = k.encode('utf-8')
            if isinstance(v, str):
                v = v.encode('utf-8')
            txn.put(k, v)

# 这注释部分是原来的
# def writeCache(env, cache):
#     with env.begin(write=True) as txn:
#         for k, v in cache.items():
#             txn.put(k, v)

def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=10995188)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = ''.join(imagePathList[i]).split()[0].replace('\n', '').replace('\r\n', '')
        label = ''.join(labelList[i])
        print(label)

        # 这块可能要改
        with open(imagePath, 'r') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
        print(cnt)
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    #LMDB文件输出路径 按需改，执行两次，一次train,一次test
    outputPath = r'./train'   
    # outputPath = r'./test'

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    imgdata = open(r"./train.txt", "rt")
    # imgdata = open(r"./test.txt", "rt")
    imagePathList = list(imgdata)

    labelList = []
    for line in imagePathList:
        word = line.split()[1]
        labelList.append(word)

    createDataset(outputPath, imagePathList, labelList)

