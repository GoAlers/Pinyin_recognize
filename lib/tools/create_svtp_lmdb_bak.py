import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm
import six
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import re

def checkImageIsValid(imageBin):
  if imageBin is None:
    return False
  imageBuf = np.fromstring(imageBin, dtype=np.uint8)
  img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
  imgH, imgW = img.shape[0], img.shape[1]
  if imgH * imgW == 0:
    return False
  return True


def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      txn.put(k.encode(), v)


def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
  """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath    : LMDB output path
      imagePathList : list of image path
      labelList     : list of corresponding groundtruth texts
      lexiconList   : (optional) list of lexicon lists
      checkValid    : if true, check the validity of every image
  """
  assert(len(imagePathList) == len(labelList))
  nSamples = len(imagePathList)
  env = lmdb.open(outputPath, map_size=109951162777*0.01)
  cache = {}
  cnt = 1
  for i in range(nSamples):
    imagePath = imagePathList[i]
    label = labelList[i]
    if len(label) == 0:
      continue
    if not os.path.exists(imagePath):
      print('%s does not exist' % imagePath)
      continue
    with open(imagePath, 'rb') as f:
      imageBin = f.read()
    if checkValid:
      if not checkImageIsValid(imageBin):
        print('%s is not a valid image' % imagePath)
        continue

    imageKey = 'image-%09d' % cnt
    labelKey = 'label-%09d' % cnt
    cache[imageKey] = imageBin
    cache[labelKey] = label.encode()
    if lexiconList:
      lexiconKey = 'lexicon-%09d' % cnt
      cache[lexiconKey] = ' '.join(lexiconList[i])
    if cnt % 1000 == 0:
      writeCache(env, cache)
      cache = {}
      print('Written %d / %d' % (cnt, nSamples))
    cnt += 1
  nSamples = cnt-1
  cache['num-samples'] = str(nSamples).encode()
  writeCache(env, cache)
  print('Created dataset with %d samples' % nSamples)

if __name__ == "__main__":

  # lmdb_output_path1 = 'mvtest'
  # with open('D:\Python\PycharmProjects\CRNN\dataset\manwen\mw_test_999.txt', 'r') as f:
  #   lines = [line.strip('\n') for line in f.readlines()]

  # lmdb_output_path2 = 'mwtrain'
  # with open('D:\Python\PycharmProjects\CRNN\dataset\manwen\mw_train_999.txt', 'r') as f:
  #   lines = [line.strip('\n') for line in f.readlines()]


  #下面两个是横向，报错调map_size=109951162777*0.01,原来109951162777*0.005
  # lmdb_output_path1 = 'mvtest_heng'
  # with open('D:\Python\PycharmProjects\CRNN\dataset\manwen\mw_test_999_heng.txt', 'r') as f:
  #   lines = [line.strip('\n') for line in f.readlines()]

  lmdb_output_path2 = 'mwtrain_heng'
  with open('D:\Python\PycharmProjects\CRNN\dataset\manwen\mw_train_999_heng.txt', 'r') as f:
    lines = [line.strip('\n') for line in f.readlines()]

  imagePathList, labelList = [], []
  for i, line in enumerate(lines):
    splits = line.split(' ')
    image_name = splits[0]+'.png'
    gt_text = splits[1]
    print(image_name, gt_text)
    imagePathList.append(image_name)
    labelList.append(gt_text)

  # createDataset(lmdb_output_path1, imagePathList, labelList)
  createDataset(lmdb_output_path2, imagePathList, labelList)