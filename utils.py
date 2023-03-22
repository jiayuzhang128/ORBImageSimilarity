import numpy as np
import glob
import os
import pickle
import cv2
import math
from matplotlib import pyplot as plt

def makeDirs(outputPath):
	if os.path.exists(outputPath + "keypoints/"):
		pass
	else:
		os.makedirs(outputPath + "keypoints/")
	if os.path.exists(outputPath + "descriptors/"):
		pass
	else:
		os.makedirs(outputPath + "descriptors/")
	if os.path.exists(outputPath + "resize/"):
		pass
	else:
		os.makedirs(outputPath + "resize/")
    
def imageLoad(samplePath, queryPath):
    # 读取样本图像
    sampleImageColor = cv2.imread(samplePath, cv2.IMREAD_UNCHANGED) 
    sampleImageColorResize = imageResize(cv2.cvtColor(sampleImageColor, cv2.COLOR_BGR2RGB))
    sampleImageGray = cv2.cvtColor(sampleImageColor, cv2.COLOR_RGB2GRAY)

    # 读取图库图像
    queryFileNames = glob.glob(queryPath + r"/*")
    queryImagesColor = []
    queryImagesColorResize = []
    queryImagesGray = []
    for name in queryFileNames:
        tempImageColor = cv2.imread(name, cv2.IMREAD_UNCHANGED)
        tempImageColorResize = imageResize(cv2.cvtColor(tempImageColor, cv2.COLOR_BGR2RGB))
        tempImageGray = cv2.cvtColor(tempImageColor,cv2.COLOR_RGB2GRAY)
        queryImagesGray.append(tempImageGray)
        queryImagesColor.append(tempImageColor)
        queryImagesColorResize.append(tempImageColorResize)
    numOfQuery = len(queryImagesColor)
    return sampleImageColor, sampleImageColorResize, sampleImageGray, queryImagesColor,queryImagesColorResize, queryImagesGray, numOfQuery, queryFileNames

def detectAndMatch(detector, matcher, sampleImage, queryImages, numOfQuery):
    # 寻找关键点和描述子
    keyPointsSample, descriptorsSample = detector.detectAndCompute(sampleImage, None)
    keyPointsQuery = []
    descriptorsQuery = []
    for image in queryImages:
        tempKeyPoints, tempDescriptors = detector.detectAndCompute(image, None)
        keyPointsQuery.append(tempKeyPoints)
        descriptorsQuery.append(tempDescriptors)

    # 特征点匹配与匹配点对筛选 
    matchesList =  []
    goodMatchesList = []
    for i in range(numOfQuery):
        goodMatches, matches = calculateMatches(descriptorsSample,descriptorsQuery[i], matcher)
        goodMatchesList.append(goodMatches)
        matchesList.append(matches)

    return keyPointsSample, descriptorsSample, keyPointsQuery, descriptorsQuery, matchesList, goodMatchesList
    
def calculateMatches(des1,des2, matcher, ratio=0.90):
    matches = matcher.knnMatch(np.float32(des1), np.float32(des2), k=2)
    goodMatches = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            goodMatches.append([m])
            
    return goodMatches, matches

def caculateSimilarity(goodMatches, matches):
    similarity = len(goodMatches)/len(matches)
    return similarity

def saveSimilarity(outputPath, best):
	np.savetxt(outputPath + "topSimilarity.txt", X=np.array([best]), fmt='%.4f')
    
def imageResize(image):
	heightD = 480
	height, width = (image.shape[0],image.shape[1])
	aspectRatio = width/height
	newSize = (int(heightD*aspectRatio),heightD)
	image = cv2.resize(image,newSize,interpolation=cv2.INTER_LINEAR)
	return image

def imageHstack(img1, img2):
	h1 = img1.shape[0]
	h2 = img2.shape[0]
	diffLines = np.abs(h1 - h2)
	if h1 > h2:
		img2Extend = cv2.copyMakeBorder(img2,0,diffLines,0,0,borderType = cv2.BORDER_CONSTANT,value=[255,255,255])
		hstack = cv2.hconcat((img1,img2Extend))
	else:
		img1Extend = cv2.copyMakeBorder(img1,0,diffLines,0,0,borderType = cv2.BORDER_CONSTANT,value=[255,255,255])
		hstack = cv2.hconcat((img1Extend,img2))
	return hstack

def imageVstack(img1, img2):
	w1 = img1.shape[1]
	w2 = img2.shape[1]
	diffLines = np.abs(w1 - w2)
	if w1 > w2:
		img2Extend = cv2.copyMakeBorder(img2,0, 0, 0, diffLines, borderType = cv2.BORDER_CONSTANT,value=[255,255,255])
		hstack = cv2.vconcat((img1,img2Extend))
	else:
		img1Extend = cv2.copyMakeBorder(img1,0,0,diffLines,0,borderType = cv2.BORDER_CONSTANT,value=[255,255,255])
		hstack = cv2.vconcat((img1Extend,img2))
	return hstack

def plotSampleAndQuery(images, simlilarity, numOfQuery, outputPath, figureName, isShow=False):
	assert len(images) == numOfQuery
	assert len(simlilarity) == numOfQuery
	imagesWithSimilarity = []
	for i in range(numOfQuery):
		imagesWithSimilarity.append((images[i], simlilarity[i]))
	imagesWithSimilarity.sort(key=lambda x:x[1], reverse=True)
	col = 4
	row = math.ceil(numOfQuery/col)
	figure, ax = plt.subplots(row, col)
	figure.canvas.set_window_title(figureName)
	figure.suptitle(figureName)
	for index, (img, sim)in enumerate(imagesWithSimilarity):
		axTemp = ax[int(index/col)][index%col]
		axTemp.set_title('Similarity: %.2f' %(sim*100) + r"%", fontsize=8)
		axTemp.axis("off")
		axTemp.imshow(img)
	plt.savefig(outputPath + figureName + ".png", dpi=300)
	if isShow:
		plt.show()

def plotKeypoints(sampleImage, keyPointsSample, queryImages, keyPointsQuery, similarity, numOfQuery, outputPath, figureName, isShow=False):
	keyPointsImageSample = cv2.drawKeypoints(sampleImage, keyPointsSample, outImage=None)
	keyPointsImagesSampleQuery = []
	for i in range(numOfQuery):
		tempKeyPointsImage = cv2.drawKeypoints(queryImages[i], keyPointsQuery[i], outImage = None)
		tempKeyPointsImageSQ = imageHstack(keyPointsImageSample, tempKeyPointsImage)
		keyPointsImagesSampleQuery.append(tempKeyPointsImageSQ)
	plotSampleAndQuery(keyPointsImagesSampleQuery, similarity, numOfQuery, outputPath, figureName, isShow=isShow)
	
def plotMatches(sampleImage, keyPointsSample, queryImages, keyPointsQuery, similarity, goodMatchesList, numOfQuery, outputPath, figureName, isShow=False):
	goodMatchesImages = []
	for i in range(numOfQuery):
		tempGoodMatchesImage = cv2.drawMatchesKnn(sampleImage, keyPointsSample, queryImages[i], keyPointsQuery[i], goodMatchesList[i], None, flags=2)
		goodMatchesImages.append(tempGoodMatchesImage)
	plotSampleAndQuery(goodMatchesImages, similarity, numOfQuery, outputPath, figureName, isShow=isShow)

def saveKeypointsAndDescriptors(keypoints,
                                kName,
                                descriptors,
                                dName,
                                outputPath):
	
	# 文件名
	filename1 = outputPath + 'keypoints\\' + kName + '.txt'
	filename2 = outputPath + 'descriptors\\' + dName + '.txt'

	# 删除一存在文件
	if os.path.exists(filename1):
		os.remove(filename1)

	elif os.path.exists(filename2):
		os.remove(filename2)

	strKeyPoints = []
	i = 0
	for point in keypoints:
		id = str(point) + '   -->index: ' + str(i)
		pt = point.pt
		size = point.size
		angle = point.angle
		response = point.response
		octave = point.octave
		class_id = point.class_id
		temp = [id , '\n\tpt: ' + str(pt) + '\n', '\tsize: ' + str(size) + '\n', '\tangle: ' + str(angle) + '\n', '\tresponse: ' + str(response) + '\n', '\toctave: ' + str(octave) + '\n', '\tclass_id: ' + str(class_id) + '\n']
		strKeyPoints.append(temp)
		i += 1

	# 保存特征点（str)
	np.savetxt(fname = filename1, X = strKeyPoints, fmt='%s')

	# 保存描述子（hex）
	np.savetxt(fname = filename2, X = descriptors, fmt = "%x")

def serializeKeypointsAndDescriptors(keypoints,
                                descriptors,
								matcher,
                                feature,
                                flags):
	# flags = 1: 训练图库中的图片
	# flags = 2: 待查询图片
	
	# File name
	filename1 = '/home/jiayu/desktop/project/orb/feature-detection-and-matching/src/results/keypoints/{}-with-{}-keypoints{}.pkl'.format(matcher, feature, flags)
	filename2 = '/home/jiayu/desktop/project/orb/feature-detection-and-matching/src/results/descriptors/{}-with-{}-descriptors{}.pkl'.format(matcher, feature, flags)

	# Delete a file if it exists
	if os.path.exists(filename1):
		os.remove(filename1)

	elif os.path.exists(filename2):
		os.remove(filename2)

	tempKeyPoints = []
	for point in keypoints:
		temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
		tempKeyPoints.append(temp)
	with open(filename1, 'wb') as fp:
		pickle.dump(tempKeyPoints, fp)

	with open(filename2, 'wb') as fp:
		pickle.dump(descriptors, fp)

def deserializeKeypointsAndDescriptors(keypointsPath, descriptorsPath):
	# flags = 1: 训练图库中的图片
	# flags = 2: 待查询图片
	
	keypoint = []
	file = open(keypointsPath,'rb')
	deserializedKeypoints = pickle.load(file)
	file.close()
	for point in deserializedKeypoints:
		temp = cv2.KeyPoint(
				x=point[0][0],
				y=point[0][1],
				size=point[1],
				angle=point[2],
				response=point[3],
				octave=point[4],
				class_id=point[5]
	)
		keypoint.append(temp)
	file = open(descriptorsPath,'rb')
	descriptor = pickle.load(file)
	file.close()
	return keypoint, descriptor

def calculateMatchesStrict(des1,des2, matcher, ratio=0.7):
    matches = matcher.knnMatch(np.float32(des2), np.float32(des1), k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            topResults1.append(m)
            
    matches = matcher.knnMatch(np.float32(des1), np.float32(des2), k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            topResults2.append(m)
    
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1.queryIdx
        match1TrainIndex = match1.trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2.queryIdx
            match2TrainIndex = match2.trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults, matches
