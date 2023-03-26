import numpy as np
import glob
import os
import cv2
import math
from matplotlib import pyplot as plt

def makeDirs(outputPath):
	"""makedirs for outputs, including keypoints descriptors and resize images.

	Args:
		outputPath (str): the output path.
	"""
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
	"""Load sample image and query images.

	Args:
		samplePath (str): sample image path.
		queryPath (str): query images path.

	Returns:
		tuple: returns sample image and query images in RGB resized-RGB resized-Gray; number of query images; file name of query images.
	"""
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
	"""Detect ORB keypoints/descriptors and match then.

	Args:
		detector (cv2.ORB_create): opencv orb detector
		matcher (cv2.matcher): opencv keypoints matcher, BF or FLANN
		sampleImage (ndarray): sample image 
		queryImages (list): list of query images
		numOfQuery (int): number of query images

	Returns:
		tuple: returns keypoints and descriptors of sample image and query images; matching results and filtered matching results.
	"""
	# detect keypoints and descriptors
	keyPointsSample, descriptorsSample = detector.detectAndCompute(sampleImage, None)
	keyPointsQuery = []
	descriptorsQuery = []
	for image in queryImages:
		tempKeyPoints, tempDescriptors = detector.detectAndCompute(image, None)
		keyPointsQuery.append(tempKeyPoints)
		descriptorsQuery.append(tempDescriptors)

	# keypoints matching and matching point pair filtering
	matchesList =  []
	goodMatchesList = []
	for i in range(numOfQuery):
		goodMatches, matches = calculateMatches(descriptorsSample,descriptorsQuery[i], matcher)
		goodMatchesList.append(goodMatches)
		matchesList.append(matches)

	return keyPointsSample, descriptorsSample, keyPointsQuery, descriptorsQuery, matchesList, goodMatchesList
    
def calculateMatches(des1,des2, matcher, ratio=0.90):
	"""Matching keypoints by compare descriptors between two images, and filtering raw matching results.

	Args:
		des1 (cv2.descriptor): descriptors1.
		des2 (cv2.descriptor): descriptors2.
		matcher (cv2.matcher): opencv keypoints matcher, BF or FLANN.
		ratio (float, optional): filtering ratio. Defaults to 0.90.

	Returns:
		tuple: returns raw matching results and filtered matching results.
	"""
	matches = matcher.knnMatch(np.float32(des1), np.float32(des2), k=2)
	goodMatches = []
	for m,n in matches:
		if m.distance < ratio*n.distance:
			goodMatches.append([m])
            
	return goodMatches, matches

def caculateSimilarity(goodMatches, matches):
	"""Caculate similarity

	Args:
		goodMatches (cv2.DMatch): filtered matching results
		matches (cv2.DMatch): raw matching results

	Returns:
		float: similarity of two inages.
	"""
	similarity = len(goodMatches)/len(matches)
	return similarity

def saveSimilarity(outputPath, best):
	"""Save similarity to txt file.

	Args:
		outputPath (str): output path.
		best (float): top similarity.
	"""
	np.savetxt(outputPath + "topSimilarity.txt", X=np.array([best]), fmt='%.4f')
    
def imageResize(image, height=480):
	"""Resize image into same height.

	Args:
		image (ndarray): image.

	Returns:
		ndarray: resized image.
	"""
	heightD = height
	height, width = (image.shape[0],image.shape[1])
	aspectRatio = width/height
	newSize = (int(heightD*aspectRatio),heightD)
	image = cv2.resize(image,newSize,interpolation=cv2.INTER_LINEAR)

	return image

def imageHstack(img1, img2):
	"""Stack image by height.

	Args:
		img1 (ndarray): image1.
		img2 (ndarray): image2.

	Returns:
		ndarray: stacked image.
	"""
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
	"""Stack image by width.

	Args:
		img1 (ndarray): image1.
		img2 (ndarray): image2.

	Returns:
		ndarray: stacked image.
	"""
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
	"""Visualize results.

	Args:
		images (list): images to visualize.
		simlilarity (list): similarity of the corresponding image.
		numOfQuery (int): number of image list(query images).
		outputPath (str): output path.
		figureName (str): name of visualized figure.
		isShow (bool, optional): show figure or not. Defaults to False.
	"""
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
	"""Visualize keypoints results.

	Args:
		sampleImage (ndarray): sample image.
		keyPointsSample (cv2.keypoints): keypoints of sample image.
		queryImages (list): images of query images.
		keyPointsQuery (list): keypoints of query images
		similarity (lists): similarity of the corresponding image.
		numOfQuery (int): number of image list(query images).
		outputPath (str): output path.
		figureName (str): name of visualized figure.
		isShow (bool, optional): show figure or not. Defaults to False.
	"""
	keyPointsImageSample = cv2.drawKeypoints(sampleImage, keyPointsSample, outImage=None)
	keyPointsImagesSampleQuery = []
	for i in range(numOfQuery):
		tempKeyPointsImage = cv2.drawKeypoints(queryImages[i], keyPointsQuery[i], outImage = None)
		tempKeyPointsImageSQ = imageHstack(keyPointsImageSample, tempKeyPointsImage)
		keyPointsImagesSampleQuery.append(tempKeyPointsImageSQ)
	plotSampleAndQuery(keyPointsImagesSampleQuery, similarity, numOfQuery, outputPath, figureName, isShow=isShow)
	
def plotMatches(sampleImage, keyPointsSample, queryImages, keyPointsQuery, similarity, goodMatchesList, numOfQuery, outputPath, figureName, isShow=False):
	"""Visualize matching results.

	Args:
		sampleImage (ndarray): sample image.
		keyPointsSample (cv2.keypoints): keypoints of sample image.
		queryImages (list): images of query images.
		keyPointsQuery (list): keypoints of query images
		similarity (lists): similarity of the corresponding image.
		goodMatchesList (list): filtered matching results. 
		numOfQuery (int): number of image list(query images).
		outputPath (str): output path.
		figureName (str): name of visualized figure.
		isShow (bool, optional): show figure or not. Defaults to False.
	"""
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
	"""Save keypoints and descriptors into txt files.

	Args:
		keypoints (cv2.keypoints): keypoints.
		kName (str): txt file name of keypoints
		descriptors (cv2.descriptors): descriptors
		dName (str): txt file name of descriptors.
		outputPath (str): output path.
	"""
	# filename
	filename1 = outputPath + 'keypoints\\' + kName + '.txt'
	filename2 = outputPath + 'descriptors\\' + dName + '.txt'

	# delete the existing files
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

	# save keypoints into txt file（format:str)
	np.savetxt(fname = filename1, X = strKeyPoints, fmt='%s')

	# save descriptors into txt file（format:hex）
	np.savetxt(fname = filename2, X = descriptors, fmt = "%x")


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
