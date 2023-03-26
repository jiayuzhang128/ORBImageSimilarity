import argparse
from utils import *

parser = argparse.ArgumentParser(
                    prog='ORBImageSimilarity',
                    description='Estimate similarity between sample image and query images and find best match')

parser.add_argument(
                    '-s', '--samplePath',
                    type=str,
                    required=True,
                    help='Location of sample image using absolute path. ie. C:\\user\\Desktop\\sample\\sample.png')

parser.add_argument(
                    '-q', '--queryPath',
                    type=str,
                    required=True,
                    help='Location of query image using absolute path. ie. C:\\user\\Desktop\\query\\')

parser.add_argument(
                    '-o', '--outputPath',
                    type=str,
                    required=True,
                    help='Location of query image using absolute path. ie. C:\\user\\Desktop\\output\\')

parser.add_argument(
                    '-i', '--isShow',
                    type=bool,
                    required=False,
                    default=False,
                    help='Show results or not. Default: Flase')

args = parser.parse_args()
assert args.samplePath.split(".")[-1] in ['png', 'bmp', 'jpg', 'jpeg', 'tiff']
assert args.queryPath[-1] == "\\"
assert args.outputPath[-1] == "\\"

# load images
sampleImageColorOrigin, sampleImageColorResize, sampleImageGray, queryImagesColorOrigin, queryImagesColorResize, queryImagesGray, numOfQuery, queryFileNames = imageLoad(args.samplePath, args.queryPath)

# initialize ORB detector
orb = cv2.ORB_create()

# initialize FLANN matcher
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)

# detect and match
keyPointsSample, descriptorsSample, keyPointsQuery, descriptorsQuery, matchesList, goodMatchesList = detectAndMatch(orb, flann, sampleImageGray, queryImagesGray, numOfQuery)

# caculate similarity
similarity = []
for i in range(numOfQuery):
    tempSimilarity = caculateSimilarity(goodMatchesList[i], matchesList[i])
    # print(tempSimilarity)
    similarity.append(tempSimilarity)
maxSimilarity = max(similarity)
bestIndex = similarity.index(maxSimilarity)
# print(bestIndex)
# print(similarity)

# make dirs for outputs
outputPath = args.outputPath
makeDirs(outputPath)

# save resized image
resizePath = outputPath + "resize\\"
sampleImageName = args.samplePath.split("\\")[-1].split('.')[-2]
cv2.imwrite(resizePath + sampleImageName + "resize" + ".png", cv2.cvtColor(sampleImageColorResize, cv2.COLOR_RGB2BGR))
queryImageNames = []
for i in range(numOfQuery):
    queryName = queryFileNames[i].split('\\')[-1].split('.')[-2]
    cv2.imwrite(resizePath + queryName + "Resize" + ".png", cv2.cvtColor(queryImagesColorResize[i],cv2.COLOR_RGB2BGR))
    queryImageNames.append(queryName)

# save best matching image and top similarity
cv2.imwrite(outputPath + "BestMatch.png", queryImagesColorOrigin[bestIndex])
saveSimilarity(outputPath, maxSimilarity)

# visualize keypoints and matches
figureNameKeypoints = "FinalKeypointsWithSimilarity"
plotKeypoints(sampleImageColorResize, keyPointsSample, queryImagesColorResize, keyPointsQuery, similarity, numOfQuery, outputPath, figureNameKeypoints,isShow=args.isShow)

figureNameMatches = "FinalMatchesWithSimilarity"
plotMatches(sampleImageColorResize,keyPointsSample, queryImagesColorResize, keyPointsQuery, similarity, goodMatchesList, numOfQuery, outputPath, figureNameMatches, isShow=args.isShow)

# save keypoints and descriptors
saveKeypointsAndDescriptors(
                            keypoints=keyPointsSample,
                            kName="keyPoints" + sampleImageName,
                            descriptors=descriptorsSample,
                            dName="descriptors" + sampleImageName,
                            outputPath=outputPath)
for i in range(numOfQuery):
    saveKeypointsAndDescriptors(
                                keypoints=keyPointsQuery[i],
                                kName="keyPoints" + queryImageNames[i],
                                descriptors=descriptorsQuery[i],
                                dName="descriptors" + queryImageNames[i],
                                outputPath=outputPath)

print("Estimate similarity done!")
    