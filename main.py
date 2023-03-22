import argparse
from utils import *

parser = argparse.ArgumentParser(
                    prog='ORBImageSimilarity',
                    description='Estimate similarity between sample image and query images and find best match')

parser.add_argument(
                    '-s', '--samplePath',
                    type=str,
                    required=True,
                    default='/home/jiayu/Desktop/project/imageSimilarity/data/sample/sample4.png',
                    help='Location of sample image using absolute path. ie. C:/user/Desktop/sample/sample.png')

parser.add_argument(
                    '-q', '--queryPath',
                    type=str,
                    required=True,
                    default='/home/jiayu/Desktop/project/imageSimilarity/data/query/',
                    help='Location of query image using absolute path. ie. C:/user/Desktop/query/')

parser.add_argument(
                    '-o', '--outputPath',
                    type=str,
                    required=True,
                    default="/home/jiayu/Desktop/project/imageSimilarity/output/",
                    help='Location of query image using absolute path. ie. C:/user/Desktop/output/')

parser.add_argument(
                    '-i', '--isShow',
                    type=bool,
                    required=False,
                    default=False,
                    help='Show matches or not.')

args = parser.parse_args()

# 读取图片
sampleImageColorOrigin, sampleImageColorResize, sampleImageGray, queryImagesColorOrigin, queryImagesColorResize, queryImagesGray, numOfQuery, queryFileNames = imageLoad(args.samplePath, args.queryPath)

# 初始化ORB检测对象
orb = cv2.ORB_create()

# 初始化FLANN匹配对象
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)

# 检测与匹配
keyPointsSample, descriptorsSample, keyPointsQuery, descriptorsQuery, matchesList, goodMatchesList = detectAndMatch(orb, flann, sampleImageGray, queryImagesGray, numOfQuery)

# 计算相似度
similarity = []
for i in range(numOfQuery):
    tempSimilarity = caculateSimilarity(goodMatchesList[i], matchesList[i])
    # print(tempSimilarity)
    similarity.append(tempSimilarity)
maxSimilarity = max(similarity)
bestIndex = similarity.index(maxSimilarity)
# print(bestIndex)
# print(similarity)

# 创建输出目录
outputPath = args.outputPath
makeDirs(outputPath)

# 保存缩放图片
resizePath = outputPath + "resize/"
sampleImageName = args.samplePath.split("/")[-1].split('.')[-2]
cv2.imwrite(resizePath + sampleImageName + "resize" + ".png", cv2.cvtColor(sampleImageColorResize, cv2.COLOR_RGB2BGR))
queryImageNames = []
for i in range(numOfQuery):
    queryName = queryFileNames[i].split('/')[-1].split('.')[-2]
    cv2.imwrite(resizePath + queryName + "Resize" + ".png", cv2.cvtColor(queryImagesColorResize[i],cv2.COLOR_RGB2BGR))
    queryImageNames.append(queryName)

# 保存最佳匹配和最高相似度
cv2.imwrite(outputPath + "BestMatch.png", queryImagesColorOrigin[bestIndex])
saveSimilarity(outputPath, maxSimilarity)

# 显示关键点和匹配情况
## 绘制关键点
figureNameKeypoints = "FinalKeypointsWithSimilarity"
plotKeypoints(sampleImageColorResize, keyPointsSample, queryImagesColorResize, keyPointsQuery, similarity, numOfQuery, outputPath, figureNameKeypoints,isShow=args.isShow)

# 显示匹配点对和相似度
figureNameMatches = "FinalMatchesWithSimilarity"
plotMatches(sampleImageColorResize,keyPointsSample, queryImagesColorResize, keyPointsQuery, similarity, goodMatchesList, numOfQuery, outputPath, figureNameMatches, isShow=args.isShow)

# 序列化关键点和描述子
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
    