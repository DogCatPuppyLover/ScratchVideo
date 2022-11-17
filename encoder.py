import math
import numpy as np
from random import random
import cv2


def findDiff(img1, img2):
    return np.average(np.abs(np.subtract(img1, img2, dtype=np.int16)))


def randomBetween(x, y):
    return int(x + (random() * (y - x)))


def clippedRandomBetween(x, y, clip1, clip2):
    clippedX = np.clip(x, clip1, clip2)
    clippedY = np.clip(y, clip1, clip2)
    return int(clippedX + (random() * (clippedY - clippedX)))


print("Video path:")

videoPath = input()

print("Converted resolution:")

resolution = int(input())

print("Lines per frame:")

numLines = int(input())

print("Number of tests per line:")

testLines = int(input())

print("Number of mutations per line:")

numMutate = int(input())

print("Converted framerate:")

convertedFramerate = int(input())

width = int(480 / resolution)

height = int(360 / resolution)

cap = cv2.VideoCapture(videoPath)

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

framerate = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter("output.mp4", fourcc, 5, (width, height))

canvas = np.zeros((height, width, 3), dtype="uint8")

print(frameCount)

print(framerate)

totalConvertedFrames = int(frameCount * (convertedFramerate / framerate))

videoData = ""

for frameI in range(totalConvertedFrames):

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frameI * (framerate / convertedFramerate)))

    ret, frame = cap.read()

    if ret == True:

        baseImg = cv2.resize(frame, (width, height))

        canvasDiff = findDiff(baseImg, canvas)

        for x in range(numLines):

            canvasDiff = findDiff(baseImg, canvas)

            adjustedCanvasDiff = int(canvasDiff / resolution)

            diffs = np.array([])

            lines = []

            for y in range(testLines):

                canvasCopy = canvas.copy()

                x1 = int(random() * width)
                y1 = int(random() * height)
                if random() < 0.5:
                    x2 = clippedRandomBetween(
                        x1 - (adjustedCanvasDiff * 2), x1 + (adjustedCanvasDiff * 2), 0, width
                    )
                    y2 = clippedRandomBetween(
                        y1 - (adjustedCanvasDiff * 2), y1 + (adjustedCanvasDiff * 2), 0, height
                    )
                else:
                    x2 = x1
                    y2 = y1
                size = clippedRandomBetween(
                    adjustedCanvasDiff / 2, (adjustedCanvasDiff * 2), 1, 600
                )

                color = baseImg[int((y1 + y2) / 2), int((x1 + x2) / 2)].astype(
                    np.float64
                )

                canvasCopy = cv2.line(canvasCopy, (x1, y1), (x2, y2), color, size)

                diff = findDiff(baseImg, canvasCopy)

                if diff < canvasDiff:

                    canvasCopy = canvas.copy()

                    for z in range(numMutate):
                        x1m = clippedRandomBetween(
                            x1 - adjustedCanvasDiff, x1 + adjustedCanvasDiff, 0, width
                        )
                        y1m = clippedRandomBetween(
                            y1 - adjustedCanvasDiff, y1 + adjustedCanvasDiff, 0, height
                        )
                        x2m = clippedRandomBetween(
                            x2 - adjustedCanvasDiff, x2 + adjustedCanvasDiff, 0, width
                        )
                        y2m = clippedRandomBetween(
                            y2 - adjustedCanvasDiff, y2 + adjustedCanvasDiff, 0, height
                        )
                        sizeM = clippedRandomBetween(size / 2, size * 2, 1, 600)

                        colorM = baseImg[
                            int((y1m + y2m) / 2), int((x1m + x2m) / 2)
                        ].astype(np.float64)

                        canvasCopy = cv2.line(
                            canvasCopy, (x1m, y1m), (x2m, y2m), color, sizeM
                        )

                        diffM = findDiff(baseImg, canvasCopy)

                        if diffM < diff:
                            x1 = x1m
                            y1 = y1m
                            x2 = x2m
                            y2 = y2m
                            size = sizeM
                            color = colorM
                            diff = diffM

                    diffs = np.append(diffs, diff)
                    lines.append([(x1, y1), (x2, y2), color, size])

            if len(diffs) > 0:
                sortedDiffs = np.argsort(diffs)

                x1, y1 = lines[sortedDiffs[0]][0]
                x2, y2 = lines[sortedDiffs[0]][1]
                color = lines[sortedDiffs[0]][2]
                size = lines[sortedDiffs[0]][3]

                canvas = cv2.line(canvas, (x1, y1), (x2, y2), color, size)

                videoData = (
                    videoData
                    + str(x1).zfill(3)
                    + str(y1).zfill(3)
                    + str(x2).zfill(3)
                    + str(y2).zfill(3)
                    + str(int(color[0] * (99 / 255))).zfill(2)
                    + str(int(color[1] * (99 / 255))).zfill(2)
                    + str(int(color[2] * (99 / 255))).zfill(2)
                    + str(size).zfill(3)
                )
            else:
                x1 = randomBetween(0, width)
                y1 = randomBetween(0, height)
                x2 = x1
                y2 = y1
                size = 1
                color = baseImg[y1, x1].astype(np.float64)

        out.write(canvas)

        cv2.imshow("frame", canvas)
        cv2.waitKey(1)

        print(frameI)
        print(100 * frameI / totalConvertedFrames)
        print(canvasDiff)
        print("")

    else:
        print(frameI)
        print(100 * frameI / totalConvertedFrames)
        print("failure")

cap.release()
out.release()

with open("output.txt", "w") as output:
    output.write(videoData)
