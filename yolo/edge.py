# pip3 install opencv-contrib==4.5.1.48 opencv-contrib-python==4.5.1.48
import numpy as np
import cv2

def getEdgeMask (img, box):
    block = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]
    y_coords, x_coords = np.meshgrid(np.arange(box[1],box[1]+box[3],10), np.arange(box[0],box[0]+box[2],10))
    coordinates = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)

    blurred = cv2.GaussianBlur(block, (5, 5), 0)
    blurred_float = blurred.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")
    edges = edgeDetector.detectEdges(blurred_float) * 255.0

    def filterOutSaltPepperNoise(edgeImg):
        # Get rid of salt & pepper noise.
        count = 0
        lastMedian = edgeImg
        median = cv2.medianBlur(edgeImg, 3)
        while not np.array_equal(lastMedian, median):
            # get those pixels that gets zeroed out
            zeroed = np.invert(np.logical_and(median, edgeImg))
            edgeImg[zeroed] = 0

            count = count + 1
            if count > 50:
                break
            lastMedian = median
            median = cv2.medianBlur(edgeImg, 3)

    edges_8u = np.asarray(edges, np.uint8)
    filterOutSaltPepperNoise(edges_8u)

    def findLargestContour(edgeImg):
        contours, hierarchy = cv2.findContours(
            edgeImg,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contoursWithArea = []
        for contour in contours:
            area = cv2.contourArea(contour)
            contoursWithArea.append([contour, area])
            
        contoursWithArea.sort(key=lambda tupl: tupl[1], reverse=True)
        largestContour = contoursWithArea[0][0]
        return largestContour

    contour = findLargestContour(edges_8u)
    # Draw the contour on the original image
    contourImg = np.copy(block)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    cv2.imwrite("contour.png", contourImg)
    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    mask = mask[:, ::10][::10]
    new_mask = np.hstack((coordinates, mask.T.reshape(-1,1))).astype(float)
    return {tuple(coord[:-1]) : coord[-1] for coord in new_mask}

# cv2.imwrite("lilphone.png", getEdgeMask(cv2.imread("left0.png"), [746, 398, 153, 163]))