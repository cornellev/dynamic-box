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
    # cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=5)

    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_FGD
    trimap[mapFg == 255] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(block, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')

    contour2 = findLargestContour(mask2)
    cv2.drawContours(contourImg, [contour2], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    mask = np.zeros_like(mask2)
    cv2.fillPoly(mask, [contour2], 255)

    mask = mask[:, ::10][::10]
    new_mask = np.hstack((coordinates, mask.T.reshape(-1,1))).astype(float)

    # blended alpha cut-out
    # mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    # mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
    # alpha = mask4.astype(float) * 1.1  # making blend stronger
    # alpha[mask3 > 0] = 255.0
    # alpha[alpha > 255] = 255.0

    # foreground = np.copy(block).astype(float)
    # foreground[mask4 == 0] = 0
    # background = np.ones_like(foreground, dtype=float) * 255.0

    # # Normalize the alpha mask to keep intensity between 0 and 1
    # alpha = alpha / 255.0
    # # Multiply the foreground with the alpha matte
    # foreground = cv2.multiply(alpha, foreground)
    # # Multiply the background with ( 1 - alpha )
    # background = cv2.multiply(1.0 - alpha, background)
    # # Add the masked foreground and background.
    # cutout = cv2.add(foreground, background)

    cv2.imwrite('contour.png', contourImg)
    return {tuple(coord[:-1]) : coord[-1] for coord in new_mask}

# cv2.imwrite("lilphone.png", getEdgeMask(cv2.imread("left0.png"), [746, 398, 153, 163]))

def findContours():

    image = cv2.imread('chair.png')  # Replace with your image path

    # Step 2: Convert to grayscale (if it's not already)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply a binary threshold to the image
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Alternative: You can use Canny edge detection instead of thresholding
    # binary_image = cv2.Canny(gray, 100, 200)

    # Step 4: Find contours
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Draw contours on the original image
    # You can draw all contours or filter based on some criteria (e.g., contour area)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Green color, thickness 2

    # Step 6: Display the result
    cv2.imwrite('find.png', image_with_contours)

findContours()