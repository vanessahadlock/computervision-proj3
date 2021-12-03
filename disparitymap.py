import numpy as np
import cv2

# use cv2.cornerHarris, threshold by 10%, cv2.cornerSubPix
def cvCorners(img):
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = np.float32(grey)

    # Find corners using harris corner detector
    blockSize = 2 
    ksize = 3
    k = 0.04 
    corner_img = cv2.cornerHarris(grey, blockSize, ksize, k)

    # Dilate and threshold the image by 10%
    thres = corner_img.max() * 0.01
    corner_img = cv2.dilate(corner_img, None)
    ret, corner_img = cv2.threshold(corner_img, thres, 255, 0)

    # use cv2.cornerSubPix to refine the corners
    # This essitially performs NMS
    ret, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(corner_img))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(corner_img, np.float32(centroids),(5,5),(-1,-1),criteria)

    # Create corner image
    res = np.hstack((centroids,corners))
    res = np.int0(res)

    ret = np.zeros_like(grey)
    ret[res[:,3],res[:,2]] = 255

    return ret


# Computes the Normalized Cross Corellation of two 2D arrays
def NCC(f, g):

    f = np.int32(f)
    g = np.int32(g)
    
    fmag = np.sum(f ** 2) ** (1/2)
    gmag = np.sum(g ** 2) ** (1/2)

    return np.sum((f / fmag) * (g / gmag))

# computer ncc for two colored images
def NCCColor(f, g):

    nccBand = []
    for i in range(3):
        nccBand.append(NCC(f[:,:,i],g[:,:,i]))
    return np.average(nccBand)

# Layers the corners on top of the orignial image in red
def addCornertoImage(img, cornerImg, outfilename):

    h, w, _ = img.shape
    for y in range(h):
        for x in range(w):

            if cornerImg[y,x] != 0:
                img[y,x] = [0,0,255]

    cv2.imwrite(outfilename, img)
    
    return


# Returns a list of elements [x1,y1,x2,y2] where (x1,y1) in img1 corresponds to (x2,y2) in img2
def findCorrespondences(img1, img2, corners_img1, corners_img2, wsize, threshold):

    # get the coordinates of the corners in img1

    if wsize % 2 == 0:
        wsize += 1
    offset = wsize // 2

    h1, w1 = corners_img1.shape
    h2, w2 = corners_img2.shape

    corners1 = []
    for y1 in range(offset, h1 - offset):
        for x1 in range(offset, w1 - offset):
            if corners_img1[y1][x1] > 0:
                corners1.append([y1, x1])

    corners2 = []          
    for y2 in range(offset, h2 - offset):
        for x2 in range(offset, w2 - offset):
            if corners_img2[y2][x2] > 0:
                corners2.append([y2, x2])

    # for each corner in img1, find the corner in img2 that is most similar

    ret = []
    for coord1 in corners1:

        # nccList = []
        nccMax = 0
        nccMax_idx = 0
        for i in range(len(corners2)):

            coord2 = corners2[i]
            
            y1, x1 = coord1
            y2, x2 = coord2 

            w1 = img1[y1 - offset : y1 + offset + 1, x1 - offset : x1 + offset + 1, :]
            w2 = img2[y2 - offset : y2 + offset + 1, x2 - offset : x2 + offset + 1, :]

            ncc = NCCColor(w1, w2)
            if(ncc > nccMax):
                nccMax = ncc
                nccMax_idx = i
    
        if nccMax > threshold:
            pt2: np.ndarray = corners2[nccMax_idx]
            ret.append(coord1[::-1] + pt2[::-1])

    return ret


# Draws the corresponding features of two images
def drawMatches(img1, img2, correspondences):

    ###  get the 3x3 transformation homography ###
    dst_pts = []
    src_pts = []
    kp1 = []
    kp2 = []

    for row in correspondences:

            pt1 = [row[0], row[1]]
            pt2 = [row[2], row[3]]

            src_pts.append(pt1)
            dst_pts.append(pt2)

            kp1.append(cv2.KeyPoint(x=pt1[0], y=pt1[1], size=1))
            kp2.append(cv2.KeyPoint(x=pt2[0], y=pt2[1], size=1))
            
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)   

    ### Refine correspondences ###

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    ###  Draw matches between images ###

    goodMatches = []
    kp1_inliers = []
    kp2_inliers = []
    count = 0
    for i in range(len(correspondences)):
        if mask[i] == 1:
            kp1_inliers.append(kp1[i])
            kp2_inliers.append(kp2[i])
            goodMatches.append(cv2.DMatch(_imgIdx=0, _queryIdx=count, _trainIdx=count, _distance=0))
            count += 1



    matches_img = cv2.drawMatches(img1, kp1_inliers, img2, kp2_inliers, goodMatches, None)

    return matches_img


def main():

    cones_left = "cones_left"
    cones_right = "cones_right"
    building_left = "cast_left"
    building_right = "cast_right"

    cones_left: np.ndarray = cv2.imread(f'images/{cones_left}.jpg') 
    cones_right: np.ndarray = cv2.imread(f'images/{cones_right}.jpg')
    building_left: np.ndarray = cv2.imread(f'images/{building_left}.jpg')
    building_right: np.ndarray = cv2.imread(f'images/{building_right}.jpg')


    ##################################################
    ### Detect corner pixels using Harris with NMS ###
    ##################################################

    print('Finding corners for cones_left...')
    corners_cones_left = cvCorners(cones_left)
    
    print('Finding corners for cones_right...')
    corners_cones_right = cvCorners(cones_right)
    
    print('Finding corners for building_left...')
    corners_building_left = cvCorners(building_left)    

    print('Finding corners for building_right...')
    corners_building_right = cvCorners(building_right)

    ##################################################
    ############## Find correspondences ##############
    ##################################################

    wNcc = 7
    thres = 0

    print("Finding correspondences for set 1...")
    correspondences1 = findCorrespondences(cones_left, cones_right, corners_cones_left, corners_cones_right, wNcc, thres)

    print("Finding correspondences for set 2...")
    correspondences2 = findCorrespondences(building_left, building_right, corners_building_left, corners_building_right, wNcc, thres)

    ##################################################
    ################## Draw Matches ##################
    ##################################################

    print("Drawing matches for set 1...")
    matches1 = drawMatches(cones_left, cones_right, correspondences1)
    cv2.imwrite("matchescones.jpg", matches1)

    print("Drawing matches for set 2...")
    matches2 =drawMatches(building_left, building_right, correspondences2)
    cv2.imwrite("matchesbuildings.jpg", matches2)



if __name__ == "__main__":
    main()