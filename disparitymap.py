import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    return matches_img, kp1_inliers, kp2_inliers

# find the fundamental matrix for the given correspondences
def FundMatrix(correspondences):

    ###  get the 3x3 transformation homography ###
    dst_pts = []
    src_pts = []

    for row in correspondences:

            pt1 = [row[0], row[1]]
            pt2 = [row[2], row[3]]

            src_pts.append(pt1)
            dst_pts.append(pt2)
    
    dst_pts = np.int32(dst_pts)
    src_pts = np.int32(src_pts)

    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

    # We select only inlier points
    src_pts = src_pts[mask.ravel()==1]
    dst_pts = dst_pts[mask.ravel()==1]

    F, _ = cv2.findFundamentalMat(src_pts, dst_pts)

    return F, src_pts, dst_pts


def imageRectification(img_left, img_right, pts1, pts2, F):
    
    h1, w1, _ = img_left.shape
    h2, w2, _ = img_right.shape

    [H1, H2, success] = cv2.stereoRectifyUncalibrated(np.float64(pts1), np.float64(pts2), F, imgSize=(w1, h1))

    print(img_left.shape)
    print(h1, w1)

    ############## Undistort (Rectify) ##############
    imgL_undistorted = cv2.warpPerspective(img_left, H1, (375, 450))
    imgR_undistorted = cv2.warpPerspective(img_right, H2, (w2, h2))
    cv2.imwrite("undistorted_Left.png", imgL_undistorted)
    cv2.imwrite("undistorted_Right.png", imgR_undistorted)

    ############## Calculate Disparity (Depth Map) ##############

    # Using StereoBM
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity_BM = stereo.compute(imgL_undistorted, imgR_undistorted)
    plt.imshow(disparity_BM, "gray")
    plt.colorbar()
    plt.show()

    return disparity_BM


def findEpipoleLines(img1, img2, pts1, pts2, F):

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    
    lines1 = cv2.computeCorrespondEpilines(pts1, 2, F)

    lines1 = lines1.reshape(-1,3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts2, 1, F)
    lines2 = lines2.reshape(-1,3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

    lines = np.concatenate((img3, img5), axis=1)

    return lines
    
   
def drawlines(img1, img2, lines, pts1, pts2):
   # img1 - image on which we draw the epilines for the points in img2 lines
   # corresponding epilines

    w, h, _ = img1.shape

    for w, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -w[2]/w[1] ])
        x1, y1 = map(int, [h, -(w[2]+w[0]*h)/w[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    
    return img1, img2


def disparitymap(img_left, img_right, pts1, pts2, F, d_max):

    # makes corresponding points from (x1, y1) and (x2, y2) into (x1, x2, y1, y2) in each row
    points = np.dstack((pts1, pts2))

    print(pts1[0])
    print(pts2[0])

    print(points[0])
    
    disparity_x = []
    disparity_y = []

    w, h, _ = img_left.shape
    
    disparity_map = np.zeros((w, h, 1), dtype=np.uint8)

    for row in points:

            x1 = row[0][0]
            x2 = row[0][1]

            y1 = row[1][0]
            y2 = row[1][1]

            disparityX = abs(x2 - x1)
            disparityY = abs(y2 - y1)

            disparity_x.append(disparityX)
            disparity_y.append(disparityY)

            # disparity_map[x] = disparityX * (255. / d_max)

    return

def main():

    img1 = "cones_left"
    img2 = "cones_right"
    img3 = "cast_left"
    img4 = "cast_right"

    cones_left: np.ndarray = cv2.imread(f'images/{img1}.jpg') 
    cones_right: np.ndarray = cv2.imread(f'images/{img2}.jpg')
    castle_left: np.ndarray = cv2.imread(f'images/{img3}.jpg')
    castle_right: np.ndarray = cv2.imread(f'images/{img4}.jpg')


    ##################################################
    ### Detect corner pixels using Harris with NMS ###
    ##################################################

    print('Finding corners for cones_left...')
    corners_cones_left = cvCorners(cones_left)
    
    print('Finding corners for cones_right...')
    corners_cones_right = cvCorners(cones_right)
    
    print('Finding corners for building_left...')
    corners_building_left = cvCorners(castle_left)    

    print('Finding corners for building_right...')
    corners_building_right = cvCorners(castle_right)

    ##################################################
    ############## Find correspondences ##############
    ##################################################

    wNcc = 7
    thres = 0
    d_max = 255

    print("Finding correspondences for set 1...")
    correspondences1 = findCorrespondences(cones_left, cones_right, corners_cones_left, corners_cones_right, wNcc, thres)

    print("Finding correspondences for set 2...")
    correspondences2 = findCorrespondences(castle_left, castle_right, corners_building_left, corners_building_right, wNcc, thres)

    ##################################################
    ################## Draw Matches ##################
    ##################################################

    print("Drawing matches for set 1...")
    cone_matches, cone_kp1, cone_kp2 = drawMatches(cones_left, cones_right, correspondences1)
    cv2.imwrite("matchescones.jpg", cone_matches)

    print("Drawing matches for set 2...")
    building_matches, building_kp1, building_kp2 = drawMatches(castle_left, castle_right, correspondences2)
    cv2.imwrite("matchesbuildings.jpg", building_matches)

    ###################################################
    #### Find the Fundamental Matrix using RANSAC #####
    ###################################################

    F_cones, cones_src_pts, cones_dst_pts = FundMatrix(correspondences1)
    F_building, building_src_pts, building_dst_pts = FundMatrix(correspondences2)


    ###################################################
    ### Draw Epipolar Lines for the inlier features ###
    ###################################################

    cones = findEpipoleLines(cones_left, cones_right, cones_src_pts, cones_dst_pts, F_cones)
    cv2.imwrite("linescones.jpg", cones)
    
    building = findEpipoleLines(castle_left, castle_right, building_src_pts, building_dst_pts, F_building)
    cv2.imwrite("linesbuildings.jpg", building)

    disparity_map = imageRectification(cones_left, cones_right, cones_src_pts, cones_dst_pts, F_cones)
    cv2.imwrite("disparitymap.jpg", disparity_map)

    # disparitymap(cones_left, cones_right, cones_src_pts, cones_dst_pts, F_cones, d_max)



    ##### To Calculate The Disparity Map #####
    # Rectify the two images so their epipolar lines will all be parallel
    # In the left image, at the first pixel use the F matrix to calculate an epipolar line
    # at that point to the other image. Then search along the epipolar line for the highest
    # corresponding point using NCCcolor() and then calculate the disparity to that point
    # in order to get the disparity at the first pixel. Iterate through each pixel in the 
    # left image and repeat this process till you get a disparity value for each pixel 



if __name__ == "__main__":
    main()