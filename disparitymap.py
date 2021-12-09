import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from numpy.core.numeric import empty_like
from scipy.interpolate import interp1d

############### 1 ###############

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

# Draws lines between the corresponding features of two images
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

############### 2 ###############

# find the fundamental matrix for the given correspondences
def FundMatrix(correspondences):

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

# draw epilines on an image
def drawEpiLines(img, lines, pts):

    ret = np.ndarray.copy(img)

    _, h, _ = img.shape

    for e, pt in zip(lines, pts):

        color = tuple(np.random.randint(0,255,3).tolist())
        
        # calculate epipolar line
        x0, y0 = map(int, [0, -e[2] / e[1]]) #x=0, y=-c/b
        x1, y1 = map(int, [h, -(e[2] + e[0]*h) / e[1]]) #x=h, y=-(c+ah)/b

        ret = cv2.line(ret, (x0, y0), (x1, y1), color, 1)
        ret = cv2.circle(ret, tuple(pt), 5, color, -1)

    return ret

def epiLines(img1, img2, pts1, pts2, F):

    # Find lines for image1 using img2 points
    img1_lines = cv2.computeCorrespondEpilines(pts2, 2, F)
    img1_lines = img1_lines.reshape(-1,3)

    # Find lines for image 2 using img1 points
    img2_lines = cv2.computeCorrespondEpilines(pts1, 1, F)
    img2_lines = img2_lines.reshape(-1,3)

    epi_left = drawEpiLines(img1, img1_lines, pts1)
    epi_right = drawEpiLines(img2, img2_lines, pts2)

    return epi_left, epi_right
    return np.concatenate((epi_left, epi_right), axis=1)

############### 3 ###############

# Warp an image given a homography so that it it fully contained in the output img
def perspective_warp(image: np.ndarray, transform: np.ndarray) -> np.ndarray:
    
    h, w = image.shape[:2]
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, transform)
    xmin = math.floor(corners_aft[:, 0, 0].min())
    ymin = math.floor(corners_aft[:, 0, 1].min())
    xmax = math.ceil(corners_aft[:, 0, 0].max())
    ymax = math.ceil(corners_aft[:, 0, 1].max())
    x_adj = math.floor(xmin - corners_aft[0, 0, 0])
    y_adj = math.floor(ymin - corners_aft[0, 0, 1])
    translate = np.eye(3)
    translate[0, 2] = -xmin
    translate[1, 2] = -ymin
    corrected_transform = np.matmul(translate, transform)
    return cv2.warpPerspective(image, corrected_transform, (math.ceil(xmax - xmin), math.ceil(ymax - ymin)))

# Reshape the images so that the epipolar lines are horizontal
def imageRectification(img_left, img_right, pts1, pts2, F):
    
    h1, w1, _ = img_left.shape
    h2, w2, _ = img_right.shape

    # [_, H1, H2] = cv2.stereoRectifyUncalibrated(np.float64(pts1), np.float64(pts2), F, imgSize=(w1, h1))
    [_, H1, H2] = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgSize=(w1, h1))

    rectified_l = perspective_warp(img_left, H1)
    rectified_r = perspective_warp(img_right, H2)
    
    return rectified_l, rectified_r

# return epiilne_img and matches_img given two rectified images
def epilinesRectified(rectified_left, rectified_right):

    # corners
    corners_left = cvCorners(rectified_left)
    corners_right = cvCorners(rectified_right)
    
    # correspondcnes
    print("Finding correspondences...")
    correspondences1 = findCorrespondences(rectified_left, rectified_right, corners_left, corners_right, 7, 0)

    #matches
    matches_img, _, _ = drawMatches(rectified_left, rectified_right, correspondences1)

    # F mat
    F_rect, rect_src, rect_dst = FundMatrix(correspondences1)

    print(f'F_rect: \n\n{F_rect}\n')

    # epilines
    epiline_l, epiline_r = epiLines(rectified_left, rectified_right, rect_src, rect_dst, F_rect)

    return epiline_l, epiline_r, matches_img

# Create disparity images using cv2.StereoBM
def cv_disparity(img_left, img_right, pts1, pts2, F):
    
    h1, w1, _ = img_left.shape
    [_, H1, H2] = cv2.stereoRectifyUncalibrated(np.float64(pts1), np.float64(pts2), F, imgSize=(w1, h1))
    imgL_undistorted = cv2.warpPerspective(img_left, H1, (w1,h1))
    imgR_undistorted = cv2.warpPerspective(img_right, H2, (w1,h1))

    stereo = cv2.StereoBM_create(numDisparities = 32, blockSize = 15)
    disparity_BM = stereo.compute(cv2.cvtColor(imgL_undistorted, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR_undistorted, cv2.COLOR_BGR2GRAY))
    return disparity_BM

# asume correspondences are on the same rows
def disparitymap_rectified1(rectified_left, rectified_right, wNcc, xrange):

    corners_left = cvCorners(rectified_left)
    corners_right = cvCorners(rectified_right)
    correspondences = findCorrespondences(rectified_left, rectified_right, corners_left, corners_right, 7, 0)
    F_rect, rect_src, rect_dst = FundMatrix(correspondences)

    if wNcc % 2 == 0:
        wNcc += 1
    offset = wNcc // 2

    disparity_img = np.zeros_like(rectified_left)
    
    h1, w1, _ = rectified_left.shape
    h2, w2, _ = rectified_right.shape

    # For every pixel in left image:
    for y1 in range(offset, h1 - offset):
        y2 = y1
        print(int(100*y1/h1))

        for x1 in range(offset, w1 - offset):

            # get window of left pixel
            win1 = rectified_left[y1 - offset : y1 + offset + 1, x1 - offset : x1 + offset + 1, :]

            if (y2 > offset) & (y2 < h2):

                bottom = x1 - xrange
                if (bottom < offset):
                    bottom = offset
                top = x1 + xrange
                if (top > (w2 - offset)):
                    top = w2 - offset

                # Iterate row and find the x value of highest corresponding point
                nccMax = 0
                corr_x = 0
                for x2 in range(bottom, top):

                    win2 = rectified_right[y2 - offset : y2 + offset + 1, x2 - offset : x2 + offset + 1, :]

                    ncc = NCCColor(win1, win2)
                    if(ncc > nccMax):
                        nccMax = ncc
                        corr_x = x2

                # Map point disparity to ret image
                if (nccMax > .9):
                    m = interp1d([0,xrange],[0,255])
                    disparity_img[y1, x1] = m(abs(x1 - corr_x))
            
    return disparity_img


### MAIN ###
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

    # print('Finding corners for cones_left...')
    # corners_cones_left = cvCorners(cones_left)
    
    # print('Finding corners for cones_right...')
    # corners_cones_right = cvCorners(cones_right)
    
    # print('Finding corners for building_left...')
    # corners_building_left = cvCorners(castle_left)    

    # print('Finding corners for building_right...')
    # corners_building_right = cvCorners(castle_right)

    # # ##################################################
    # # ##### Find correspondences  / Draw Matches #######
    # # ##################################################

    # wNcc = 7
    # thres = 0
    # d_max = 255

    # print("Finding correspondences for set 1...")
    # correspondences1 = findCorrespondences(cones_left, cones_right, corners_cones_left, corners_cones_right, wNcc, thres)
    # cone_matches, _, _ = drawMatches(cones_left, cones_right, correspondences1)
    # cv2.imwrite("corr/cones.png", cone_matches)

    # print("Finding correspondences for set 2...")
    # correspondences2 = findCorrespondences(castle_left, castle_right, corners_building_left, corners_building_right, wNcc, thres)
    # castle_matches, _, _ = drawMatches(castle_left, castle_right, correspondences2)
    # cv2.imwrite("corr/castle.png", castle_matches)

    # ###################################################
    # #### Find the Fundamental Matrix using RANSAC #####
    # ###################################################

    # F_cones, cones_src_pts, cones_dst_pts = FundMatrix(correspondences1)
    # F_castle, castle_src_pts, castle_dst_pts = FundMatrix(correspondences2)

    # np.save('Fmat/F_cones_double.npy', F_cones)
    # np.save('Fmat/cones_src.npy', cones_src_pts)
    # np.save('Fmat/cones_dst.npy', cones_dst_pts)

    # np.save('Fmat/F_castle_double.npy', F_castle)
    # np.save('Fmat/castle_src.npy', castle_src_pts)
    # np.save('Fmat/castle_dst.npy', castle_dst_pts)
    
    # return

    F_cones = np.load('Fmat/F_cones_double.npy')
    cones_src_pts = np.load('Fmat/cones_src.npy')
    cones_dst_pts = np.load('Fmat/cones_dst.npy')

    F_castle = np.load('Fmat/F_castle_double.npy')
    castle_src_pts = np.load('Fmat/castle_src.npy')
    castle_dst_pts = np.load('Fmat/castle_dst.npy')

    # print(f'F_cones1: \n\n{F_cones}\n')
    # print(f'F_castle1: \n\n{F_castle}\n')

    ###################################################
    ### Draw Epipolar Lines for the inlier features ###
    ###################################################

    # cones_epiline_img_l, cones_epiline_img_r = epiLines(cones_left, cones_right, cones_src_pts, cones_dst_pts, F_cones)
    # cv2.imwrite("epilines/cones_l.jpg", cones_epiline_img_l)
    # cv2.imwrite("epilines/cones_r.jpg", cones_epiline_img_r)
    
    # castle_epiline_img_l, castle_epiline_img_r = epiLines(castle_left, castle_right, castle_src_pts, castle_dst_pts, F_castle)
    # cv2.imwrite("epilines/castle_l.jpg", castle_epiline_img_l)
    # cv2.imwrite("epilines/castle_r.jpg", castle_epiline_img_r)

    ###################################################
    ############# Rectify Images using F ##############
    ###################################################


    cones_left_rect, cones_right_rect = imageRectification(cones_left, cones_right, cones_src_pts, cones_dst_pts, F_cones)
    
    # epiline_img_l, epiline_img_r, matches_img = epilinesRectified(cones_left_rect, cones_right_rect)
    # cv2.imwrite("rectified/cones_epiline_l.png", epiline_img_l)
    # cv2.imwrite("rectified/cones_epiline_r.png", epiline_img_r)

    castle_left_rect, castle_right_rect = imageRectification(castle_left, castle_right, castle_src_pts, castle_dst_pts, F_castle)
    
    # epiline_img_l, epiline_img_r, matches_img = epilinesRectified(castle_left_rect, castle_right_rect)
    # cv2.imwrite("rectified/castle_epiline_l.png", epiline_img_l)
    # cv2.imwrite("rectified/castle_epiline_r.png", epiline_img_r)

    ###################################################
    ########### Compute Dense Disparity Map ###########
    ###################################################

    s = 0.5
    wNcc = 7
    xrange = 50

    h, w, _ = castle_right.shape
    rectifiedImg1 = cv2.resize(castle_left_rect, (int(h*s), int(w*s)))
    rectifiedImg2 = cv2.resize(castle_right_rect, (int(h*s), int(w*s)))

    disparityimg = disparitymap_rectified1(rectifiedImg1, rectifiedImg2, wNcc, xrange)
    cv2.imwrite("disparity/castle_70_res_wNcc7_xrange50.png", disparityimg)

    return

    rectifiedImg1, rectifiedImg2 = imageRectification(castle_left, castle_right, building_src_pts, building_dst_pts, F_building)
    cv2.imwrite("castle_rectified_left.png", rectifiedImg1)
    cv2.imwrite("castle_rectified_right.png", rectifiedImg2)

    disparityimg = disparitymap(castle_left, castle_right, building_src_pts, building_dst_pts, F_building, d_max)
    # cv2.imwrite("disparuty_img2.png", disparityimg)

    # Rectify the two images so their epipolar lines will all be parallel
    # In the left image, at the first pixel use the F matrix to calculate an epipolar line
    # at that point to the other image. Then search along the epipolar line for the highest
    # corresponding point using NCCcolor() and then calculate the disparity to that point
    # in order to get the disparity at the first pixel. Iterate through each pixel in the 
    # left image and repeat this process till you get a disparity value for each pixel 


if __name__ == "__main__":
    main()