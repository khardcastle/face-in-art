#!/usr/bin/env python
# coding: utf-8

## Code in this file was either taken directly, or adapted from, https://github.com/spmallick/learnopencv


from imutils import face_utils
import imutils
import dlib
import cv2
import glob
import numpy as np
import math


# In[ ]:


# crop the image so it's just the face
def compute_face_crop(image):
    
    # This will take an image file, and will return information about where the face is, the image with the face bounding box, and the cropped image
    
    # read image, copy it so nothing funky happens, convert to grayscale for opencv
    image11 = np.copy(image)
    imgtest = cv2.cvtColor(image11, cv2.COLOR_BGR2GRAY)

    # load the face classifier
    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # run the face classifier for the image. returns bounding box of the face
    faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.2, minNeighbors=5)
    # plot the image and the bounding box of the face
    if len(faces) > 2:
        display('too many faces')
        faces = []
        face_detect = []
        cropped_image = image
        return
    else:
        (x, y, w, h) = faces[0]
        pad = 25;
        image_size = np.shape(imgtest)
        # detect the face region
        minx = max(0,x-pad)
        miny= max(0,y-pad)
        maxx = min(image_size[1],x+w+pad)
        maxy = min(image_size[0],y+w+pad) 
        rectangle = (minx, miny, maxx, maxy)
        face_detect = cv2.rectangle(image11, (minx, miny), (maxx, maxy), (0, 0, 0), 0)

        # save the cropped image
        cropped_image = image[miny:maxy, minx:maxx,:]
        
        return faces, face_detect, cropped_image, rectangle


# In[ ]:

def compute_landmarks(image, detector, predictor):
    lm_image = np.copy(image)
    lm_image = cv2.cvtColor(lm_image, cv2.COLOR_BGR2GRAY)  
    detected_image = detector(lm_image, 0)
    points1 = predictor(lm_image, detected_image[0])
    points1 = face_utils.shape_to_np(points1)
        
    points = []
    for x,y in points1:
        points.append((int(x), int(y)))
    
    return points, points1

# In[ ]:

def draw_landmarks(image, points):
    lm_image1 = np.copy(image)
    #lm_image = cv2.cvtColor(lm_image, cv2.COLOR_BGR2GRAY)   

    for (x, y) in points:
    	x1 = int(x)
    	y1 = int(y)
    	cv2.circle(lm_image1, (x1, y1), 5, (0, 0, 0), -1)

    return lm_image1

# In[ ]:


def find_center(image, points):

	(x, y, w, h) = cv2.boundingRect(np.array(points))
	center_x = x+w/2
	center_y = y+h/2
	center = center_x, center_y

	return center

# In[ ]:

def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)  
  
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
    
    inPts.append([np.int(xin), np.int(yin)])

    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
    
    outPts.append([np.int(xout), np.int(yout)])
    
    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    
    return tform[0]

 # In[ ]:

def align_image(image, source_points, dest_points, w, h):

	M1 = similarityTransform(source_points, dest_points)
	translated_image = cv2.warpAffine(image, M1, (w,h))

	return translated_image

# In[ ]:

# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


# In[ ]:


def calculateDelaunayTriangles(image, t_points):
    
    delaunay_color = (255,255,255)
    
    image_triangle = np.copy(image)
    sizeImg = image_triangle.shape    
    rect = (0, 0, sizeImg[1], sizeImg[0])
    
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv

    for p in t_points:
    	subdiv.insert(p)
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
        
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            
            cv2.line(image_triangle, pt1, pt2, delaunay_color, 2, cv2.LINE_AA, 0)
            cv2.line(image_triangle, pt2, pt3, delaunay_color, 2, cv2.LINE_AA, 0)
            cv2.line(image_triangle, pt3, pt1, delaunay_color, 2, cv2.LINE_AA, 0)
            
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(t_points)):                    
                    if(abs(pt[j][0] - t_points[k][0]) < 1.0 and abs(pt[j][1] - t_points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []       
            
    return delaunayTri, image_triangle


# In[ ]:


def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# In[ ]:


def warpTriangle(img1, img2, t1, t2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 


# In[ ]:


def computeHullIndex(points1, points):
    hullIndex1 = cv2.convexHull(points1, returnPoints = False)
    hullPoints = []
    for i in range(0, len(hullIndex1)):
        hullPoints.append(points[int(hullIndex1[i])])
        
    return hullPoints, hullIndex1


# In[ ]:


def compute_warped_image(image1,image2,dt_image2,points1,points2):
    image1_warped = np.copy(image2)

    for i in range(0, len(dt_image2)):
        t1 = []
        t2 = []
        
    #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(points1[dt_image2[i][j]])
            t2.append(points2[dt_image2[i][j]])
        
        warpTriangle(image1, image1_warped, t1, t2)
        
    return image1_warped


# In[ ]:


def clone_image(hullPoints2,image1,image2):

    hull8U = []
    for i in range(0, len(hullPoints2)):
        hull8U.append((hullPoints2[i][0], hullPoints2[i][1]))

    mask = np.zeros(image2.shape, dtype = image2.dtype) 

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hullPoints2]))    

    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(image1), image2, mask, center, cv2.NORMAL_CLONE)
    
    return output, mask

