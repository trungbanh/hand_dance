import cv2 
import numpy as np

# body mpi 
protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

nPoints = 15
video = cv2.VideoCapture('Khmer-Chol-Chnam-Thmay.webm')

print ("okey chạy nè")

ten = 0

while True:
    ten = ten +1
    ret, frame = video.read()
    if (ten > 130 and ten % 3 == 0):
        

        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth/frameHeight

        inHeight = 386
        inWidth = int(((aspect_ratio*inHeight)*8)//8)

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        points = []
        frameCopy = np.zeros(frame.shape)
        threshold = 0.2

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
            if prob > threshold :
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else :
                points.append(None)

        # # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frameCopy, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(frameCopy, points[partA], 1, (0, 0, 255), thickness=1, lineType=cv2.FILLED)

        cv2.imwrite('data/'+str(ten)+'.jpg',frameCopy)
        

print ("xong het roi")
