import cv2

def read(path, video):
    '''
        @get  path, video as string

        @return void
    '''

    realpath = path + video
    i = 0
    myvideo = cv2.VideoCapture(realpath)
    while(myvideo.isOpened()):
        ret, frame = myvideo.read()
        i+=1
        cv2.imshow('frame',frame)
        if (cv2.waitKey(0) == ord('c')):
            cv2.imwrite('data/'+str(i)+'.png',frame)

        if (cv2.waitKey(0) == ord('q')) :
            break

    myvideo.release()
    cv2.destroyAllWindows()

path = './mua/khmer/'
video = 'Khmer-Chol-Chnam-Thmay.webm'

read (path, video)
