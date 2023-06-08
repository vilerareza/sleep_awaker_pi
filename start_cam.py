import asyncio
import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist
import dlib
import time
from threading import Condition
#from asyncio import Condition

# from skimage.color import rgb2gray

# Face predictor path
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# define a video capture object
vid = cv.VideoCapture(0)

# dlib face detector
face_detector = dlib.get_frontal_face_detector()
# dlib face landmark predictor
face_predictor = dlib.shape_predictor(predictor_path)

# Right eye landmark index
r_index = [*range(36, 42)]
# left eye landmark index
l_index = [*range(42, 48)]
# Target size
img_size = (640, 480)

blink_condition = Condition()
is_blinked = False


# async def time_the_blink(timeout = 3):

#     global blink_condition
#     global is_blinked
    
#     with blink_condition:

#         print ('wait the blink')
#         if not (blink_condition.wait(timeout = timeout)):
#             print ('Blink timeout')
#             # Do something here
#         else:
#             print ('blink refreshed')
        
#         is_blinked = False


async def wait_for_blink(timeout = 3):

    print ('task2')
    # global blink_condition
    # global is_blinked
    
    # with blink_condition:

    #     print ('wait the blink')
    #     if not (blink_condition.wait(timeout = timeout)):
    #         print ('Blink timeout')
    #         # Do something here
    #     else:
    #         print ('blink refreshed')


async def start_camera(blink_threshold = 0.2, blink_timeout = 3):

    global blink_condition
    global is_blinked

    async def shape_to_np(shape):

        coords = np.zeros((shape.num_parts, 2), dtype='int')

        for i in range (shape.num_parts):
            coords [i] = (shape.part(i).x, shape.part(i).y)
        
        return coords


    async def calculate_ear(eye):

        # Distance of vertical landmarks 1
        a = dist.euclidean(eye[1], eye[5])
        # Distance of vertical landmarks 2
        b = dist.euclidean(eye[2], eye[4])
        # Distance of horizontal landmarks
        c = dist.euclidean(eye[0], eye[3])

        # Eye aspect ratio
        return ((a+b) / (2*c))


    while(True):

        t1 = time.time()
        ret, frame = vid.read()
        # print (frame.shape)
        # Frame conversion to gray
        # Resize the frame
        #h, w = frame.shape[:2]
        # factor = img_size[0]/w
        # frame = cv.resize(frame, (int(w*factor), int(h*factor)))
        img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #img_gray = (rgb2gray(frame[:,:,::-1])*255).astype('uint8')
        # Need to 
        # Face detection
        rects = face_detector(img_gray, 0)
        # print (rect)
        # Display the resulting frame
        # cv.imshow('frame', frame)

        for rect in rects:
            shape = face_predictor(img_gray, rect)
            shape = await shape_to_np(shape)
            eye_l = shape[l_index]
            eye_r = shape[r_index]
            ear_r = await calculate_ear(eye_r)
            ear_l = await calculate_ear(eye_l)
            # ear_avg = (ear_r + ear_l) / 2
            ear_max = max(ear_r, ear_l)
            # print (f'ear: {ear_max}')

            # Check the blink
            if ear_max < blink_threshold:
                #print (f'ear: {ear_max}')
                #print ('blink occured')
                #if not is_blinked:
                print ('blinked')
                    # is_blinked=True
                    # time_the_blink(timeout=blink_timeout)
            else:
                pass
                #with blink_condition:
                #   blink_condition.notify_all()

            leftEyeHull = cv.convexHull(eye_l)
            rightEyeHull = cv.convexHull(eye_r)
            cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Display the resulting frame
        cv.imshow('frame', frame)

        # the 'q' button is set as the
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
        t2 = time.time()
        # print (f'frame_time: {t2-t1}')

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()

async def main():
    global blink_condition
    # Start camera
    task_start_camera = asyncio.create_task(start_camera())
    #print ('task1')
    task_wait_for_blink = asyncio.create_task(wait_for_blink())
    #print ('task2')
    #await task_start_camera
    await task_wait_for_blink
    with blink_condition:
        blink_condition.notify_all()

    print ('end')

asyncio.run (main())