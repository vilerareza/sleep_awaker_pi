import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist
import dlib
import time
from threading import Thread, Condition
from relay import Relay


# Face predictor path
predictor_path = '../shape_predictor_68_face_landmarks.dat'

# define a video capture object
vid = cv.VideoCapture(0)

# Relay Pin on GPIO 17 (Raspberry Header Pin 11)
relay = Relay(pin=17)

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

blink_event_condition = Condition()
unblink_event_condition = Condition()
is_blinked = False


def wait_for_blink(timeout = 3):

    global vid
    global blink_event_condition
    global unblink_event_condition
    global is_blinked
    global relay
    
    while True:

        with blink_event_condition:
            # Wait for the blink event
            # print ('Wait for blink...')
            blink_event_condition.wait()

        with unblink_event_condition:
            print ('Blink. Waiting for unblink...')
            if not (unblink_event_condition.wait(timeout=timeout)):
                # Wait for the blink event
                print ('Unblink timeout')
                # DRIVE BLINK SIGNAL HERE
                relay.on()
            else:
                print ('blink refreshed')
                # DRIVE BLINK SIGNAL HERE
                relay.off()


def start_camera(blink_threshold = 0.25, unblink_threshold = 0.28):

    global blink_event_condition
    global unblink_event_condition
    global is_blinked

    def shape_to_np(shape):

        coords = np.zeros((shape.num_parts, 2), dtype='int')

        for i in range (shape.num_parts):
            coords [i] = (shape.part(i).x, shape.part(i).y)
        
        return coords


    def calculate_ear(eye):

        # Distance of vertical landmarks 1
        a = dist.euclidean(eye[1], eye[5])
        # Distance of vertical landmarks 2
        b = dist.euclidean(eye[2], eye[4])
        # Distance of horizontal landmarks
        c = dist.euclidean(eye[0], eye[3])

        # Eye aspect ratio
        return ((a+b) / (2*c))


    while(True):

        try:
            t1 = time.time()
            ret, frame = vid.read()
            # Frame conversion to gray
            # Resize the frame
            #h, w = frame.shape[:2]
            # factor = img_size[0]/w
            # frame = cv.resize(frame, (int(w*factor), int(h*factor)))
            img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            img_gray = cv.rotate(img_gray, cv.ROTATE_180)
            # Face detection
            rects = face_detector(img_gray, 0)

            for rect in rects:
                shape = face_predictor(img_gray, rect)
                shape = shape_to_np(shape)
                eye_l = shape[l_index]
                eye_r = shape[r_index]
                ear_r = calculate_ear(eye_r)
                ear_l = calculate_ear(eye_l)

                ear_max = max(ear_r, ear_l)
                print (f'ear: {ear_max}')

                # Check the blink
                if ear_max < blink_threshold:
                    #print (f'ear: {ear_max}')
                    #print ('blink')
                    with blink_event_condition:
                        blink_event_condition.notify_all()

                elif ear_max >= unblink_threshold:
                    with unblink_event_condition:
                        unblink_event_condition.notify_all()

                leftEyeHull = cv.convexHull(eye_l)
                rightEyeHull = cv.convexHull(eye_r)
                cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
            t2 = time.time()
            print (f'frame_time: {t2-t1}')

        except:
            # On error, release the cap object
            vid.release()
            break


def stop_thread(stop_event, thread):
    print ('Stopping thread')
    stop_event.set()
    thread.join()


def main():

    print ('Starting...')
    # Start camera
    start_camera_t = Thread(target = start_camera, args=(0.2,))
    # Wait for blink
    wait_for_blink_t = Thread(target = wait_for_blink, args=(3,))
    start_camera_t.daemon = True
    start_camera_t.start()
    wait_for_blink_t.daemon = True
    wait_for_blink_t.start()

    while True:
        time.sleep(3)
        # Check the thread status and break if one of it no longer active
        if not (start_camera_t.is_alive() and wait_for_blink_t.is_alive()):
            # DRIVE ERROR SIGNAL HERE
            break

    print ('end')

if __name__ == '__main__':
    main()