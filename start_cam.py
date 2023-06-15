'''
[Last updated: 13-06-2023]
This program will be running on Raspberry Pi Device.
The program monitors for eye blink condition by locating the face landmarks -
and measure the Eye Aspect Ratio (EAR). 
The EAR below the specified threshold for specified period of time will be considered as "eyes closed' condition.
The 'eyes closed' condition will drive a specified GPIO pin that can be connected to external device such as relay.
'''
import argparse
import time
from threading import Condition, Thread
import cv2 as cv
import dlib
import numpy as np
from scipy.spatial import distance as dist

from relay import Relay


# Relay Pin
relay = None
# Blink and unblink condition variables
blink_event_condition = None
unblink_event_condition = None
# Flag for blink
is_blinked = False
# Flag for presence of face
is_face = False


# Waiting for the blink
def wait_for_blink(timeout = 3):

    global blink_event_condition
    global unblink_event_condition
    global is_blinked
    global is_face
    global relay
    
    while True:

        # Delay to let the relay on after blink timeout. 
        # Also meant to prevent intense loop when there is no face
        time.sleep(1.5)

        if not is_blinked:

            # Wait for the blink event
            with blink_event_condition:
                print ('Wait for blink...')
                blink_event_condition.wait()
                # Restart the loop if there is no face detected
                if not is_face:
                    # Turn off the relay
                    relay.off()
                    continue
                print ('Blinked...')

        # Wait for unblink event
        with unblink_event_condition:
            print ('Waiting for unblink...')
            if not (unblink_event_condition.wait(timeout=timeout)):
                print ('Unblink timeout')
                # DRIVE BLINK RELAY HERE
                relay.on()
            else:
                # Restart the loop if there is no face detected
                if not is_face:
                    # Turn off the relay
                    relay.off()
                    continue
                print ('blink refreshed')
                # Turn off the relay
                relay.off()


def start_camera(blink_threshold = 0.25, 
                 unblink_threshold = 0.28, 
                 flip = True, 
                 res=(640,480),
                 predictor_path = '.'
                 ):

    global relay
    global blink_event_condition
    global unblink_event_condition
    global is_blinked
    global is_face

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

    # define a video capture object
    vid = cv.VideoCapture(0)
    # Setting resolution
    #vid.set(3, 640)
    #vid.set(3, 480)
    # dlib face detector
    face_detector = dlib.get_frontal_face_detector()
    # dlib face landmark predictor
    face_predictor = dlib.shape_predictor(predictor_path)
    # blink and unblink condition variables
    # Right eye landmark index
    r_index = [*range(36, 42)]
    # left eye landmark index
    l_index = [*range(42, 48)]
    # Make sure relay is off before starting
    relay.off()

    while(True):

        try:
            #t1 = time.time()
            # Read the frame
            ret, frame = vid.read()
            # Frame conversion to gray
            img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Flip
            if flip:
                img_gray = cv.rotate(img_gray, cv.ROTATE_180)
            # Face detection
            rects = face_detector(img_gray, 0)

            if len(rects) == 0:
                # Face is detected
                print ('no face')
                # No face is detected
                is_face = False
                # Reset the blink flag
                is_blinked = False
                # Clear the waiting thread
                with blink_event_condition:
                    blink_event_condition.notify_all()
                with unblink_event_condition:
                    unblink_event_condition.notify_all()
                continue

            else:
                # Face is detected
                print ('face detected')
                is_face = True
                for rect in rects:
                    shape = face_predictor(img_gray, rect)
                    shape = shape_to_np(shape)
                    eye_l = shape[l_index]
                    eye_r = shape[r_index]
                    ear_r = calculate_ear(eye_r)
                    ear_l = calculate_ear(eye_l)

                    #ear_max = max(ear_r, ear_l)
                    ear_max = (ear_r + ear_l)/2
                    print (f'ear: {ear_max}')

                    # Check the blink
                    if ear_max < blink_threshold:
                        print ('blink occur')
                        # Blink occur
                        is_blinked = True
                        with blink_event_condition:
                            blink_event_condition.notify_all()

                    elif ear_max >= unblink_threshold:
                        # Unblink occur
                        is_blinked = False
                        with unblink_event_condition:
                            unblink_event_condition.notify_all()
        
            #t2 = time.time()
            #print (f'frame_time: {t2-t1}')

        except Exception as e:
            print (e)
            # On error, release the cap object
            vid.release()
            break


def main(blink_thres = 0.25, 
         unblink_thres = 0.28, 
         blink_timeout = 3.0, 
         flip=True, 
         relay_pin=17, 
         predictor_path = '.',
         res = (640, 480)):

    global relay 
    global blink_event_condition 
    global unblink_event_condition
    
    print ('Starting...')

    # Default Relay Pin on GPIO 17 (Raspberry Header Pin 11)
    relay = Relay(pin=relay_pin)
    # Blink and unblink condition variables
    blink_event_condition = Condition()
    unblink_event_condition = Condition()
    # Start camera
    start_camera_t = Thread(target = start_camera, args=(blink_thres,
                                                        unblink_thres, flip, res, predictor_path,))
    # Wait for blink
    wait_for_blink_t = Thread(target = wait_for_blink, args=(blink_timeout,))
    start_camera_t.daemon = True
    start_camera_t.start()
    wait_for_blink_t.daemon = True
    wait_for_blink_t.start()
    print ('Running...')

    while True:
        time.sleep(3)
        # Check the thread status and break if one of it no longer active
        if not (start_camera_t.is_alive() and wait_for_blink_t.is_alive()):
            # DRIVE ERROR SIGNAL HERE
            break
    
    # Make sure relay is off before quit
    relay.off()
    print ('end')


if __name__ == '__main__':

    # Argument handler
    parser = argparse.ArgumentParser()
    parser.add_argument('--blink_thres', type = float, default = 0.25, required = False)
    parser.add_argument('--unblink_thres', type = float, default = 0.28, required = False)
    parser.add_argument('--blink_timeout', type = float, default = 3.0, required = False)
    parser.add_argument('--flip', type = bool, default = True, required = False)
    parser.add_argument('--relay_pin', type = int, default = 17, required = False)
    parser.add_argument('--pred_path', type = str, default = '../shape_predictor_68_face_landmarks.dat', required = False)

    # Parsing
    args = parser.parse_args()
    blink_thres = args.blink_thres
    unblink_thres = args.blink_thres
    blink_timeout = args.blink_timeout
    flip = args.flip
    relay_pin = args.relay_pin
    predictor_path = args.pred_path

    # Run
    main(blink_thres, unblink_thres, blink_timeout, flip, relay_pin, predictor_path)
