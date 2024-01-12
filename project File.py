import cv2
import imutils
import numpy as np
import argparse

# Function to detect people in a frame using HOG descriptor
def detect(frame):
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1)
        person += 1
    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('output', frame)

    return frame

# Function to detect people in a video file
def detectByPathVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check is False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while video.isOpened():
        check, frame = video.read()
        if check:
            frame = imutils.resize(frame, width=min(1000, frame.shape[1]))
            frame = detect(frame)
            capname = "cap"
            if writer is not None:
                writer.write(frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

# Function to detect people using the camera
def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame = video.read()

        frame = detect(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Function to detect people in an image file
def detectByPathImage(path, output_path):
    image = cv2.imread(path)

    image = imutils.resize(image, width=min(800, image.shape[1])) 

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to handle command-line arguments and call relevant functions
def humanDetector(args):
    image_path = args["image"]
    video_path = args['video']
    
    # Check if camera flag is set to 'true'
    if str(args["camera"]).lower() == 'true':
        camera = True 
    else:
        camera = False

    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MP4'), 10, (600, 600))

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])

# Function to parse command-line arguments
def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="Path to the video file")
    arg_parse.add_argument("-i", "--image", default=None, help="Path to the image file")
    arg_parse.add_argument("-c", "--camera", default=False, help="Flag indicating to use the camera (true/false)")
    arg_parse.add_argument("-o", "--output", type=str, help="Path for output video or directory for output image")
    args = vars(arg_parse.parse_args())

    return args

# Entry point of the script
if __name__ == "__main__":
    # Initialize HOG descriptor for people detection
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Parse command-line arguments
    args = argsParser()
    
    # Call the main function to perform human detection
    humanDetector(args)
