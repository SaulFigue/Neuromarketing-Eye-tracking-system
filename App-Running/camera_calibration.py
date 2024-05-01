import glob
from datetime import datetime
from argparse import ArgumentParser

import cv2
import numpy as np
import yaml

from webcam import WebcamSource
   
def record_video(width: int, height: int, fps: int) -> None:
    """
    Create a mp4 video file with `width`x`height` and `fps` frames per second.
    Shows a preview of the recording every 5 frames.

    :param width: width of the video
    :param height: height of the video
    :param fps: frames per second
    :return: None
    """

    source = WebcamSource(width=width, height=height, fps=fps, buffer_size=10)
    video_writer = cv2.VideoWriter(f'{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # video_writer = cv2.VideoWriter('date1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # start = time.time()

    for idx, frame in enumerate(source):
        video_writer.write(frame)        
        source.show(frame, only_print=idx % 5 != 0)
        
        # end = time.time()
        # if (end-start)>5:
        #     video_writer.release()
        #     cv2.destroyAllWindows()
        #     break


def split_video(video_path):
    # capture = cv2.VideoCapture('C:/Users/Saul/Documents/gaze-tracking-pipeline-main/2023-05-31_122045.mp4')
    capture = cv2.VideoCapture(video_path)
 
    frameNr = 0
    
    while (True):
    
        success, frame = capture.read()
    
        if success:
            # cv2.imwrite(f'C:/Users/Saul/Documents/gaze-tracking-pipeline-main/frames/frame_{frameNr}.png', frame)
            cv2.imwrite(f'./frames/frame_{frameNr}.png', frame)
    
        else:
            break
    
        frameNr = frameNr+1
    
    capture.release()

# def calibration(image_path, every_nth: int = 1, debug: bool = False, chessboard_grid_size=(7, 7)):
def calibration(image_path, every_nth: int = 1, debug: bool = False, chessboard_grid_size=(8, 7)):    
    """
    Perform camera calibration on the previously collected images.
    Creates `calibration_matrix.yaml` with the camera intrinsic matrix and the distortion coefficients.

    :param image_path: path to all png images
    :param every_nth: only use every n_th image
    :param debug: preview the matched chess patterns
    :param chessboard_grid_size: size of chess pattern
    :return:
    """

    x, y = chessboard_grid_size
    # x, y = 7,6

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((y * x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(f'{image_path}/*.png')[::every_nth]
    print('images number = ', len(images))

    found = 0
    for fname in images:
        img = cv2.imread(fname)  # Capture frame-by-frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
            # Cambiar el (11,11) por (7,7) para tener la dimension de cada cuadrito en 15mm
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            found += 1

            cv2.drawChessboardCorners(img, chessboard_grid_size, corners2, ret)
            # cv2.imwrite('./frames_for_DeepARC/frames/frame{:02d}.jpg'.format(found),img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    print("Number of images used for calibration: ", found)

    # When everything done, release the capture
    cv2.destroyAllWindows()

    # calibration, rotation and translation vectors for camera
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('rms', rms)

    # transform the matrix and distortion coefficients to writable lists
    data = {
        'rms': np.asarray(rms).tolist(),
        'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()
    }

    # and save it to a file
    with open("calibration_matrix.yaml", "w") as f:
        yaml.dump(data, f)

    print(data)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--record", type=bool, default=False)
    parser.add_argument("--split", type=bool, default=False)
    parser.add_argument("--calibrate", type=bool, default=False)
    parser.add_argument("--video_path", type=bool, default=False, help='Add the ruth path of the video to be splitted')
    args = parser.parse_args()

    if args.record==True:
        # 1. record video a video with the checkboard in different position and inclination
        record_video(width=1280, height=720, fps=30)
    
    elif args.split==True:
        # 2. split video into frames
        split_video(args.video_path)

    elif args.calibrate==True:
        # 3. run calibration on images to extract camera matrix, rotation matrix, distant coefficient
        calibration('./frames2', 30, debug=True)

    
