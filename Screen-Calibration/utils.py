import random
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Tuple, Union

import cv2
import numpy as np
import yaml
import csv


from webcam import WebcamSource


def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """
    Get monitor dimensions from Gdk.
    from on https://github.com/NVlabs/few_shot_gaze/blob/master/demo/monitor.py
    :return: tuple of monitor width and height in mm and pixels or None
    """
    try:
       import tkinter as tk

       root = tk.Tk()
       root.withdraw()

       # h_mm = 203,  w_mm = 361
       h_mm = root.winfo_screenmmheight()
       w_mm = root.winfo_screenmmwidth()

       # h_px = 768,  w_px = 1366
       h_pixels = root.winfo_screenheight() 
       w_pixels = root.winfo_screenwidth()

       return (w_mm, h_mm), (w_pixels, h_pixels)

    except ModuleNotFoundError:
        return None, None

# For Screen calibration, fixed points will be used.
"""
    1.- First = O
    2.- First = X
    3.- First = *
    ___________________________________
    |  O             O             O  |
    |         *      X       *        |
    |                                 |
    |  O      X      O       X     O  |
    |                                 |
    |         *      X       *        |
    |  O             O             O  |
    ___________________________________

"""

mm, pxl = get_monitor_dimensions()

screen_w = pxl[0]
screen_h = pxl[1]
wpd = 83 # width padding
hpd = 80 # height padding

wof = 600 # width offset
hof = 304 # height offset

target_points = [[wpd,hpd],[wpd+wof,hpd],[wpd+(wof*2),hpd],
                 [wpd,hpd+hof],[wpd+wof,hpd+hof],[wpd+(wof*2),hpd+hof],
                 [wpd,hpd+(hof*2)],[wpd+wof,hpd+(hof*2)],[wpd+(wof*2),hpd+(hof*2)]]
"""
wpd = 33 # width padding
hpd = 30 # height padding

wof = 325 # width offset
hof = 236 # height offset

target_points = [[wpd,hpd],[wpd+wof,hpd],[wpd+(wof*2),hpd],[wpd+(wof*3),hpd],[wpd+(wof*4),hpd],
                 [wpd,hpd+hof],[wpd+wof,hpd+hof],[wpd+(wof*2),hpd+hof],[wpd+(wof*3),hpd+hof],[wpd+(wof*4),hpd+hof],
                 [wpd,hpd+(hof*2)],[wpd+wof,hpd+(hof*2)],[wpd+(wof*2),hpd+(hof*2)],[wpd+(wof*3),hpd+(hof*2)],[wpd+(wof*4),hpd+(hof*2)],
                 [wpd,hpd+(hof*3)],[wpd+wof,hpd+(hof*3)],[wpd+(wof*2),hpd+(hof*3)],[wpd+(wof*3),hpd+(hof*3)],[wpd+(wof*4),hpd+(hof*3)]]

"""
# target_points = [[32, 32],[screen_w/3 - 1, 32],[screen_w/2 - 1, 32],[(2*screen_w)/3 - 1, 32],[screen_w - 32, 32],
#                  [32, screen_h/3 - 1],[screen_w/3 - 1, screen_h/3 - 1],[screen_w/2 - 1, screen_h/3 - 1],[(2*screen_w)/3 - 1, screen_h/3 - 1],[screen_w - 32, screen_h/3 - 1],
#                  [32, screen_h/2 - 1],[screen_w/3 - 1, screen_h/2 - 1],[screen_w/2 - 1, screen_h/2 - 1],[(2*screen_w)/3 - 1, screen_h/2 - 1],[screen_w - 32, screen_h/2 - 1],
#                  [32, (2*screen_h)/3 - 1],[screen_w/3 - 1, (2*screen_h)/3 - 1],[screen_w/2 - 1, (2*screen_h)/3 - 1],[(2*screen_w)/3 - 1, (2*screen_h)/3 - 1],[screen_w - 32, (2*screen_h)/3 - 1],
#                  [32, screen_h - 32],[screen_w/3 - 1,screen_h - 32],[screen_w/2 - 1, screen_h - 32],[(2*screen_w)/3 - 1, screen_h - 32],[screen_w - 32, screen_h - 32]]

# target_points = [[32, 32],[screen_w/2 - 1, 32],[screen_w - 32, 32],
#                  [32, screen_h/2 - 1],[screen_w/2 - 1, screen_h/2 - 1],[screen_w - 32, screen_h / 2 - 1],
#                  [32, screen_h - 32],[screen_w/2 - 1, screen_h - 32],[screen_w - 32, screen_h - 32],
#                  [screen_w/2 - 1, screen_h/2 - 1],[screen_w/3 - 1, screen_h/2 -1],[(2*screen_w)/3 - 1, screen_h/2 -1],
#                  [screen_w/2 - 1, screen_h/2 - 1],[screen_w/2 - 1, screen_h/3 -1],[screen_w/2 - 1, (2*screen_h)/3 -1],
#                  [screen_w/2 - 1, screen_h/2 - 1],[screen_w/3 - 1, screen_h/3 -1],[(2*screen_w)/3 - 1, screen_h/3 -1],
#                  [screen_w/3 - 1, (2*screen_h)/3 -1],[(2*screen_w)/3 - 1, (2*screen_h)/3 -1], [screen_w/2 - 1, screen_h/2 - 1]]

# target_points = [[32, 32],[screen_w/2 - 1, 32],[screen_w - 32, 32],
#                  [32, screen_h/2 - 1],[screen_w/2 - 1, screen_h/2 - 1],[screen_w - 32, screen_h / 2 - 1],
#                  [32, screen_h - 32],[screen_w/2 - 1, screen_h - 32],[screen_w - 32, screen_h - 32]]

# target_points = [[screen_w/2 - 1, screen_h/2 - 1]]

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2


class TargetOrientation(Enum):
    # /////////////////////
    UP = 119    # W
    DOWN = 115  # S
    LEFT = 97   # A
    RIGHT = 100 # D
    # /////////////////////

def create_image(monitor_pixels: Tuple[int, int], center=(0, 0), circle_scale=0.7, orientation=TargetOrientation.RIGHT, target='E') -> Tuple[np.ndarray, float, bool]:
    """
    Create image to display on screen.

    :param monitor_pixels: monitor dimensions in pixels
    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param orientation: orientation of the target
    :param target: char to write on image
    :return: created image, new smaller circle_scale and bool that indicated if it is th last frame in the animation
    """
    width, height = monitor_pixels

    if orientation == TargetOrientation.LEFT or orientation == TargetOrientation.RIGHT:
        img = np.zeros((height, width, 3), np.float32)    # Black background window display
        img.fill(255) # White window display

        if orientation == TargetOrientation.LEFT:
            center = (width - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.LEFT:
            img = cv2.flip(img, 1)
    else:  # TargetOrientation.UP or TargetOrientation.DOWN
        img = np.zeros((width, height, 3), np.float32)    # Black background window display
        img.fill(255)  # White background window display
        center = (center[1], center[0])

        if orientation == TargetOrientation.UP:
            center = (height - center[0], center[1])

        end_animation_loop = write_text_on_image(center, circle_scale, img, target)

        if orientation == TargetOrientation.UP:
            img = cv2.flip(img, 1)

        img = img.transpose((1, 0, 2))

    return img / 255, circle_scale * 0.9, end_animation_loop


def write_text_on_image(center: Tuple[int, int], circle_scale: float, img: np.ndarray, target: str):
    """
    Write target on image and check if last frame of the animation.

    :param center: center of the circle and the text
    :param circle_scale: scale of the circle
    :param img: image to write data on
    :param target: char to write
    :return: True if last frame of the animation
    """
    text_size, _ = cv2.getTextSize(target, FONT, TEXT_SCALE, TEXT_THICKNESS)
    # cv2.circle(img, center, int(text_size[0] * 5 * circle_scale), (32, 32, 32), -1)
    cv2.circle(img, center, int(text_size[0] * 5 * circle_scale), (0, 0, 0), -1)
    text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

    end_animation_loop = circle_scale < random.uniform(0.2, 0.5)
    if not end_animation_loop:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (17, 112, 170), TEXT_THICKNESS, cv2.LINE_AA)
    else:
        cv2.putText(img, target, text_origin, FONT, TEXT_SCALE, (252, 125, 11), TEXT_THICKNESS, cv2.LINE_AA)

    return end_animation_loop


# def get_random_position_on_screen(monitor_pixels: Tuple[int, int]) -> Tuple[int, int]:
#     """
#     Get random valid position on monitor.

#     :param monitor_pixels: monitor dimensions in pixels
#     :return: tuple of random valid x and y coordinated on monitor
#     """
#     return int(random.uniform(0, 1) * monitor_pixels[0]), int(random.uniform(0, 1) * monitor_pixels[1])

# def show_point_on_screen(window_name: str, base_path: str, monitor_pixels: Tuple[int, int], source: WebcamSource) -> Tuple[str, Tuple[int, int], float]:
#     """
#     Show one target on screen, full animation cycle. Return collected data if data is valid

#     :param window_name: name of the window to draw on
#     :param base_path: path where to save the image to
#     :param monitor_pixels: monitor dimensions in pixels
#     :param source: webcam source
#     :return: collected data otherwise None
#     """
#     circle_scale = 1.
#     center = get_random_position_on_screen(monitor_pixels)
#     end_animation_loop = False
#     orientation = random.choice(list(TargetOrientation))

#     file_name = None
#     time_till_capture = None

#     while not end_animation_loop:
#         image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
#         cv2.imshow(window_name, image)

#         for _ in range(10):  # workaround to not speed up the animation when buttons are pressed
#             if cv2.waitKey(50) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 sys.exit()

#     if end_animation_loop:
#         file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         # print(file_name)
#         start_time_color_change = time.time()
#         # print('time_till_capture = '+str(time.time())+' - '+str(start_time_color_change))

#         while time.time() - start_time_color_change < 0.5:
#             if cv2.waitKey(42) & 0xFF == orientation.value:
#                 source.clear_frame_buffer()
#                 cv2.imwrite(f'{base_path}/{file_name}.jpg', next(source))
#                 time_till_capture = time.time() - start_time_color_change
#                 break

#     cv2.imshow(window_name, np.zeros((monitor_pixels[1], monitor_pixels[0], 3), np.float32))
#     cv2.waitKey(500)

#     return f'{file_name}.jpg', center, time_till_capture

def get_new_target_on_screen():
    """
    Get the next target position on monitor from fixed target_points array.

    :return: tuple of x and y coordinated on monitor
    """
    x = target_points[0][0]
    y = target_points[0][1]

    target_points.pop(0)

    return int(x), int(y)


from collections import defaultdict
import pandas as pd
collected_data = defaultdict(list)

def show_calibration_points(bg_imgs: np.ndarray, point_on_screen: Tuple[int, int],window_name: str, base_path: str, monitor_pixels: Tuple[int, int], source: WebcamSource) -> Tuple[str, Tuple[int, int], float]:                     
    circle_scale = 1.
    center = get_new_target_on_screen() # Current Point on screen (x_pxl, y_pxl)
    end_animation_loop = False
    """
    bg_imgs = ['imagen0','imagen1','imagen2','imagen3','imagen4','imagen5','imagen6',
               'imagen7','imagen8','imagen9','imagen10','imagen11','imagen12','imagen13',
               'imagen14','imagen15','imagen16','imagen17','imagen18','imagen19']
    """

    file_name = None
    time_till_capture = None

    while not end_animation_loop:
        # Create visible circle target point
        bg_image = './circulos/'+str(bg_imgs[0])+'.png'
        background_image = cv2.imread(bg_image)
        image = cv2.resize(background_image, (monitor_pixels[0], monitor_pixels[1]))
        #image = background_image
        circle_scale = circle_scale - 0.1
        end_animation_loop = circle_scale < random.uniform(0.2, 0.5)
        # image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        cv2.imshow(window_name, image)

        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        collected_data['GT_Screen_P_pxl'].append(center)
        collected_data['PD_Screen_P_pxl'].append(point_on_screen)   # AQUI ESTA MAL
                        # EL PONT ON SCREEN SIEMPRE ES EL MISMO PORQ NO SE ESTA ACTUALIZNDO

        pd.DataFrame(collected_data).to_csv(f'{base_path}/data.csv',index=False)        

        cv2.imwrite(f'{base_path}/{file_name}.jpg', next(source))
        # time_till_capture = 0.500000#time.time() - start_time_color_change

        #for _ in range(6):  # workaround to not speed up the animation when buttons are pressed
        if cv2.waitKey(50) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()

        time.sleep(0.3) # To wait 1 second before changing the circle_scale

    """
    if end_animation_loop:
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time_color_change = time.time()

        while time.time() - start_time_color_change < 0.5:
            #if cv2.waitKey(42) & 0xFF == orientation.value: # Hay que eliminar esto 
                                                            # para no presionar ninguna letra
            source.clear_frame_buffer()
            cv2.imwrite(f'{base_path}/{file_name}.jpg', next(source))
            time_till_capture = time.time() - start_time_color_change
            break
       """ 
    cv2.imshow(window_name, np.zeros((monitor_pixels[1], monitor_pixels[0], 3), np.float32))
    cv2.waitKey(250)
    bg_image.pop(0)

    #return f'{file_name}.jpg', center, time_till_capture




def show_calibration_points0(window_name: str, base_path: str, monitor_pixels: Tuple[int, int], source: WebcamSource) -> Tuple[str, Tuple[int, int], float]:
    """
    Show one target on screen, full animation cycle. Return collected data if data is valid

    :param window_name: name of the window to draw on
    :param base_path: path where to save the image to
    :param monitor_pixels: monitor dimensions in pixels
    :param source: webcam source
    :return: collected data otherwise None
    """
                     
    circle_scale = 1.
    center = get_new_target_on_screen() # Current Point on screen (x_pxl, y_pxl)
    end_animation_loop = False
    orientation = random.choice(list(TargetOrientation))

    file_name = None
    time_till_capture = None

    while not end_animation_loop:
        # Create visible circle target point
        image, circle_scale, end_animation_loop = create_image(monitor_pixels, center, circle_scale, orientation)
        cv2.imshow(window_name, image)

        for _ in range(6):  # workaround to not speed up the animation when buttons are pressed
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()

    if end_animation_loop:
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time_color_change = time.time()

        while time.time() - start_time_color_change < 0.5:
            #if cv2.waitKey(42) & 0xFF == orientation.value: # Hay que eliminar esto 
                                                            # para no presionar ninguna letra
            source.clear_frame_buffer()
            cv2.imwrite(f'{base_path}/{file_name}.jpg', next(source))
            time_till_capture = time.time() - start_time_color_change
            break
        


    cv2.imshow(window_name, np.zeros((monitor_pixels[1], monitor_pixels[0], 3), np.float32))
    cv2.waitKey(500)

    return f'{file_name}.jpg', center, time_till_capture


# De aqui ya empiezo a meter funcciones necesarias para correr el modelo a la par

def get_camera_matrix(base_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera_matrix and dist_coefficients from `{base_path}/calibration_matrix.yaml`.

    :param base_path: base path of data
    :return: camera intrinsic matrix and dist_coefficients
    """
    with open(f'{base_path}/calibration_matrix.yaml', 'r') as file:
        calibration_matrix = yaml.safe_load(file)
    camera_matrix = np.asarray(calibration_matrix['camera_matrix']).reshape(3, 3)
    dist_coefficients = np.asarray(calibration_matrix['dist_coeff'])
    return camera_matrix, dist_coefficients

def get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, shape, results, face_model, face_model_all, landmarks_ids):
    """
    Fit `face_model` onto `face_landmarks` using `solvePnP`.

    :param camera_matrix: camera intrinsic matrix
    :param dist_coefficients: distortion coefficients
    :param shape: image shape
    :param results: output of MediaPipe FaceMesh
    :return: full face model in the camera coordinate system
    """
    height, width, _ = shape
    face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
    face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

    rvec, tvec = None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Initial fit
    for _ in range(10):
        success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy

    head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))
    return np.dot(head_rotation_matrix, face_model.T) + tvec.reshape((3, 1)), np.dot(head_rotation_matrix, face_model_all.T) + tvec.reshape((3, 1))  # 3D positions of facial landmarks

def gaze_2d_to_3d(gaze: np.ndarray) -> np.ndarray:
    """
    pitch and yaw to 3d vector

    :param gaze: 2D gaze values (pitch, yaw)
    :return: 3d vector
    """
    x = -np.cos(gaze[0]) * np.sin(gaze[1])
    y = -np.sin(gaze[0])
    z = -np.cos(gaze[0]) * np.cos(gaze[1])
    return np.array([x, y, z])

def ray_plane_intersection(support_vector: np.ndarray, direction_vector: np.ndarray, plane_normal: np.ndarray, plane_d: np.ndarray) -> np.ndarray:
    """
    Calulate the intersection between the gaze ray and the plane that represents the monitor.

    :param support_vector: support vector of the gaze
    :param direction_vector: direction vector of the gaze
    :param plane_normal: normal of the plane
    :param plane_d: d of the plane
    :return: point in 3D where the the person is looking at on the screen
    """
    a11 = direction_vector[1]
    a12 = -direction_vector[0]
    b1 = direction_vector[1] * support_vector[0] - direction_vector[0] * support_vector[1]

    a22 = direction_vector[2]
    a23 = -direction_vector[1]
    b2 = direction_vector[2] * support_vector[1] - direction_vector[1] * support_vector[2]

    line_w = np.array([[a11, a12, 0], [0, a22, a23]])
    line_b = np.array([[b1], [b2]])

    matrix = np.insert(line_w, 2, plane_normal, axis=0)
    bias = np.insert(line_b, 2, plane_d, axis=0)

    return np.linalg.solve(matrix, bias).reshape(3)


def plane_equation(rmat: np.ndarray, tmat: np.ndarray) -> np.ndarray:
    """
    Computes the equation of x-y plane.
    The normal vector of the plane is z-axis in rotation matrix. 
    And tmat provide on point in the plane.

    :param rmat: rotation matrix
    :param tmat: translation matrix
    :return: (a, b, c, d), where the equation of plane is ax + by + cz = d
    """

    assert type(rmat) == type(np.zeros(0)) and rmat.shape == (3, 3), "There is an error about rmat."
    assert type(tmat) == type(np.zeros(0)) and tmat.size == 3, "There is an error about tmat."

    n = rmat[:, 2]
    origin = np.reshape(tmat, (3))

    a = n[0]
    b = n[1]
    c = n[2]

    d = origin[0] * n[0] + origin[1] * n[1] + origin[2] * n[2]
    return np.array([a, b, c, d])


def get_point_on_screen(monitor_mm: Tuple[float, float], monitor_pixels: Tuple[float, float], result: np.ndarray) -> Tuple[int, int]:
    """
    Calculate point in screen in pixels.

    :param monitor_mm: dimensions of the monitor in mm
    :param monitor_pixels: dimensions of the monitor in pixels
    :param result: predicted point on the screen in mm
    :return: point in screen in pixels
    """

    result_x = result[0]
    result_x = -result_x + monitor_mm[0] / 2
    result_x = result_x * (monitor_pixels[0] / monitor_mm[0])

    result_y = result[1]
    result_y = result_y - 10  # 20 mm offset
    result_y = min(result_y, monitor_mm[1])
    result_y = result_y * (monitor_pixels[1] / monitor_mm[1])

    return tuple(np.asarray([result_x, result_y]).round().astype(int))


