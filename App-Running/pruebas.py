# =================================================================
# ---------------- TO TEST EXTERNALS WEBCAMS ----------------------
# =================================================================
"""
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)

ret, frame = cap.read()
print(ret)
print(frame)

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.show()

cap.release()
"""

# =================================================================
# ---- TO GENERATE CAMERA CALIBRATION VIDEO FOR DeepARC PAGE ------
# =================================================================
"""
import cv2
import os

# Folder path containing the JPG images
folder_path = "./frames_for_DeepARC/frames/"

# Get the list of JPG files in the folder
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])

# Read the first image to get the dimensions
first_image_path = os.path.join(folder_path, file_list[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

# Define the output video path
output_path = "./frames_for_DeepARC/videos/Camera_Calibration.mp4"

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

frame_duration = int(0.3 * 30)  # 1.5 seconds at 30 frames per second
# Write each image to the video
for file_name in file_list:
    image_path = os.path.join(folder_path, file_name)
    frame = cv2.imread(image_path)

    # Write the frame multiple times to achieve the desired duration
    for _ in range(frame_duration):
        out.write(frame)
    

# Release the VideoWriter and close the output file
out.release()
"""

# =================================================================
# ---- TO GENERATE SCREEN CALIBRATION VIDEO FOR DeepARC PAGE ------
# =================================================================
"""
import cv2
import os

# Folder path containing the JPG images
folder_path = "C:/Users/Saul/Documents/screen-calibration/data/p00/3D_plots"

# Get the list of JPG files in the folder
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

# Read the first image to get the dimensions
first_image_path = os.path.join(folder_path, file_list[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

# Define the output video path
output_path = "C:/Users/Saul/Documents/screen-calibration/data/p00/3D_plots_videos/Screen_Calibration.mp4"

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

frame_duration = int(1 * 30)  # 1.5 seconds at 30 frames per second
# Write each image to the video
for file_name in file_list:
    image_path = os.path.join(folder_path, file_name)
    frame = cv2.imread(image_path)

    # Write the frame multiple times to achieve the desired duration
    for _ in range(frame_duration):
        out.write(frame)
    

# Release the VideoWriter and close the output file
out.release()
"""

# =================================================================
# ---- TO GENERATE SCREEN CALIBRATION VIDEO FOR DeepARC PAGE ------
# =================================================================

"""
from sklearn.linear_model import LinearRegression

def train_regression_model(predicted_coordinates, real_coordinates):
    # Crea un modelo de regresión lineal
    model = LinearRegression()

    # Entrena el modelo utilizando las coordenadas predichas y las coordenadas reales conocidas
    model.fit(predicted_coordinates, real_coordinates)

    return model

def estimate_ground_truth(predicted_coordinates, model):
    # Estima las coordenadas reales utilizando el modelo entrenado
    estimated_ground_truth = model.predict(predicted_coordinates)

    return estimated_ground_truth

# Coordenadas predichas y coordenadas reales conocidas para entrenar el modelo
predicted_coordinates_array = [[32,32], [454,32], [682,32], [909,32], [1334,32],
                               [32,255], [454,255], [682,255], [909,255], [1334,255],
                               [32,383], [454,383], [682,383], [909,383], [1334,383],
                               [32,511], [454,511], [682,511], [909,511], [1334,511],
                               [32,736], [454,736], [682,736], [909,736], [1334,511]]
real_coordinates_array = [[442,131], [447,186], [514,160], [600,198], [622,181],
                          [565,246], [537,230], [525,237], [529,241], [545,233],
                          [501,258], [512,256], [514,271], [480,264], [476,294],
                          [474,296], [489,325], [480,352], [470,380], [481,403],
                          [484,436], [478,465], [461,489], [466,519], [495,531]]

# Entrenar el modelo de regresión
model = train_regression_model(real_coordinates_array, predicted_coordinates_array)

# Nuevas coordenadas predichas para estimar las coordenadas reales desconocidas
# new_predicted_coordinates_array = [[443,132], [510,165], [615,170],
#                                    [503,250], [515,268], [472,291],
#                                    [479,433], [464,487], [493,528]]

# new_predicted_coordinates_array = [[442,131], [447,186], [514,160], [600,198], [622,181],
#                                    [565,246], [537,230], [525,237], [529,241], [545,233],
#                                    [501,258], [512,256], [514,271], [480,264], [476,294],
#                                    [474,296], [489,325], [480,352], [470,380], [481,403],
#                                    [484,436], [478,465], [461,489], [466,519], [495,531]]

new_predicted_coordinates_array = [[484,436]]

# Estimar las coordenadas reales desconocidas utilizando el modelo entrenado
estimated_ground_truth_array = estimate_ground_truth(new_predicted_coordinates_array, model)

print("Coordenadas predichas:")
for coord in new_predicted_coordinates_array:
    print(coord)

print("\nEstimación de las coordenadas reales:")
for coord in estimated_ground_truth_array:
    print(coord)

"""

from PIL import Image

# Open the image
image = Image.open("./AliExpress_Stimuli.png")  # Replace "your_image.jpg" with the actual image file path

# Resize the image to fit the screen size
screen_width, screen_height = 1366, 768
resized_image = image.resize((screen_width, screen_height))

# Save the resized image
resized_image.save("AliExpress_Stimuli2.png")  # Replace "resized_image.jpg" with the desired output file path







