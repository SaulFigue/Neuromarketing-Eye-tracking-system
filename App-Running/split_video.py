import cv2
 
# Mejorar esto, que se automatice, que la ruta del video.mp4 
# se saque con un codigo y se guarde en una variable string
# que luego sera ingresada dentro del cv2.VideoCapture
# Asi no se ingresara manualmente

capture = cv2.VideoCapture('C:/Users/Saul/Documents/gaze-tracking-pipeline-main/2023-05-31_122045.mp4')
 
frameNr = 0
 
while (True):
 
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(f'C:/Users/Saul/Documents/gaze-tracking-pipeline-main/frames2/frame_{frameNr}.png', frame)
        # cv2.imwrite(f'C:/Users/Saul/Documents/gaze-tracking-pipeline-main/frames/frame_{frameNr}.png', frame)
 
    else:
        break
 
    frameNr = frameNr+1
 
capture.release()