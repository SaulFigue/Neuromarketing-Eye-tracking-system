
## -------------- FOR Target Orientation settings --------------
"""
import cv2

# Create a named window to display the video feed
cv2.namedWindow("Video Feed")

# Start capturing video from the default camera
capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video feed
    ret, frame = capture.read()
    
    # Display the captured frame in the named window
    cv2.imshow("Video Feed", frame)
    
    # Wait for a key press event for up to 10 milliseconds
    key = cv2.waitKey(10)
    
    # Check if a key was pressed
    if key != -1:
        # Print the value of the pressed key
        print("Pressed key:", key)
        
        # If the pressed key is 'q', exit the loop
        if key == ord('q'):
            break

# Release the video capture and destroy the named window
capture.release()
cv2.destroyAllWindows()
"""

## -------------- FOR Rotation and Translation matrices ----------------
# import math as m
# import numpy as np

# alpha = m.atan2(0.125837, -0.9892245)
# alpha_degrees = m.degrees(alpha)

# gamma = m.atan2(0.07483271, -0.9892245)
# gamma_degrees = m.degrees(gamma)

# print("alpha ≈", alpha, "radians")
# print("alpha ≈", alpha_degrees, "degrees")
# print("gamma ≈", gamma, "radians")
# print("gamma ≈", gamma_degrees, "degrees")

# numerator   = - 60.3 + 203
# denominator = 520

# p_angle = m.atan(numerator/denominator) - alpha
# p_hat = m.pi - p_angle

# print("p_angle", p_angle)
# print("p_hat", p_hat)

# R_matrix = np.asarray([[1,0,0],
#                       [0,m.cos(p_angle),-m.sin(p_angle)],
#                       [0,m.sin(p_angle),m.cos(p_angle)]])

# print ("R_matrix = ",R_matrix)


"""
---------------- This is for eyes center ccs ---------------
"""
# eh = y_eye_screen = 60.3 mm
# ew = x_eye_screen = 150.50 mm
# d = z_eye_screen  = 520 mm

# e_scs = np.asarray([[60.3],
#                     [150.50],
#                     [520]])

# # EN VEZ DE LOS E_CCS_LF Y E_CCS_RG.... PRUEBA EL FACE_CENTER
# e_ccs_lf = np.asarray([[46.26489256],
#                        [-11.0642478],
#                        [580.8423556]])

# e_ccs_rg = np.asarray([[-16.64101014],
#                        [-7.258023210],
#                        [580.87790406]])

# e_ccs = (e_ccs_lf + e_ccs_rg)/2

# print("e_css = ",e_ccs)
# #print("Trans_e_css = ",np.transpose(e_ccs))

# print("R_matrix*e_ccs = ",np.dot(R_matrix,e_ccs))

# T_vector =  e_scs - np.dot(R_matrix,e_ccs)

# print("T_vector = ",T_vector)

# -------------------------------

"""
R = [[1,0,0],
     [0,-0.92324345,0.38421548],
     [0,-0.38421548,-0.92324345]]

T = [45.48805879,-81.13341445,1052.75545811]
"""

# R = np.asarray([[1,0,0],
#      [0,-0.92324345,0.38421548],
#      [0,-0.38421548,-0.92324345]])

# T = np.asarray([45.48805879,-81.13341445,1052.75545811])

# print("R = ",R)
# print("T = ",T)


"""
---------------- This is for face center ccs ----------------
"""
import math as m
import numpy as np

alpha = m.atan2(0.02200218, -0.99863255)
alpha_degrees = m.degrees(alpha)

gamma = m.atan2(-0.04742457, -0.99863255)
gamma_degrees = m.degrees(gamma)

print("alpha ≈", alpha, "radians")
print("alpha ≈", alpha_degrees, "degrees")
print("gamma ≈", gamma, "radians")
print("gamma ≈", gamma_degrees, "degrees")

numerator   = - 60.3 + 203
denominator = 630

p_angle = m.atan(numerator/denominator) - alpha
p_hat = m.pi - p_angle

print("p_angle", p_angle)
print("p_hat", p_hat)

R_matrix = np.asarray([[1,0,0],
                      [0,m.cos(p_hat),-m.sin(p_hat)],
                      [0,m.sin(p_hat),m.cos(p_hat)]])

print ("R_matrix = ",R_matrix)


# eh = y_eye_screen = 60.3 mm
# ew = x_eye_screen = 150.50 mm
# d = z_eye_screen  = 630 mm

e_scs = np.asarray([[60.3],
                    [150.50],
                    [630]])


fc_ccs = np.asarray([[7.056461190],
                     [-67.63641704],
                     [650.80261675]])

T_vector =  e_scs - np.dot(R_matrix,fc_ccs)
print("T_vector = ",T_vector)

# --------------- TRANSFORMATION PARAMETERS ---------------

"""
R = [[1,0,0],
     [0,-0.92324345,0.38421548],
     [0,-0.38421548,-0.92324345]]

T = [45.48805879,-81.13341445,1052.75545811]
"""

