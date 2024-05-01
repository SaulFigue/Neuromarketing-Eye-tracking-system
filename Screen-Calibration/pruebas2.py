import cv2
import numpy as np

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

wpd = 83 # width padding
hpd = 80 # height padding

wof = 600 # width offset
hof = 304 # height offset

target_points = [[wpd,hpd],[wpd+wof,hpd],[wpd+(wof*2),hpd],
                 [wpd,hpd+hof],[wpd+wof,hpd+hof],[wpd+(wof*2),hpd+hof],
                 [wpd,hpd+(hof*2)],[wpd+wof,hpd+(hof*2)],[wpd+(wof*2),hpd+(hof*2)]]

print(target_points)
"""
circle_diameter = 30
image_size = (1366, 768)
circle_color = (255, 255, 255)  # Blanco

# Crear una imagen en blanco

idx = 0
for point in target_points:
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    center_x, center_y = point
    
    # Dibujar el círculo en la imagen
    cv2.circle(image, (center_x, center_y), circle_diameter // 2, circle_color, -1)

    # Guardar la imagen en un archivo
    cv2.imwrite(f"./circulos/imagen{idx}.png", image)
    idx += 1

# Mostrar la imagen
cv2.imshow("Circulos", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""