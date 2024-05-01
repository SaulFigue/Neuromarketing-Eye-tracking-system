import os
import sys
import argparse
import cv2
import csv
import numpy
from matplotlib import pyplot, image
WINDOW_NAME = 'Heatmap generation'

def draw_display(dispsize, imagefile=None):
   

    # construct screen (black background)
    screen = numpy.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    # if an image location has been passed, draw the image

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = cv2.imread(imagefile)
        img = cv2.resize(img, (dispsize[0], dispsize[1]))
    
        cv2.imshow(WINDOW_NAME, img)

    if  cv2.waitKey(500) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return img

def gaussian(x, sx, y=None, sy=None):
  
    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = numpy.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))
    return M



def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):

    # IMAGE
    img = draw_display(dispsize, imagefile=img_path)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = gwh / 2
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strt + gazepoints[i][1] - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the is heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            y = int(y)
            x = int(x)
            gwh = int(gwh)
            i = int(i)
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    strt = int(strt)
    dispsize = (int(dispsize[0]), int(dispsize[1]))
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN
    
    opacidad = 0.5
    imagen_combinada = cv2.addWeighted(img, 1.0 - opacidad, heatmap, opacidad, 0)
    # draw heatmap on top of image
    cv2.imshow(WINDOW_NAME,imagen_combinada)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    
    # save the figure if a file name was provided
    if savefilename != None:
        cv2.imwrite(savefilename,imagen_combinada)


input_path = './participants/p00/gazepoints0.csv'

with open(input_path) as f:
	reader = csv.reader(f)
	raw = list(reader)
	
	gaza_data = []
	if len(raw[0]) == 2:
		gaze_data = list(map(lambda q: (int(q[0]), int(q[1]), 1), raw))
	else:
		gaze_data =  list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw))


monitor_dimension = [1366,768]
img_Stim = '1'
img_path = f"./Stimulus/{img_Stim}.png"
save_path = f"./participants/p00/Heatmaps/heatmap{img_Stim}.png"
draw_display(monitor_dimension,img_path)
draw_heatmap(gaze_data, monitor_dimension, imagefile=img_path, alpha=0.5, savefilename=save_path, gaussianwh=200, gaussiansd=None)

