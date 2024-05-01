"""
from renset import resnet50
from dataset.mpii_face_gaze_dataset import get_dataloaders
import torch.nn as nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


train_dataloader, valid_dataloader, test_dataloader = get_dataloaders("./data", 1, 0, 64)

writer = SummaryWriter("torchlogs/")
model = resnet50(pretrained=True)
writer.add_graph(model, train_dataloader)
writer.close()
"""

# --------------------------------------------------------------
#                   PRUEBAS LEER ARCHIVO H5
# --------------------------------------------------------------

import h5py

file = h5py.File('./data/data.h5', 'r')

print(list(file.keys()))

file_name_base = file['file_name_base']
gaze_location = file['gaze_location']
gaze_pitch = file['gaze_pitch']
gaze_yaw = file['gaze_yaw']
screen_size = file['screen_size']

print('file_name_base = ',list(file_name_base[:1]))
print('gaze_location = ',list(gaze_location[:1]))
print('gaze_pitch = ',list(gaze_pitch[:1]))
print('gaze_yaw = ',list(gaze_yaw[:1]))
print('screen_size = ',list(screen_size[:1]))

# --------------------------------------------------------------
#                   PRUEBAS ESCRIBIR ARCHIVO H5
# --------------------------------------------------------------
# import h5py
# import pandas as pd
# import numpy as np

# # Read the CSV file into a DataFrame
# dataframe = pd.read_csv('./data/data.csv')

# # Convert the data types of the problematic columns
# # dataframe['column_name'] = dataframe['column_name'].astype(dtype=tuple)

# # Create an HDF5 file
# file = h5py.File('./data/data2.h5', 'w')

# # Store the converted columns in the HDF5 file
# for column_name, column_data in dataframe.items():
#     # Check if the column data type is compatible with HDF5
#     if column_data.dtype == np.object:
#         column_data = column_data.astype('string')  # Convert to string data type
#     dataset = file.create_dataset(column_name, data=column_data)

# # Close the HDF5 file
# file.close()







# model = resnet50(pretrained=True)  #ReLU 3-65 [16,512,12,12]

# model.summarize(max_depth=1)

"""
model = resnet50(pretrained=True)
# Tiene que llegar a un output shape = [16, 128, 6, 6]

cnn_face = nn.Sequential(
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1,
    model.layer2,   # This ends at: ReLU 3-65, with Output Shape = [16,512,12,12]

    # Let's add some convolutions and ReLu and BatchNorm similar to Vgg16 to ends with 128,6,6

    nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  #Output Shape = [16,128,12,12]
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3,3), bias=False),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  #Output Shape = [16,128,6,6]
    nn.ReLU(inplace=True),
)
"""

"""
# Tiene que llegar a un output shape = [16, 128, 4, 6]
cnn_eye = nn.Sequential(
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1,
    model.layer2, # This ends at: ReLU 3-65, with Output Shape = [16,512,8,12]

    #Let's add some convolutions and ReLu and BatchNorm similar to Vgg16 to ends with 128,4,6

    nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  #Output Shape = [16,128,8,12]
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2,3), bias=False),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  #Output Shape = [16,128,4,6]
    nn.ReLU(inplace=True),
)
"""

# print(cnn_face)
# print(cnn_eye)

"""

batch_size = 16
summary(model, 
    # (batch_size, 1),
    (batch_size, 3, 96, 96))  # full face
    # (batch_size, 3, 64, 96),  # right eye
    # (batch_size, 3, 64, 96)  # left eye
# ), dtypes=[torch.long, torch.float, torch.float, torch.float])

"""






"""
import h5py
filename = "./data/data.h5"

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    data = list(f[a_group_key])

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array

    print('ds_obj = ',ds_obj)
    print('ds_arr = ',ds_arr)

"""
    ## ///////////////////////////// USING RESNET-50 ///////////////////////////////
"""   
         self.cnn_face = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Sequential(*list(models.resnet50(pretrained=True).children())[4:-2]),  # transfer learning from ResNet-50 pretrained on ImageNet
            nn.Conv2d(2048, 128, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )


"""


## -----------------------------------------------------------------------------
## -----------------------------------------------------------------------------
## -----------------------------------------------------------------------------

"""
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchinfo import summary
from torchvision import models


class SELayer(nn.Module):
    """
    # Squeeze-and-Excitation layer

    # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    
"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(  # Excitation (similar to attention)
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FinalModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subject_biases = nn.Parameter(torch.zeros(15 * 2, 2))  # pitch and yaw offset for the original and mirrored participant

        self.cnn_face = nn.Sequential(
            models.resnet50(pretrained=True).conv1,
            models.resnet50(pretrained=True).bn1,
            models.resnet50(pretrained=True).relu,
            models.resnet50(pretrained=True).maxpool,
            models.resnet50(pretrained=True).layer1,
            models.resnet50(pretrained=True).layer2,
            models.resnet50(pretrained=True).layer3,
            models.resnet50(pretrained=True).layer4,
        )

        self.cnn_eye = nn.Sequential(
            models.resnet50(pretrained=True).conv1,
            models.resnet50(pretrained=True).bn1,
            models.resnet50(pretrained=True).relu,
            models.resnet50(pretrained=True).maxpool,
            models.resnet50(pretrained=True).layer1,
            models.resnet50(pretrained=True).layer2,
            models.resnet50(pretrained=True).layer3,
            models.resnet50(pretrained=True).layer4,
        )

        self.fc_face = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.cnn_eye2fc = nn.Sequential(
            SELayer(2048),

            nn.Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            SELayer(256),

            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            SELayer(128),
        )

        self.fc_eye = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten"""