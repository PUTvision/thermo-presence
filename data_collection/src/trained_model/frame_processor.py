import os

import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


# from config import IR_CAMERA_RESOLUTION, TEMPERATURE_NORMALIZATION__MIN, TEMPERATURE_NORMALIZATION__MAX
IR_CAMERA_RESOLUTION_X = 32
IR_CAMERA_RESOLUTION_Y = 24

IR_CAMERA_RESOLUTION = (IR_CAMERA_RESOLUTION_Y, IR_CAMERA_RESOLUTION_X)

# for frames normalization
TEMPERATURE_NORMALIZATION__MIN = 20
TEMPERATURE_NORMALIZATION__MAX = 35


from typing import Tuple
import torch
from torch import nn
class AutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.conv = DoubleConv(32, 64, 3, 1)
        self.upconv1 = ExpandBlock(64, 32, 3, 1)
        self.upconv2 = ExpandBlock(32, 16, 3, 1)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
    def forward(self, x):
        # downsampling part
        x, conv2, conv1 = self.encoder(x)
        x = self.conv(x)
        x = self.upconv1(x, conv2)
        x = self.upconv2(x, conv1)
        x = self.out_conv(x)

        x = x[:, 0, :, :]  # get rid of one dimension
        return x


class ExpandBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = torch.cat((x, encoder_features), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = ContractBlock(in_channels, 16, 3, 1)
        self.conv2 = ContractBlock(16, 32, 3, 1)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, conv1 = self.conv1(x)
        x, conv2 = self.conv2(x)
        return x, conv2, conv1
    def forward_simple(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ContractBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        x = self.pool(features)
        return x, features


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )



class FrameProcessor:
    def __init__(self):
        self.latest_output_frame = None
        self.sum_of_values_for_one_person = 52
        #self.model = UNET(1, 1).double()
        self.model = AutoEncoder(1, 1).double()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'unet_v2small_cpu2')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.train(False)

    def process_frame(self, raw_frame):
        frame_2d = np.reshape(raw_frame, IR_CAMERA_RESOLUTION)

        frame_normalized = (frame_2d - TEMPERATURE_NORMALIZATION__MIN) * \
                           (1 / (TEMPERATURE_NORMALIZATION__MAX - TEMPERATURE_NORMALIZATION__MIN))
        frame_for_model = torch.tensor(frame_normalized)[np.newaxis, :, :][np.newaxis, :, :, :]
        with torch.no_grad():
            model_out_frame = self.model(frame_for_model)[0]
        self.latest_output_frame = model_out_frame.numpy()
        return self.latest_output_frame

    def get_people_count_on_latest_frame(self):
        if self.latest_output_frame is None:
            return -1
        return np.sum(self.latest_output_frame) / self.sum_of_values_for_one_person


if __name__ == '__main__':
    fp = FrameProcessor()
    raw_frame = np.array([26.30,25.40,24.62,24.32,23.96,24.13,24.02,23.67,23.98,24.12,23.89,23.96,24.26,23.89,24.10,24.15,24.16,23.91,24.30,24.62,24.57,24.73,24.86,24.35,24.54,24.76,25.17,24.54,24.62,24.49,24.64,24.72,24.94,25.28,24.57,25.03,24.21,24.33,23.74,24.34,24.12,24.20,24.01,24.31,24.15,24.42,24.14,24.20,24.32,24.28,24.25,24.69,24.66,25.09,25.00,25.21,24.49,24.95,24.66,24.98,25.36,25.47,24.80,25.06,25.55,25.09,26.00,26.18,24.42,23.54,23.89,23.94,24.17,23.78,24.33,23.90,24.26,24.18,24.32,24.03,24.32,23.96,24.38,24.50,26.70,27.40,27.41,27.27,25.65,25.27,25.30,24.98,25.42,24.89,25.21,25.87,25.52,
                          25.90,26.28,26.04,24.06,24.37,24.00,24.23,23.92,24.04,24.41,24.46,24.30,24.51,24.62,24.35,24.12,24.29,24.15,24.53,28.01,28.55,28.44,28.22,25.69,25.75,25.12,25.37,25.12,25.68,25.49,24.66,26.08,26.11,27.20,25.16,24.86,24.53,24.05,24.05,24.25,24.31,24.10,24.37,24.56,24.24,24.22,24.13,24.48,24.38,24.39,24.77,28.24,28.71,28.54,28.01,26.94,26.82,25.83,25.53,25.25,25.18,25.11,25.25,25.66,26.08,25.53,25.00,24.79,25.03,24.30,24.13,24.43,24.37,24.14,24.36,24.16,24.19,
                          24.25,24.44,24.24,24.25,24.21,24.85,28.01,28.34,28.12,27.89,26.98,28.14,26.41,25.86,25.07,25.78,24.80,25.27,26.23,25.72,24.81,24.51,25.44,24.68,24.74,24.50,24.68,24.29,24.09,24.39,24.23,24.17,24.30,24.32,24.29,24.60,24.40,24.49,26.57,27.82,28.14,27.34,26.90,27.69,26.14,25.33,25.38,25.06,24.48,24.77,26.18,26.82,24.87,24.96,25.49,25.17,24.40,25.13,24.25,25.18,24.17,24.11,24.49,24.18,24.35,24.49,24.33,24.53,24.51,24.74,26.17,27.09,27.33,26.04,26.09,26.34,25.47,25.21,24.75,24.56,24.49,25.30,26.54,26.63,24.94,25.41,25.32,24.74,25.46,25.40,27.16,25.78,26.40,24.17,24.26,24.28,24.38,
                          24.29,24.43,24.62,24.58,24.62,25.09,25.15,25.16,24.91,24.54,24.51,24.49,24.58,24.63,24.68,24.95,24.84,26.81,27.32,25.08,25.07,25.40,25.34,26.05,27.11,27.12,28.91,25.48,26.05,24.26,24.44,24.60,24.49,24.58,24.66,24.62,24.76,24.23,24.54,24.13,24.28,24.50,24.55,24.67,24.48,24.67,24.53,24.29,25.18,26.62,27.28,25.30,25.43,25.79,25.55,26.41,27.44,28.50,29.07,27.70,25.46,24.76,24.37,24.78,24.71,24.71,24.68,24.82,24.42,24.18,23.88,24.17,24.42,24.78,24.77,24.51,
                          24.43,24.75,24.35,25.17,24.62,26.31,27.40,25.72,25.93,25.43,25.42,27.39,27.29,28.52,28.17,26.92,26.39,24.81,24.85,24.49,24.57,24.76,24.69,24.43,24.32,23.96,24.09,24.06,24.61,24.59,24.72,24.72,24.28,24.38,24.40,24.65,25.14,25.81,25.71,25.59,25.74,25.27,25.78,26.65,27.84,27.82,27.98,25.83,25.94,24.86,24.90,24.45,24.64,24.83,24.55,24.52,24.44,24.35,24.32,24.44,24.51,24.78,24.32,24.63,24.85,24.75,24.55,24.79,24.73,25.85,25.57,25.36,25.60,25.39,25.37,26.59,25.84,28.31,27.04,25.86,24.87,24.95,24.89,24.59,24.96,24.70,24.90,24.66,24.62,24.29,24.46,24.55,24.51,24.52,24.57,24.42,24.81,
                          24.63,24.63,24.67,24.87,25.46,25.05,25.19,25.09,25.14,25.17,25.19,25.73,26.17,25.72,25.22,25.33,25.07,24.83,25.12,24.92,25.15,24.74,24.93,24.85,25.22,24.73,24.94,24.65,24.97,24.63,24.85,24.70,24.67,24.77,24.69,25.17,24.88,25.76,25.40,25.35,25.39,25.33,25.68,25.47,25.82,25.54,25.23,25.40,24.99,25.17,25.04,25.26,25.05,25.13,25.17,24.91,24.99,25.36,24.96,25.15,24.90,24.81,24.62,24.97,24.47,25.01,24.54,25.29,26.57,25.80,26.06,25.54,27.46,26.74,25.75,25.52,
                          25.11,25.13,25.28,25.50,25.31,25.43,25.15,25.49,25.30,25.14,24.98,25.01,25.28,25.03,25.04,24.97,25.05,24.93,25.05,24.65,24.69,24.56,25.15,25.40,26.88,26.72,25.88,26.11,27.62,27.41,25.54,25.57,25.48,25.37,25.45,25.44,25.64,25.62,25.62,25.55,25.12,25.32,24.53,24.95,25.11,25.23,25.20,25.19,24.97,24.99,24.98,24.88,24.79,25.04,24.98,25.18,27.10,26.77,26.94,26.84,29.03,28.50,26.17,25.22,25.72,25.36,25.69,25.70,25.62,25.62,26.09,25.74,25.48,25.22,24.93,24.73,
                          24.96,25.17,25.48,25.34,25.40,25.25,24.78,24.93,25.20,24.62,25.33,25.04,26.88,27.34,27.59,28.39,29.27,28.78,25.72,25.85,25.38,25.67,25.49,25.41,25.82,25.75,26.16,25.85,25.45,25.30,24.62,24.80,25.18,25.45,25.04,25.55,24.91,25.39,25.01,24.99,25.17,25.25,24.58,24.80,27.64,27.09,27.18,27.96,27.94,27.12,25.31,25.52,26.18,25.46,25.52,25.65,25.70,25.92,25.98,25.79,25.72,25.17,24.68,25.21,25.81,25.36,25.26,25.17,25.26,24.71,24.90,24.65,25.05,24.69,24.88,24.96,26.91,27.62,27.37,27.54,26.78,26.63,25.92,26.13,25.69,26.03,25.89,25.60,25.45,25.69,25.87,25.65,25.48,25.41,24.58,25.04,25.57,
                          25.62,25.32,24.97,24.99,25.05,24.94,25.18,24.94,25.19,25.00,25.65,27.27,25.84,26.61,26.67,27.08,25.93,25.68,25.48,25.85,25.53,26.04,25.74,26.05,25.46,25.65,25.39,25.72,25.60,25.83,25.77,26.02,26.23,25.12,25.31,25.49,25.31,25.32,25.28,25.29,25.52,25.61,25.92,27.54,27.43,26.99,27.30,26.67,26.24,26.08,26.16,25.60,25.32,25.40,25.58,25.31,25.52,25.60,25.67,25.09,25.38,25.73,26.24,25.78,25.92,25.74,25.94,25.08,25.43,24.95,25.63,25.23,25.43,25.33,25.69])
    pf = fp.process_frame(raw_frame)
    print(pf)
    print(fp.get_people_count_on_latest_frame())



