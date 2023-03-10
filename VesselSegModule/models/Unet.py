# implement by chatgpt
import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, 3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.max_pool2d(x1, 2)
        x2 = nn.functional.relu(self.conv2(x2))
        x3 = nn.functional.max_pool2d(x2, 2)
        x3 = nn.functional.relu(self.conv3(x3))
        x4 = nn.functional.max_pool2d(x3, 2)
        x4 = nn.functional.relu(self.conv4(x4))
        x5 = nn.functional.max_pool2d(x4, 2)
        x5 = nn.functional.relu(self.conv5(x5))
        x6 = nn.functional.relu(self.upconv1(x5))
        x6 = torch.cat([x4, x6], dim=1)
        x6 = nn.functional.relu(self.conv6(x6))
        x7 = nn.functional.relu(self.upconv2(x6))
        x7 = torch.cat([x3, x7], dim=1)
        x7 = nn.functional.relu(self.conv7(x7))
        x8 = nn.functional.relu(self.upconv3(x7))
        x8 = torch.cat([x2, x8], dim=1)
        x8 = nn.functional.relu(self.conv8(x8))
        x9 = nn.functional.relu(self.upconv4(x8))
        x9 = torch.cat([x1, x9], dim=1)
        x9 = nn.functional.relu(self.conv9(x9))
        x10 = self.conv10(x9)
        return x10