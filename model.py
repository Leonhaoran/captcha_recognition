import json

import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, height, width, digit_num, class_num):
        super(CNNModel, self).__init__()
        self.digit_num = digit_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        input_num = (height // 8) * (width // 8) * 64

        self.fc1 = nn.Sequential(
            nn.Linear(input_num, 1024),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, class_num)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        x = x.view(x.size(0), self.digit_num, -1)
        return x


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]
    width = config["resize_width"]
    characters = config["characters"]
    digit_num = config["digit_num"]
    class_num = len(characters) * digit_num

    model = CNNModel(height, width, digit_num, class_num)
    print(model)
