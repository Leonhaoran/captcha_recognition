from dataset import CaptchaDataset
from model import CNNModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import json
import os

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]
    width = config["resize_width"]

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])

    train_data_path = config["train_data_path"]
    characters = config["characters"]
    batch_size = config["batch_size"]
    epoch_num = config["epoch_num"]
    digit_num = config["digit_num"]
    learning_rate = config["learning_rate"]

    class_num = len(characters) * digit_num

    model_save_path = config["model_save_path"]
    model_name = config["model_name"]
    model_save_name = model_save_path + "/" + model_name

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_data = CaptchaDataset(train_data_path, transform, characters)
    train_load = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(height, width, digit_num, class_num).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Begin training...")
    for epoch in range(epoch_num):
        for batch_idx, (data, label) in enumerate(train_load):
            data, label = data.to(device), label.to(device)

            output = model(data)

            loss = torch.tensor(0.0).to(device)
            for i in range(digit_num):
                loss += criterion(output[:, i, :], label[:, i])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predicted = torch.argmax(output, dim=2)
            correct = (predicted == label).all(dim=1).sum().item()
            acc = correct / data.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{epoch_num}"
                      f"| Batch {batch_idx}/{len(train_load)}"
                      f"| Loss {loss.item():.4f}"
                      f"| Acc {acc:.4f}")

            if (epoch + 1) % 10 == 0:
                checkpoint = model_save_path + "check.epoch" + str(epoch + 1) + ".pth"
                torch.save(model.state_dict(), checkpoint)
                print(f"checkpoint saved at {checkpoint}")

    torch.save(model.state_dict(), model_save_name)
    print(f"model saved at {model_save_name}")
