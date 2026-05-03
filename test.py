from dataset import CaptchaDataset
from model import CNNModel

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import json

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]
    width = config["resize_width"]

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()])

    test_data_path = config["test_data_path"]
    characters = config["characters"]
    digit_num = config["digit_num"]
    class_num = len(characters) * digit_num
    test_model_path = config["test_model_path"]

    test_data = CaptchaDataset(test_data_path, transform, characters)
    test_load = DataLoader(test_data, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(height, width, digit_num, class_num).to(device)
    model.eval()
    model.load_state_dict(torch.load(test_model_path, map_location=device))

    correct = 0
    total = 0

    for (x, y) in test_load:
        x, y = x.to(device), y.to(device)

        pred = model(x)

        if torch.equal(pred.argmax(dim=2).squeeze(0), y.squeeze(0)):
            correct += 1
        total += 1

    acc = correct / total
    print("Test accuracy: {:.2f}%".format(acc * 100))
