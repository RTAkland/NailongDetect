import os
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torch import Tensor, nn
from torchvision import transforms

from nl import is_nailong

api = FastAPI()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)],
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.max1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.max2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.max3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.fla = nn.Flatten()
        self.lin1 = nn.Linear(64 * 4 * 4, 64)
        self.drop = nn.Dropout(0.25)
        self.lin2 = nn.Linear(64, 11)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fla(x)
        x = self.lin1(x)
        x = self.drop(x)
        return self.lin2(x)


model_filename = "nailong(0.5122).pth"
model = Net()
if Path("./models").exists():
    model.load_state_dict(
        torch.load(
            Path("./models") / model_filename,
            weights_only=True,
            map_location="cpu",
        ),
    )


def check_image(image: np.ndarray):
    """
    :param image: OpenCV图像数组。
    :return: 如果图像中有奶龙，返回True；否则返回False。
    """
    image = cv2.resize(image, (32, 32))
    image = transform(image)
    image = image.unsqueeze(0)  # type: ignore
    output = model(image)
    return output.argmax(1) == 10


class Payload(BaseModel):
    url: str


@api.post("/nailong")
async def contains_nailong(payload: Payload):
    filename = payload.url.split("/")[-1]
    with open(filename, "wb") as f:
        f.write(requests.get(payload.url).content)
    img = cv2.imread(filename)
    result = check_image(img)
    os.remove(filename)
    return {"result": bool(result[0])}

@api.post("/nailong2")
async def contains_nailong2(payload: Payload):
    filename = payload.url.split("/")[-1]
    with open(filename, "wb") as f:
        f.write(requests.get(payload.url).content)
    result = is_nailong(filename)
    os.remove(filename)
    return {"result": result}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)
