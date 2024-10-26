import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms

# 定义图像预处理函数
img_trans = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.ToTensor()
])

device = torch.device("cpu")  # 确保在 CPU 上运行
model = torchvision.models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# 加载模型的权重
state_dict = torch.load('./models/e31.pt', map_location=device)
model.load_state_dict(state_dict)
model.eval()

def is_nailong(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img_trans(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    outputs = model(img)
    prob = nn.Softmax(dim=1)(outputs)[0]
    is_nailong_ = prob[1].item() > 0.90

    return is_nailong_
