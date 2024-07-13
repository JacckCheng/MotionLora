import os
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

class Modified(Dataset):
    def __init__(self, folder_path):
        self.data = []
        self.folder_path = folder_path

        # 归一化训练数据
        self.clip_img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        for filename in os.listdir(folder_path):
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                file_path = os.path.join(folder_path, filename)
                image = Image.open(file_path).convert('RGB')  # Convert to RGB to ensure consistency
                # self.data.append(self.clip_img_transforms(image))
                self.data.append(image)

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    # from animatediff.utils.util import save_videos_grid  # 确保这个导入是必要的

    dataset = Modified("/home/chengtianle/MotionLora/data/inputs/resized_images")
    
    # 暂时这样设置，仅使用主进程，且batch_size设为2
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)  # 调整 num_workers，根据你的系统资源

    for idx, batch in enumerate(dataloader):
        print(batch.shape)  # batch 是一个 Tensor，形状为 (batch_size, 3, 256, 256)
