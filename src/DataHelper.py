import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import random

image_size = (224, 224)
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(30),
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

# Download from https://archive.org/download/ukbench/ukbench.zip
class UKBench(Dataset):
    def __init__(self, root_dir="../data/ukbench/", transform=transform, irr_num=10):
        self.root_dir = root_dir
        self.imgs = sorted(os.listdir(self.root_dir))
        self.transform = transform
        self.irr_num = irr_num

        #self.X = [self.getimage(i) for i in range(0, 10200)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        relevant_imgs = self.get_bound(idx)
        possible_irr = [i for i in range(0, 10200) if i not in relevant_imgs]
        relevant_imgs.remove(idx)
        choices = random.choices(possible_irr, k=self.irr_num)
        del possible_irr
        
        xi = self.getimage(idx)
        Xplus = [self.getimage(i) for i in relevant_imgs]
        Xminus = [self.getimage(i) for i in choices]

        return xi, torch.stack(Xplus), torch.stack(Xminus)

    def getimage(self, img_id):
        img = Image.open(self.root_dir + self.imgs[img_id]).convert("RGB")
        return self.transform(img)
    
    def get_bound(self, i):
        i += 1
        if(i % 4 == 0):
            return list(range(i-1, i+4))
        else:
            remainder = i % 4
            return list(range(i - remainder, i + (4 - remainder)))

if __name__ == "__main__":
    dataset = UKBench()

    xi, xp, xm = dataset[12]
    print(xi.shape)
    print(xp.shape)
    print(xm.shape)