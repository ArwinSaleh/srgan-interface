from tensorlayerx.vision.transforms import Compose, RandomCrop, RandomFlipHorizontal, Normalize, Resize
from tensorlayerx.dataflow import Dataset
import tensorlayerx as tlx
from config import config


hr_transform = Compose([
    RandomCrop(size=(384, 384)),
    RandomFlipHorizontal(),
])
nor = Normalize(mean=(127.5), std=(127.5), data_format='HWC')
lr_transform = Resize(size=(96, 96))


class TrainData(Dataset):

    def __init__(self, hr_trans=hr_transform, lr_trans=lr_transform):
        self.train_hr_imgs = tlx.vision.load_images(path=config.TRAIN.hr_img_path)
        self.hr_trans = hr_trans
        self.lr_trans = lr_trans

    def __getitem__(self, index):
        img = self.train_hr_imgs[index]
        hr_patch = self.hr_trans(img)
        lr_patch = self.lr_trans(hr_patch)
        return nor(lr_patch), nor(hr_patch)

    def __len__(self):
        return len(self.train_hr_imgs)