import os
os.environ['TL_BACKEND'] = 'tensorflow' # Just modify this line, easily change to any framework! PyTorch will coming soon!
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
import time
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset, DataLoader
from srgan import SRGAN_g, SRGAN_d
from config import config
from tensorlayerx.vision.transforms import Compose, RandomCrop, Normalize, RandomFlipHorizontal, Resize
import vgg
from tensorlayerx.model import TrainOneStep
from tensorlayerx.nn import Module
import cv2

# create folders to save result images and trained models
save_dir = "samples"
tlx.files.exists_or_mkdir(save_dir)
checkpoint_dir = "models"
tlx.files.exists_or_mkdir(checkpoint_dir)

G = SRGAN_g()

###====================== PRE-LOAD DATA ===========================###
valid_hr_imgs = tlx.vision.load_images(path=config.VALID.hr_img_path )
###========================LOAD WEIGHTS ============================###
G.load_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
G.set_eval()
imid = 0  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
valid_hr_img = valid_hr_imgs[imid]
valid_lr_img = np.asarray(valid_hr_img)
hr_size1 = [valid_lr_img.shape[0], valid_lr_img.shape[1]]
valid_lr_img = cv2.resize(valid_lr_img, dsize=(hr_size1[1] // 4, hr_size1[0] // 4))
valid_lr_img_tensor = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]


valid_lr_img_tensor = np.asarray(valid_lr_img_tensor, dtype=np.float32)
valid_lr_img_tensor = valid_lr_img_tensor[np.newaxis, :, :, :]
valid_lr_img_tensor= tlx.ops.convert_to_tensor(valid_lr_img_tensor)
size = [valid_lr_img.shape[0], valid_lr_img.shape[1]]

out = tlx.ops.convert_to_numpy(G(valid_lr_img_tensor))
out = np.asarray((out + 1) * 127.5, dtype=np.uint8)
print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
print("[*] save images")
tlx.vision.save_image(out[0], file_name='valid_gen.png', path=save_dir)
tlx.vision.save_image(valid_lr_img, file_name='valid_lr.png', path=save_dir)
tlx.vision.save_image(valid_hr_img, file_name='valid_hr.png', path=save_dir)
out_bicu = cv2.resize(valid_lr_img, dsize = [size[1] * 4, size[0] * 4], interpolation = cv2.INTER_CUBIC)
tlx.vision.save_image(out_bicu, file_name='valid_hr_cubic.png', path=save_dir)