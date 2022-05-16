import os
os.environ['TL_BACKEND'] = 'tensorflow' # Just modify this line, easily change to any framework! PyTorch will coming soon!
import time
import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from srgan import SRGAN_g, SRGAN_d
from config import config
import vgg
from tensorlayerx.model import TrainOneStep
from data import TrainData
from loss import WithLoss_D, WithLoss_G, WithLoss_init

###====================== HYPER-PARAMETERS ===========================###
batch_size = 8
n_epoch_init = config.TRAIN.n_epoch_init
n_epoch = config.TRAIN.n_epoch
# create folders to save result images and trained models
save_dir = "samples"
tlx.files.exists_or_mkdir(save_dir)
checkpoint_dir = "models"
tlx.files.exists_or_mkdir(checkpoint_dir)


G = SRGAN_g()
D = SRGAN_d()
VGG = vgg.VGG19(pretrained=False, end_with='pool4', mode='dynamic')
# automatic init layers weights shape with input tensor.
# Calculating and filling 'in_channels' of each layer is a very troublesome thing.
# So, just use 'init_build' with input shape. 'in_channels' of each layer will be automaticlly set.
G.init_build(tlx.nn.Input(shape=(8, 96, 96, 3)))
D.init_build(tlx.nn.Input(shape=(8, 384, 384, 3)))

G.set_train()
D.set_train()
VGG.set_eval()
train_ds = TrainData()
train_ds_img_nums = len(train_ds)
train_ds = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

lr_v = tlx.optimizers.lr.StepDecay(learning_rate=0.05, step_size=1000, gamma=0.1, last_epoch=-1, verbose=True)
g_optimizer_init = tlx.optimizers.Momentum(lr_v, 0.9)
g_optimizer = tlx.optimizers.Momentum(lr_v, 0.9)
d_optimizer = tlx.optimizers.Momentum(lr_v, 0.9)
g_weights = G.trainable_weights
d_weights = D.trainable_weights
net_with_loss_init = WithLoss_init(G, loss_fn=tlx.losses.mean_squared_error)
net_with_loss_D = WithLoss_D(D_net=D, G_net=G, loss_fn=tlx.losses.sigmoid_cross_entropy)
net_with_loss_G = WithLoss_G(D_net=D, G_net=G, vgg=VGG, loss_fn1=tlx.losses.sigmoid_cross_entropy,
                                loss_fn2=tlx.losses.mean_squared_error)

trainforinit = TrainOneStep(net_with_loss_init, optimizer=g_optimizer_init, train_weights=g_weights)
trainforG = TrainOneStep(net_with_loss_G, optimizer=g_optimizer, train_weights=g_weights)
trainforD = TrainOneStep(net_with_loss_D, optimizer=d_optimizer, train_weights=d_weights)

# initialize learning (G)
n_step_epoch = round(train_ds_img_nums // batch_size)
for epoch in range(n_epoch_init):
    for step, (lr_patch, hr_patch) in enumerate(train_ds):
        step_time = time.time()
        loss = trainforinit(lr_patch, hr_patch)
        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
            epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, float(loss)))

# adversarial learning (G, D)
n_step_epoch = round(train_ds_img_nums // batch_size)
for epoch in range(n_epoch):
    for step, (lr_patch, hr_patch) in enumerate(train_ds):
        step_time = time.time()
        loss_g = trainforG(lr_patch, hr_patch)
        loss_d = trainforD(lr_patch, hr_patch)
        print(
            "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f}, d_loss: {:.3f}".format(
                epoch, n_epoch, step, n_step_epoch, time.time() - step_time, float(loss_g), float(loss_d)))
    # dynamic learning rate update
    lr_v.step()

    if (epoch != 0) and (epoch % 10 == 0):
        G.save_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
        D.save_weights(os.path.join(checkpoint_dir, 'd.npz'), format='npz_dict')