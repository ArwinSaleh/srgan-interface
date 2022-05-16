from tensorlayerx.nn import Module
import tensorlayerx as tlx


class WithLoss_init(Module):
    def __init__(self, G_net, loss_fn):
        super(WithLoss_init, self).__init__()
        self.net = G_net
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        out = self.net(lr)
        loss = self.loss_fn(out, hr)
        return loss


class WithLoss_D(Module):
    def __init__(self, D_net, G_net, loss_fn):
        super(WithLoss_D, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        fake_patchs = self.G_net(lr)
        logits_fake = self.D_net(fake_patchs)
        logits_real = self.D_net(hr)
        d_loss1 = self.loss_fn(logits_real, tlx.ones_like(logits_real))
        d_loss1 = tlx.ops.reduce_mean(d_loss1)
        d_loss2 = self.loss_fn(logits_fake, tlx.zeros_like(logits_fake))
        d_loss2 = tlx.ops.reduce_mean(d_loss2)
        d_loss = d_loss1 + d_loss2
        return d_loss


class WithLoss_G(Module):
    def __init__(self, D_net, G_net, vgg, loss_fn1, loss_fn2):
        super(WithLoss_G, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.vgg = vgg
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

    def forward(self, lr, hr):
        fake_patchs = self.G_net(lr)
        logits_fake = self.D_net(fake_patchs)
        feature_fake = self.vgg((fake_patchs + 1) / 2.)
        feature_real = self.vgg((hr + 1) / 2.)
        g_gan_loss = 1e-3 * self.loss_fn1(logits_fake, tlx.ones_like(logits_fake))
        g_gan_loss = tlx.ops.reduce_mean(g_gan_loss)
        mse_loss = self.loss_fn2(fake_patchs, hr)
        vgg_loss = 2e-6 * self.loss_fn2(feature_fake, feature_real)
        g_loss = mse_loss + vgg_loss + g_gan_loss
        return g_loss