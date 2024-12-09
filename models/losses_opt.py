import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM, SSIM

from models.vgg import Vgg19


###############################################################################
# Functions
###############################################################################
def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class ContainLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(ContainLoss, self).__init__()
        self.eps = eps

    def forward(self, predict_t, predict_r, input_image):
        pix_num = np.prod(input_image.shape)
        predict_tx, predict_ty = compute_gradient(predict_t)
        predict_rx, predict_ry = compute_gradient(predict_r)
        input_x, input_y = compute_gradient(input_image)

        out = torch.norm(predict_tx / (input_x + self.eps), 2) ** 2 + \
              torch.norm(predict_ty / (input_y + self.eps), 2) ** 2 + \
              torch.norm(predict_rx / (input_x + self.eps), 2) ** 2 + \
              torch.norm(predict_ry / (input_y + self.eps), 2) ** 2

        return out / pix_num


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


def l1_norm_dim(x, dim):
    return torch.mean(torch.abs(x), dim=dim)


def l1_norm(x):
    return torch.mean(torch.abs(x))


def l2_norm(x):
    return torch.mean(torch.square(x))


def gradient_norm_kernel(x, kernel_size=10):
    out_h, out_v = compute_gradient(x)
    shape = out_h.shape
    out_h = F.unfold(out_h, kernel_size=(kernel_size, kernel_size), stride=(1, 1))
    out_h = out_h.reshape(shape[0], shape[1], kernel_size * kernel_size, -1)
    out_h = l1_norm_dim(out_h, 2)
    out_v = F.unfold(out_v, kernel_size=(kernel_size, kernel_size), stride=(1, 1))
    out_v = out_v.reshape(shape[0], shape[1], kernel_size * kernel_size, -1)
    out_v = l1_norm_dim(out_v, 2)
    return out_h, out_v


class KTVLoss(nn.Module):
    def __init__(self, kernel_size=10):
        super().__init__()
        self.kernel_size = kernel_size
        self.criterion = nn.L1Loss()
        self.eps = 1e-6

    def forward(self, out_l, out_r, input_i):
        out_l_normx, out_l_normy = gradient_norm_kernel(out_l, self.kernel_size)
        out_r_normx, out_r_normy = gradient_norm_kernel(out_r, self.kernel_size)
        input_normx, input_normy = gradient_norm_kernel(input_i, self.kernel_size)
        norm_l = out_l_normx + out_l_normy
        norm_r = out_r_normx + out_r_normy
        norm_target = input_normx + input_normy + self.eps
        norm_loss = (norm_l / norm_target + norm_r / norm_target).mean()

        out_lx, out_ly = compute_gradient(out_l)
        out_rx, out_ry = compute_gradient(out_r)
        input_x, input_y = compute_gradient(input_i)
        gradient_diffx = self.criterion(out_lx + out_rx, input_x)
        gradient_diffy = self.criterion(out_ly + out_ry, input_y)
        grad_loss = gradient_diffx + gradient_diffy

        loss = norm_loss * 1e-4 + grad_loss
        return loss


class MTVLoss(nn.Module):
    def __init__(self, kernel_size=10):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.norm = l1_norm

    def forward(self, out_l, out_r, input_i):
        out_lx, out_ly = compute_gradient(out_l)
        out_rx, out_ry = compute_gradient(out_r)
        input_x, input_y = compute_gradient(input_i)

        norm_l = self.norm(out_lx) + self.norm(out_ly)
        norm_r = self.norm(out_rx) + self.norm(out_ry)
        norm_target = self.norm(input_x) + self.norm(input_y)

        gradient_diffx = self.criterion(out_lx + out_rx, input_x)
        gradient_diffy = self.criterion(out_ly + out_ry, input_y)

        loss = (norm_l / norm_target + norm_r / norm_target) * 1e-5 + gradient_diffx + gradient_diffy

        return loss


class ReconsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.norm = l1_norm

    def forward(self, out_l, out_r, input_i):
        content_diff = self.criterion(out_l + out_r, input_i)
        out_lx, out_ly = compute_gradient(out_l)
        out_rx, out_ry = compute_gradient(out_r)
        input_x, input_y = compute_gradient(input_i)

        gradient_diffx = self.criterion(out_lx + out_rx, input_x)
        gradient_diffy = self.criterion(out_ly + out_ry, input_y)

        loss = content_diff + (gradient_diffx + gradient_diffy) * 0.5

        return loss


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()  # absorb sigmoid into BCELoss

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                target_tensor = self.get_target_tensor(input_i, target_is_real)
                loss += self.loss(input_i, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


class DiscLoss():
    def name(self):
        return 'SGAN'

    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA=None, fakeB=None, realB=None):
        pred_fake = None
        pred_real = None
        loss_D_fake = 0
        loss_D_real = 0
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero

        if fakeB is not None:
            pred_fake = net.forward(fakeB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, 0)

        # Real
        if realB is not None:
            pred_real = net.forward(realB)
            loss_D_real = self.criterionGAN(pred_real, 1)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, pred_fake, pred_real


class DiscLossR(DiscLoss):
    # RSGAN from
    # https://arxiv.org/abs/1807.00734
    def name(self):
        return 'RSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake - pred_real, 1)

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())

        loss_D = self.criterionGAN(pred_real - pred_fake, 1)  # BCE_stable loss
        return loss_D, pred_fake, pred_real


class DiscLossRa(DiscLoss):
    # RaSGAN from
    # https://arxiv.org/abs/1807.00734
    def name(self):
        return 'RaSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)

        loss_G = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 0)
        loss_G += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 1)
        return loss_G * 0.5

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())

        loss_D = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 1)
        loss_D += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 0)
        return loss_D * 0.5, pred_fake, pred_real


class MS_SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1, size_average=True, channel=3)

    def forward(self, output, target):
        return 1 - self.ms_ssim(output, target)


class SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1, size_average=True, channel=3)

    def forward(self, output, target):
        return 1 - self.ssim(output, target)


def init_loss(opt, tensor):
    disc_loss = None
    content_loss = None

    loss_dic = {}

    t_pixel_loss = ContentLoss()
    t_pixel_loss.initialize(
        MultipleLoss([nn.MSELoss(), MS_SSIM_Loss(), GradientLoss()], [1.0, 0.4, 0.6]))

    loss_dic['t_pixel'] = t_pixel_loss

    r_pixel_loss = ContentLoss()
    r_pixel_loss.initialize(
        MultipleLoss([nn.MSELoss()], [4.0]))

    loss_dic['r_pixel'] = r_pixel_loss
    loss_dic['recons'] = ReconsLoss()

    loss_dic['mtv'] = MTVLoss()
    loss_dic['ktv'] = KTVLoss()

    if opt.lambda_gan > 0:
        if opt.gan_type == 'sgan' or opt.gan_type == 'gan':
            disc_loss = DiscLoss()
        elif opt.gan_type == 'rsgan':
            disc_loss = DiscLossR()
        elif opt.gan_type == 'rasgan':
            disc_loss = DiscLossRa()
        else:
            raise ValueError("GAN [%s] not recognized." % opt.gan_type)

        disc_loss.initialize(opt, tensor)
        loss_dic['gan'] = disc_loss

    return loss_dic


if __name__ == '__main__':
    x = torch.randn(3, 32, 224, 224).cuda()
    import time

    s = time.time()
    out1, out2 = gradient_norm_kernel(x)
    t = time.time()
    print(t - s)
    print(out1.shape, out2.shape)
