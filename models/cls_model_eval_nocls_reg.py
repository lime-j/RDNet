import torch
from torch import nn
import torch.nn.functional as F
from models.losses import DINOLoss
import os
import numpy as np
from collections import OrderedDict
from ema_pytorch import EMA
from models.arch.classifier import PretrainedConvNext
import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch
#from models.arch.dncnn import effnetv2_s
from .base_model import BaseModel
from PIL import Image
from os.path import join
#from torchviz import make_dot
from models.arch.RDnet_ import FullNet_NLP
import timm

load_dict_large='/home/xteam/zhaohao/pycharmproject/YTMT/data/naf_revcol_tiny.pth'
def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)

        gradx = (img[..., 1:, :] - img[..., :-1, :]).abs().sum(dim=1, keepdim=True)
        grady = (img[..., 1:] - img[..., :-1]).abs().sum(dim=1, keepdim=True)

        gradX[..., :-1, :] += gradx
        gradX[..., 1:, :] += gradx
        gradX[..., 1:-1, :] /= 2

        gradY[..., :-1] += grady
        gradY[..., 1:] += grady
        gradY[..., 1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class YTMTNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        identity = False
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)

        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])

        self.input = input
        self.identity = identity
        self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.target_r = target_r
        self.data_name = data_name

        self.issyn = False if 'real' in data else True
        self.aligned = False if 'unaligned' in data else True

        if target_t is not None:
            self.target_edge = self.edge_map(self.target_t)

    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        self._eval()
        self.set_input(data, 'eval')
        with torch.no_grad():
            self.forward_eval()

            output_i = tensor2im(self.output_j[6])
            output_j = tensor2im(self.output_j[7])
            target = tensor2im(self.target_t)
            target_r = tensor2im(self.target_r)

            if self.aligned:
                res = index.quality_assess(output_i, target)
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    savedir = join(savedir, suffix, name)
                    os.makedirs(savedir, exist_ok=True)
                    Image.fromarray(output_i.astype(np.uint8)).save(
                        join(savedir, '{}_t.png'.format(self.opt.name)))
                    Image.fromarray(output_j.astype(np.uint8)).save(
                        join(savedir, '{}_r.png'.format(self.opt.name)))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 't_label.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'm_input.png'))
                else:
                    if not os.path.exists(join(savedir, 'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(
                        join(savedir, 'transmission_layer', str(self._count) + '.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(
                        join(savedir, 'blended', str(self._count) + '.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))

            if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                return

        with torch.no_grad():
            output_i, output_j = self.forward()
            output_i = tensor2im(output_i)
            output_j = tensor2im(output_j)
            if self.data_name is not None and savedir is not None:
                Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}_l.png'.format(self.opt.name)))
                Image.fromarray(output_j.astype(np.uint8)).save(join(savedir, name, '{}_r.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))


class ClsModel(YTMTNetBase):
    def name(self):
        return 'ytmtnet'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_c = None

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()
        self.net_c.eval()

    def _train(self):
        self.net_i.train()
        self.net_c.eval()
    def initialize(self, opt):
        self.opt=opt
        BaseModel.initialize(self, opt)

        in_channels = 3
        self.vgg = None

        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472
        channels = [64, 128, 256, 512]
        layers = [2, 2, 4, 2]
        num_subnet = opt.num_subnet
        self.net_c = PretrainedConvNext("convnext_small_in22k").cuda()
        
        self.net_c.load_state_dict(torch.load('/home/xteam/zhaohao/pycharmproject/YTMT/cls_newdis_058_00014384.pt')['icnn'])

        self.net_i = FullNet_NLP(channels, layers, num_subnet, opt.loss_col,num_classes=1000, drop_path=0,save_memory=True, inter_supv=True, head_init_scale=None,kernel_size=3).to(self.device)
    
        self.edge_map = EdgeMap(scale=1).to(self.device)
    
        if self.isTrain:
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1, 0.1, 0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1, 0.1, 0.1, 0.1], indices=[8, 13, 22, 31],
                                                criterions=[losses.CX_loss] * 3 + [nn.L1Loss()]))
            else:
                raise NotImplementedError
            self.scaler=torch.cuda.amp.GradScaler()
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                self.dinoloss=DINOLoss()
            self.loss_dic['t_cx'] = cxloss
        
            self.optimizer_G = torch.optim.Adam(self.net_i.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)


            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)


    def backward_D(self):
        loss_D=[]
        weight=self.opt.weight_loss
        for p in self.netD.parameters():
            p.requires_grad = True
        for i in range(4):
            loss_D_1, pred_fake_1, pred_real_1 = self.loss_dic['gan'].get_loss(
                self.netD, self.input, self.output_j[2*i], self.target_t)
            loss_D.append(loss_D_1*weight)
            weight+=self.opt.weight_loss
        loss_sum=sum(loss_D)

        self.loss_D, self.pred_fake, self.pred_real = (loss_sum, pred_fake_1, pred_real_1)

        (self.loss_D * self.opt.lambda_gan).backward(retain_graph=True)

    def get_loss(self, out_l, out_r):
        loss_G_GAN_sum=[]
        loss_icnn_pixel_sum=[]
        loss_rcnn_pixel_sum=[]
        loss_icnn_vgg_sum=[]
        weight=self.opt.weight_loss
        for i in range(self.opt.loss_col):
            out_r_clean=out_r[2*i]
            out_r_reflection=out_r[2*i+1]
            if i != self.opt.loss_col -1:
                loss_G_GAN = 0
                loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(out_r_clean, self.target_t)
                loss_rcnn_pixel = self.loss_dic['r_pixel'].get_loss(out_r_reflection, self.target_r) * 1.5 * self.opt.r_pixel_weight
                loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(out_r_clean, self.target_t) * self.opt.lambda_vgg
            else:
                if self.opt.lambda_gan>0:

                    loss_G_GAN=0
                else:
                    loss_G_GAN=0
                loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(out_r_clean, self.target_t)
                loss_rcnn_pixel = self.loss_dic['r_pixel'].get_loss(out_r_reflection, self.target_r) * 1.5 * self.opt.r_pixel_weight
                loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(out_r_clean, self.target_t) * self.opt.lambda_vgg

            loss_G_GAN_sum.append(loss_G_GAN*weight)
            loss_icnn_pixel_sum.append(loss_icnn_pixel*weight)
            loss_rcnn_pixel_sum.append(loss_rcnn_pixel*weight)
            loss_icnn_vgg_sum.append(loss_icnn_vgg*weight)
            weight=weight+self.opt.weight_loss
        return sum(loss_G_GAN_sum), sum(loss_icnn_pixel_sum), sum(loss_rcnn_pixel_sum), sum(loss_icnn_vgg_sum)

    def backward_G(self):

        self.loss_G_GAN,self.loss_icnn_pixel, self.loss_rcnn_pixel, \
        self.loss_icnn_vgg = self.get_loss(self.output_i, self.output_j)

        self.loss_exclu = self.exclusion_loss(self.output_i, self.output_j, 3)

        self.loss_recons = self.loss_dic['recons'](self.output_i, self.output_j, self.input) * 0.2

        self.loss_G =  self.loss_G_GAN +self.loss_icnn_pixel + self.loss_rcnn_pixel + \
                      self.loss_icnn_vgg
        self.scaler.scale(self.loss_G).backward()



    def hyper_column(self, input_img):
        hypercolumn = self.vgg(input_img)
        _, C, H, W = input_img.shape
        hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for
                       feature in hypercolumn]
        input_i = [input_img]
        input_i.extend(hypercolumn)
        input_i = torch.cat(input_i, dim=1)
        return input_i

    def forward(self):
        # without edge
        
        self.output_j=[]
        input_i = self.input
        if self.vgg is not None:
            input_i = self.hyper_column(input_i)
        with torch.no_grad():
            ipt = self.net_c(input_i)
        output_i, output_j = self.net_i(input_i,ipt,prompt=True)
        self.output_i = output_i
        for i in range(self.opt.loss_col):
            out_reflection, out_clean = output_j[i][:, :3, ...], output_j[i][:, 3:, ...]
            self.output_j.append(out_clean) 
            self.output_j.append(out_reflection) 
        return self.output_i, self.output_j


    @torch.no_grad() 
    def forward_eval(self):
       
        self.output_j=[]
        input_i = self.input
        if self.vgg is not None:
            input_i = self.hyper_column(input_i)
        ipt = self.net_c(input_i)
        
        output_i, output_j = self.net_i(input_i,ipt,prompt=True)
        self.output_i = output_i #alpha * output_i + beta
        for i in range(self.opt.loss_col):
            out_reflection, out_clean = output_j[i][:, :3, ...], output_j[i][:, 3:, ...]
            self.output_j.append(out_clean) 
            self.output_j.append(out_reflection)
        return self.output_i, self.output_j

    def optimize_parameters(self):
        self._train()
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def return_output(self):
        output_clean = self.output_j[1]
        output_reflection = self.output_j[0]
        output_clean = tensor2im(output_clean).astype(np.uint8)
        output_reflection = tensor2im(output_reflection).astype(np.uint8)
        input=tensor2im(self.input)
        return output_clean,output_reflection,input
    def exclusion_loss(self, img_T, img_R, level=3, eps=1e-6):
        loss_gra=[]
        weight=0.25
        for i in range(4):
            grad_x_loss = []
            grad_y_loss = []
            img_T=self.output_j[2*i]
            img_R=self.output_j[2*i+1]
            for l in range(level):
                grad_x_T, grad_y_T = self.compute_grad(img_T)
                grad_x_R, grad_y_R = self.compute_grad(img_R)

                alphax = (2.0 * torch.mean(torch.abs(grad_x_T))) / (torch.mean(torch.abs(grad_x_R)) + eps)
                alphay = (2.0 * torch.mean(torch.abs(grad_y_T))) / (torch.mean(torch.abs(grad_y_R)) + eps)

                gradx1_s = (torch.sigmoid(grad_x_T) * 2) - 1  # mul 2 minus 1 is to change sigmoid into tanh
                grady1_s = (torch.sigmoid(grad_y_T) * 2) - 1
                gradx2_s = (torch.sigmoid(grad_x_R * alphax) * 2) - 1
                grady2_s = (torch.sigmoid(grad_y_R * alphay) * 2) - 1

                grad_x_loss.append(((torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2)))) + eps) ** 0.25)
                grad_y_loss.append(((torch.mean(torch.mul(grady1_s.pow(2), grady2_s.pow(2)))) + eps) ** 0.25)

                img_T = F.interpolate(img_T, scale_factor=0.5, mode='bilinear')
                img_R = F.interpolate(img_R, scale_factor=0.5, mode='bilinear')
            loss_gradxy = torch.sum(sum(grad_x_loss) / 3) + torch.sum(sum(grad_y_loss) / 3)
            loss_gra.append(loss_gradxy*weight)
            weight+=0.25


        return sum(loss_gra) / 2

    def contain_loss(self, img_T, img_R, img_I, eps=1e-6):
        pix_num = np.prod(img_I.shape)
        predict_tx, predict_ty = self.compute_grad(img_T)
        predict_tx, predict_ty = self.compute_grad(img_T)
        predict_rx, predict_ry = self.compute_grad(img_R)
        input_x, input_y = self.compute_grad(img_I)

        out = torch.norm(predict_tx / (input_x + eps), 2) ** 2 + \
              torch.norm(predict_ty / (input_y + eps), 2) ** 2 + \
              torch.norm(predict_rx / (input_x + eps), 2) ** 2 + \
              torch.norm(predict_ry / (input_y + eps), 2) ** 2

        return out / pix_num

    def compute_grad(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

    def load(self, model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = torch.load(icnn_path)
        model.net_i.load_state_dict(state_dict['icnn'])
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(),
            #'ema' : self.ema.state_dict(),
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)
