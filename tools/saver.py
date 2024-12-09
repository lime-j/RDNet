import torch
import torch.nn as nn
import os
import time
from tools import mutils

saved_grad = None
saved_name = None

base_url = './results'
os.makedirs(base_url, exist_ok=True)


def normalize_tensor_mm(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def normalize_tensor_sigmoid(tensor):
    return nn.functional.sigmoid(tensor)


def save_image(tensor, name=None, save_path=None, exit_flag=False, timestamp=False, nrow=4, split_dir=None):
    if split_dir:
        _base_url = os.path.join(base_url, split_dir)
    else:
        _base_url = base_url
    os.makedirs(_base_url, exist_ok=True)
    import torchvision.utils as vutils
    grid = vutils.make_grid(tensor.detach().cpu(), nrow=nrow)

    if save_path:
        vutils.save_image(grid, save_path)
    else:
        if timestamp:
            vutils.save_image(grid, f'{_base_url}/{name}_{mutils.get_timestamp()}.png')
        else:
            vutils.save_image(grid, f'{_base_url}/{name}.png')
    if exit_flag:
        exit(0)


def save_feature(tensor, name, exit_flag=False, timestamp=False):
    import torchvision.utils as vutils
    # tensors = [tensor, normalize_tensor_mm(tensor), normalize_tensor_sigmoid(tensor)]
    tensors = [tensor]
    titles = ['original', 'min-max', 'sigmoid']
    if timestamp:
        name += '_' + str(time.time()).replace('.', '')

    for index, tensor in enumerate(tensors):
        _data = tensor.detach().cpu().squeeze(0).unsqueeze(1)
        num_per_row = 4
        if _data.shape[0] / 4 > 4:
            num_per_row = int(_data.shape[0] / 4)
        num_per_row = 8
        grid = vutils.make_grid(_data, nrow=num_per_row)
        vutils.save_image(grid, f'{base_url}/{name}_{titles[index]}.png')
        print(f'{base_url}/{name}_{titles[index]}.png')
    if exit_flag:
        exit(0)


def save(tensor, name, exit_flag=False):
    import torchvision.utils as vutils
    grid = vutils.make_grid(tensor.detach().cpu().squeeze(0).unsqueeze(1), nrow=4)
    # grid = (grid - grid.min()) / (grid.max() - grid.min())
    # print(grid)
    vutils.save_image(grid, f'{base_url}/{name}.png')
    if exit_flag:
        exit(0)


def save_grid_direct(grad, name):
    grad = grad.view(1, 8, 320, 320) * 255 / (320 * 320)
    # grad = grad.view(grad.shape[0],grad)
    save(grad.clamp(0, 255), name)

    module_grad = grad.clamp(-200, 200)
    print(module_grad.min().item(), module_grad.max().item(), module_grad.mean().item())
    module_grad_flat = module_grad.flatten()
    print(name, len(module_grad_flat[module_grad_flat < 0]) / len(module_grad_flat),
          len(module_grad_flat[module_grad_flat < 0]), len(module_grad_flat[module_grad_flat == 0]))
    import matplotlib.pyplot as plt
    import numpy as np
    y, x = np.histogram(module_grad.cpu().flatten().numpy(), bins=50, density=True)
    # plt.hist(module_grad.cpu().flatten().numpy(), 50)
    # for a, b in zip(x[:-1], y):
    #     print(a, b)
    # print(x)
    # print(y)
    plt.bar(x[:-1], y)
    # print('hist', hist)
    # print(module_grad.cpu().flatten().numpy())
    plt.show()


def save_grid(grad, name, exit_flag=False):
    global saved_grad, saved_name
    print(grad.shape)
    if saved_grad is None:
        print(name)
        saved_grad = grad
        saved_name = name
    else:
        # grad_flat = grad.flatten()
        # print(name, len(grad_flat[grad_flat < 0]) / len(grad_flat))

        module_grad = grad / (saved_grad + 1e-7)
        print(module_grad.max())
        save(module_grad.clamp(0, 255) / 255., name)

        module_grad = module_grad.clamp(-300, 300)
        print(module_grad.min().item(), module_grad.max().item(), module_grad.mean().item())
        module_grad_flat = module_grad.flatten()
        print(name, len(module_grad_flat[module_grad_flat < 0]) / len(module_grad_flat),
              len(module_grad_flat[module_grad_flat < 0]), len(module_grad_flat[module_grad_flat == 0]))
        import matplotlib.pyplot as plt
        import numpy as np
        y, x = np.histogram(module_grad.cpu().flatten().numpy(), bins=50, density=True)
        # plt.hist(module_grad.cpu().flatten().numpy(), 50)
        # for a, b in zip(x[:-1], y):
        #     print(a, b)
        # print(x)
        # print(y)
        plt.bar(x[:-1], y)
        # print('hist', hist)
        # print(module_grad.cpu().flatten().numpy())
        plt.show()
        exit(0)
    # print(len(grad))
    # print(grad)
    # print(grad[0].shape)
    # grad = grad[0]
    #
    # grad_flat = grad.flatten()
    # print('--------***')
    # print('--------***')
    # print('--------***')
    # print(name, len(grad_flat[grad_flat < 0]) / len(grad_flat))
    # print('--------***')
    # print('--------***')
    # print('--------***')

    # import torchvision.transforms as vtrans
    # import matplotlib.pyplot as plt
    # plt.hist()
    # plt.imshow(vtrans.ToPILImage()(grid))
    # plt.title(name + ' grad')
    # plt.show()

    #
    # if name in ['HE', 'CE Module', 'SOFT']:
    #     if saved_grad is None:
    #         saved_grad = grad
    #         saved_name = name
    #     else:
    #         grad = grad.reshape_as(saved_grad)
    #         print((saved_grad - grad).mean())
    #         print('diff: ', (saved_grad - grad).abs().max().item())
    #         print('mean: ', name, grad.mean().item(), saved_name, saved_grad.mean().item())
    #
    #         saved_grad = grad
    #         saved_name = name
    if exit_flag:
        exit(0)


def show_grid(grid, name, exit_flag=False):
    import torchvision.utils as vutils
    import torchvision.transforms as vtrans
    import matplotlib.pyplot as plt

    grid = (grid - grid.min()) / (grid.max() - grid.min())
    grid = vutils.make_grid(grid.cpu().squeeze(0).unsqueeze(1), nrow=4)

    # name = unique.get_unique(name)
    plt.imshow(vtrans.ToPILImage()(grid))
    plt.title(name)
    plt.show()
    # vutils.save_image(grid, f'/home/huqiming/workspace/Pytorch_Retinaface/results/{name}.png')
    if exit_flag:
        exit(0)


def show_img(img, name, exit_flag=False):
    import torchvision.utils as vutils
    import torchvision.transforms as vtrans
    import matplotlib.pyplot as plt

    grid = vutils.make_grid(img.cpu().squeeze(0))

    # name = unique.get_unique(name)
    plt.imshow(vtrans.ToPILImage()(grid))
    plt.title(name)
    plt.show()
    # vutils.save_image(grid, f'/home/huqiming/workspace/Pytorch_Retinaface/results/{name}.png')
    if exit_flag:
        exit(0)


class SaverBlock(nn.Module):
    def __init__(self):
        super(SaverBlock, self).__init__()

    def forward(self, x):
        save_feature(x[0], 'intermediate_', timestamp=True)
        return x
