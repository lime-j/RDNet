import os
from os.path import join

import torch.backends.cudnn as cudnn

# import data.sirs_dataset as datasets
import data.dataset_sir as datasets
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils

opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False
datadir = os.path.join(os.path.expanduser('~'), '/opt/datasets/sirs')

eval_dataset_real = datasets.DSRTestDataset(join(datadir, f'test/real20_{opt.real20_size}'),
                                            fns=read_fns('data/real_test.txt'), if_align=opt.if_align)
eval_dataset_solidobject = datasets.DSRTestDataset(join(datadir, 'test/SIR2/SolidObjectDataset'),
                                                   if_align=opt.if_align)
eval_dataset_postcard = datasets.DSRTestDataset(join(datadir, 'test/SIR2/PostcardDataset'), if_align=opt.if_align)
eval_dataset_wild = datasets.DSRTestDataset(join(datadir, 'test/SIR2/WildSceneDataset'), if_align=opt.if_align)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=True,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_wild = datasets.DataLoader(
    eval_dataset_wild, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

engine = Engine(opt, eval_dataset_real, eval_dataset_solidobject, eval_dataset_postcard, eval_dataloader_wild)

"""Main Loop"""
result_dir = os.path.join('./results', opt.name, mutils.get_formatted_time())

res1 = engine.eval(eval_dataloader_real, dataset_name='testdata_real',
                  savedir=join(result_dir, 'real20'), suffix='real20')

res2 = engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject',
                  savedir=join(result_dir, 'solidobject'), suffix='solidobject')
res3 = engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard',
                  savedir=join(result_dir, 'postcard'), suffix='postcard')

res4 = engine.eval(eval_dataloader_wild, dataset_name='testdata_wild',
                  savedir=join(result_dir, 'wild'), suffix='wild')

