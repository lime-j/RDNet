from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--eval_freq', type=int, default=1, help='frequency of evaluation')
        self.parser.add_argument('--save_freq', type=int, default=1, help='frequency of save eval samples')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--debug', action='store_true',
                                 help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--finetune', action='store_true',
                                 help='finetune the network using identity inputs and outputs')
        self.parser.add_argument('--if_align', action='store_true',
                                 help='if align 4x')

        # self.parser.add_argument('--graph', action='store_true',
        #                          help='print computation graph')
        # for training (Note: in train_sirs.py, we mannually tune the training protocol, but you can also use following setting by modifying the code in errnet_model.py)
        self.parser.add_argument('--nEpochs', '-n', type=int, default=60, help='# of epochs to run')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')

        self.parser.add_argument('--r_pixel_weight', '-rw', type=float, default=1.0, help='weight for r_pixel loss')

        self.parser.add_argument('--low_sigma', type=float, default=2, help='min sigma in synthetic dataset')
        self.parser.add_argument('--high_sigma', type=float, default=5, help='max sigma in synthetic dataset')
        self.parser.add_argument('--low_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
        self.parser.add_argument('--high_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')

        # data augmentation
        self.parser.add_argument('--real20_size', type=int, default=420, help='scale images to compat size')
        self.parser.add_argument('--batchSize', '-b', type=int, default=2, help='input batch size')
        self.parser.add_argument('--loadSize', type=str, default='224,336,448', help='scale images to multiple size')
        self.parser.add_argument('--fineSize', type=str, default='224,224', help='then crop to this size')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--debug_eval', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--graph', action='store_true', help='print graph')

        # for discriminator
        self.parser.add_argument('--which_model_D', type=str, default='disc_vgg', choices=['disc_vgg', 'disc_patch'])
        self.parser.add_argument('--gan_type', type=str, default='rasgan',
                                 help='gan/sgan : Vanilla GAN; rasgan : relativistic gan')
        # loss weight
        self.parser.add_argument('--unaligned_loss', type=str, default='vgg',
                                 help='learning rate policy: vgg|mse|ctx|ctx_vgg')
        self.parser.add_argument('--tv_type', type=str, default=None, choices=['ktv', 'mtv'])
        self.parser.add_argument('--vgg_layer', type=int, default=31, help='vgg layer of unaligned loss')
        self.parser.add_argument('--init_lr', type=float, default=1e-2, help='initial learning rate')
        self.parser.add_argument('--fixed_lr', type=float, default=0, help='initial learning rate')
        self.parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for gan loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.1, help='weight for vgg loss')
        self.parser.add_argument('--weight_loss',type=float,default=0.25,help='weight fot overall loss')
        self.parser.add_argument('--num_subnet',type=int,default=4,help='num_number of subnet')
        self.parser.add_argument('--dataset',type=float,default=0.5,help='the setting of dataset')
        self.parser.add_argument('--loss_col',type=int,default=4,help='numcol for loss')
        self.parser.add_argument('--drop_path',type=float,default=0.6,help='drop_path')

        

        self.isTrain = True
