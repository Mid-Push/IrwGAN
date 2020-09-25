import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import loss
from util.util import to_data
import os

class IRWGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=1.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_irw_A', type=float, default=1.0, help='weight for controlling the sparsity of beta')
            parser.add_argument('--lambda_irw_B', type=float, default=1.0, help='weight for controlling the sparsity of beta')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'irw_A', 'irw_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'cycle_A', 'idt_A']
        visual_names_B = ['real_B', 'fake_A', 'cycle_B', 'idt_B']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['gen_a2b', 'gen_b2a', 'dis_a', 'dis_b', 'beta_net_a', 'beta_net_b']
            self.opt_names = ['optimizer_G', 'optimizer_D', 'optimizer_B']
        else:  # during test time, only load Gs
            self.model_names = ['gen_a2b', 'gen_b2a']
            self.opt_names = []

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.gen_a2b = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.initG, opt.init_gain, self.gpu_ids)
        self.gen_b2a = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.initG, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.dis_a = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.sn, opt.initD, opt.init_gain, self.gpu_ids)
            self.dis_b = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.sn, opt.initD, opt.init_gain, self.gpu_ids)
            self.beta_net_a = networks.define_BetaNet(opt.input_nc, opt.ndf, 4, opt.normG, opt.initG, opt.init_gain, self.gpu_ids, is_use=('A' in opt.beta_mode))
            self.beta_net_b = networks.define_BetaNet(opt.input_nc, opt.ndf, 4, opt.normG, opt.initG, opt.init_gain, self.gpu_ids, is_use=('B' in opt.beta_mode))

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # define loss functions, ignore samples that have beta<threshold
            self.gan_criterion_a2b = loss.get_gan_loss(opt.gan_type)(self.dis_b, opt.threshold)
            self.gan_criterion_b2a = loss.get_gan_loss(opt.gan_type)(self.dis_a, opt.threshold)
            self.criterionCycle = loss.IRW_L1_Loss(opt.threshold)
            self.criterionIdt = loss.IRW_L1_Loss(opt.threshold)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.gen_a2b.parameters(), self.gen_b2a.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.dis_a.parameters(), self.dis_b.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_B = torch.optim.Adam(itertools.chain(self.beta_net_a.parameters(), self.beta_net_b.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_B)

    def print_information(self, opt):
        print('#### Information ####')
        print('# task: %s' % opt.task)
        print('# model_dir: %s' % opt.model_dir)
        print('# dataset_size: %d' % opt.dataset_size)
        print('# gan_type: %s' % opt.gan_type)
        print()
        print('#### Weight ####')
        print('# lambda_A: %.f' % opt.lambda_A)
        print('# lambda_B: %.f' % opt.lambda_B)
        print('# lambda_identity: %.f' % opt.lambda_identity)
        print()
        print('#### Generator #####')
        print('# netG: %s' % opt.netG)
        print('# initG: %s' % opt.initG)
        print('# normG: %s' % opt.normG)
        print()
        print('#### Discriminator ####')
        print('# netD: %s' % opt.netD)
        print('# initD: %s' % opt.initD)
        print('# normD: %s' % opt.normD)
        print('# sn: %s' % opt.sn)
        print()
        print('#### Model Specific ####')
        print('# beta_mode: %s' % opt.beta_mode)
        print('# threshold: %s' % opt.threshold)
        print('# batch_size: %d' % opt.batch_size)
        print('# lambda_irw_A: %.f' % opt.lambda_irw_A)
        print('# lambda_irw_B: %.f' % opt.lambda_irw_B)

    @property
    def model_dir(self):
        opt = self.opt
        sn = '_sn' if self.opt.sn else ''
        thr = 'thr{}'.format(opt.threshold)
        self.opt.task = self.opt.dataroot.strip('/').split('/')[-1]
        model_dir_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}{}".format(self.opt.task, self.opt.gan_type,
                                    self.opt.batch_size, self.opt.beta_mode,
                                    self.opt.lambda_irw_A, self.opt.lambda_irw_B, thr,
                                    self.opt.netD, opt.normD, sn)
        return os.path.join(opt.checkpoints_dir, model_dir_name)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_As = input['A'].to(self.device)
        self.real_Bs = input['B'].to(self.device)
        self.image_paths_A = input['A_paths']
        self.image_paths_B = input['B_paths']

    def test(self, x_A, x_B):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.gen_a2b.eval()
        self.gen_b2a.eval()
        real_A_pool = []
        real_B_pool = []
        fake_B_pool = []
        fake_A_pool = []
        cycle_A_pool = []
        cycle_B_pool = []
        idt_A_pool = []
        idt_B_pool = []
        with torch.no_grad():
            for i in range(len(x_A)):
                real_A = x_A[i:i+1]; real_A_pool.append(real_A)
                real_B = x_B[i:i+1]; real_B_pool.append(real_B)
                fake_B  = self.gen_a2b(real_A).detach(); fake_B_pool.append(fake_B)
                cycle_A = self.gen_b2a(fake_B).detach(); cycle_A_pool.append(cycle_A)
                idt_A   = self.gen_b2a(real_A).detach(); idt_A_pool.append(idt_A)
                fake_A  = self.gen_b2a(real_B).detach(); fake_A_pool.append(fake_A)
                cycle_B = self.gen_a2b(fake_A).detach(); cycle_B_pool.append(cycle_B)
                idt_B   = self.gen_a2b(real_B).detach(); idt_B_pool.append(idt_B)

        return [real_A_pool, fake_B_pool, cycle_A_pool, idt_A_pool, real_B_pool, fake_A_pool, cycle_B_pool, idt_B_pool]

    def forward(self, x_A, x_B):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_A = x_A
        self.real_B = x_B
        self.fake_B = self.gen_a2b(self.real_A)  # G_A(A)
        self.cycle_A = self.gen_b2a(self.fake_B)   # G_B(G_A(A))
        self.idt_A = self.gen_b2a(self.real_A)
        self.fake_A = self.gen_b2a(self.real_B)  # G_B(B)
        self.cycle_B = self.gen_a2b(self.fake_A)   # G_A(G_B(B))
        self.idt_B  = self.gen_a2b(self.real_B)

    def compute_loss_D(self, real_a, real_b, fake_a, fake_b, beta_a, beta_b):
        self.loss_D_A = self.gan_criterion_a2b.dis_loss(real_b, beta_b, fake_b, beta_a)
        self.loss_D_B = self.gan_criterion_b2a.dis_loss(real_a, beta_a, fake_a, beta_b)
        self.loss_D = self.loss_D_A + self.loss_D_B
        return self.loss_D

    def compute_loss_G(self, beta_a, beta_b):
        """
        Calculate the loss for generators G_A and G_B
        Only G_loss is used to update Beta!!!
        """
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A, beta_a.detach()) * lambda_A * lambda_idt
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B, beta_b.detach()) * lambda_B * lambda_idt
        # GAN loss D_A(G_A(A))
        self.loss_G_A, self.loss_unweight_A = self.gan_criterion_a2b.gen_loss(self.fake_B, beta_a.detach())
        # GAN loss D_B(G_B(B))
        self.loss_G_B, self.loss_unweight_B = self.gan_criterion_b2a.gen_loss(self.fake_A, beta_b.detach())
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.cycle_A, self.real_A, beta_a.detach()) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.cycle_B, self.real_B, beta_b.detach()) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        #return self.loss_G
        return self.loss_G

    def optimize_parameters(self):
        """
        Three steps:
        train G_A and G_B
        train D_A and D_B
        train beta_net_A, beta_net_B
        """

        #------------------------------------------------------------------
        # train G_A and G_B
        self.set_requires_grad([self.dis_a, self.dis_b, self.beta_net_a, self.beta_net_b], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.fake_A_pool = []; self.fake_B_pool = []
        self.beta_a_pool = []; self.beta_b_pool = []
        self.loss_unweight_A_pool = [];  self.loss_unweight_B_pool = []
        self.loss_G_A_pool = []; self.loss_G_B_pool = []
        batch_size = self.opt.batch_size
        # use gradient accumulation to mimic batch-wise training
        beta_as = self.beta_net_a(self.real_As).detach()
        beta_bs = self.beta_net_b(self.real_Bs).detach()
        for i in range(batch_size):
            x_A = self.real_As[i:i+1]; x_B = self.real_Bs[i:i+1]
            self.beta_a = beta_as[i]; self.beta_b = beta_bs[i]
            # forward and backward loss
            self.forward(x_A, x_B)      # compute fake images and reconstruction images.
            loss_G = self.compute_loss_G(self.beta_a, self.beta_b) / batch_size  # calculate gradients for G_A and G_B
            loss_G.backward()         # backward for G_A, G_B update
            self.fake_B_pool.append(self.fake_B.detach()) # buffer for training D
            self.fake_A_pool.append(self.fake_A.detach())
            self.beta_a_pool.append(self.beta_a.detach())
            self.beta_b_pool.append(self.beta_b.detach())
            self.loss_unweight_A_pool.append(self.loss_unweight_A) # buffer for training beta_net
            self.loss_unweight_B_pool.append(self.loss_unweight_B)
        self.optimizer_G.step()       # update G_A and G_B's weights

        #------------------------------------------------------------------
        # train D_A and D_B
        self.set_requires_grad([self.dis_a, self.dis_b], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        for i in range(batch_size):
            loss_D = self.compute_loss_D(self.real_As[i:i+1], self.real_Bs[i:i+1],
                            self.fake_A_pool[i], self.fake_B_pool[i],
                            self.beta_a_pool[i], self.beta_b_pool[i]) / batch_size     # calculate gradients for D_B
            loss_D.backward()
        self.optimizer_D.step()  # update D_A and D_B's weights

        #------------------------------------------------------------------
        # train beta_net_A, beta_net_B
        self.loss_irw_A = 0
        self.loss_irw_B = 0
        if self.opt.beta_mode in ['A', 'B', 'AB']:
            self.set_requires_grad([self.beta_net_a, self.beta_net_b], True)
            self.optimizer_B.zero_grad()
            beta_as = self.beta_net_a(self.real_As.detach())
            beta_bs = self.beta_net_b(self.real_Bs.detach())
            self.loss_beta_A = 0
            self.loss_beta_B = 0
            for i in range(batch_size):
                beta_a = beta_as[i]
                beta_b = beta_bs[i]
                self.loss_beta_A += (beta_a * self.loss_unweight_A_pool[i]/batch_size)
                self.loss_beta_B += (beta_b * self.loss_unweight_B_pool[i]/batch_size)
            self.loss_irw_A = torch.norm(beta_as) / batch_size
            self.loss_irw_B = torch.norm(beta_bs) / batch_size
            irw_loss = self.opt.lambda_irw_A * self.loss_irw_A + self.opt.lambda_irw_B * self.loss_irw_B
            self.loss_beta = self.loss_beta_A + self.loss_beta_B + irw_loss
            self.loss_beta.backward()
            self.optimizer_B.step()

        #------------------------------------------------------------------
        # save input images and betas
        images = torch.cat([self.real_As.detach(), self.real_Bs.detach()], 0)
        fake_images = torch.cat(self.fake_B_pool + self.fake_A_pool, dim=0)
        betas = self.beta_a_pool + self.beta_b_pool
        return images, fake_images, betas

    def get_betas(self, x_as, x_bs):
        self.beta_net_a.eval()
        self.beta_net_b.eval()
        with torch.no_grad():
            beta_as = torch.ones([len(x_as), 1, 1, 1]).to(self.device)
            beta_bs = torch.ones([len(x_bs), 1, 1, 1]).to(self.device)
            if self.opt.beta_mode in ['A', 'AB']:
                beta_as = self.beta_net_a(x_as).detach()
            if self.opt.beta_mode in ['B', 'AB']:
                beta_bs = self.beta_net_b(x_bs).detach()
        self.beta_net_a.train()
        self.beta_net_b.train()
        images = torch.cat([x_as, x_bs], 0)
        betas = torch.cat([beta_as, beta_bs],0)
        return images, betas

    def debug(self, x_as, x_bs):
        with torch.no_grad():
            print('Debugging trainA...')
            for i in range(len(x_as)):
                x = x_as[i].unsqueeze(0)
                dis_fake_b = self.dis_b(self.gen_a2b(x))
                dis_raw_a = self.dis_b(x)
                if len(dis_fake_b) == 1:
                    print(dis_raw_a[0].mean().item(),
                      dis_fake_b[0].mean().item())
                else:
                    print(dis_raw_a[0].mean().item(), dis_raw_a[1].mean().item(),
                          dis_fake_b[0].mean().item(), dis_fake_b[1].mean().item())
            print('Debugging trainB...')
            for i in range(len(x_bs)):
                x = x_bs[i].unsqueeze(0)
                dis_fake_a = self.dis_a(self.gen_b2a(x))
                dis_raw_b = self.dis_a(x)
                if len(dis_fake_b) == 1:
                    print(dis_raw_b[0].mean().item(),
                          dis_fake_a[0].mean().item())
                else:
                    print(dis_raw_b[0].mean().item(), dis_raw_b[1].mean().item(),
                          dis_fake_a[0].mean().item(), dis_fake_a[1].mean().item())


