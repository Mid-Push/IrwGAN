import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import loss
from util.util import to_data
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
            parser.add_argument('--lambda_irw_A', type=float, default=0.0, help='weight for controlling the sparsity of beta')
            parser.add_argument('--lambda_irw_B', type=float, default=0.0, help='weight for controlling the sparsity of beta')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'cycle_A', 'idt_A']
        visual_names_B = ['real_B', 'fake_A', 'cycle_B', 'idt_B']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['gen_a2b', 'gen_b2a', 'dis_a', 'dis_b', 'beta_net_a', 'beta_net_b']
        else:  # during test time, only load Gs
            self.model_names = ['gen_a2b', 'gen_b2a']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.gen_a2b = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.gen_b2a = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.dis_a = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, 'normal', 0.02, self.gpu_ids)
            self.dis_b = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, 'normal', 0.02, self.gpu_ids)
            self.beta_net_a = networks.define_BetaNet(opt.input_nc, opt.ndf, 3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.beta_net_b = networks.define_BetaNet(opt.input_nc, opt.ndf, 3, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # define loss functions
            self.gan_criterion_a2b = loss.IRW_MS_LSGANLoss(self.dis_b)
            self.gan_criterion_b2a = loss.IRW_MS_LSGANLoss(self.dis_a)
            self.criterionCycle = loss.IRW_L1_Loss()
            self.criterionIdt = loss.IRW_L1_Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.gen_a2b.parameters(), self.gen_b2a.parameters(), self.beta_net_a.parameters(), self.beta_net_b.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.dis_a.parameters(), self.dis_b.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            print('#### Information ####')
            print('# lambda_irw_A: %f' % opt.lambda_irw_A)
            print('# lambda_irw_B: %f' % opt.lambda_irw_B)
            print('# lambda_identity: %f'%opt.lambda_identity)
            print('# lr: %f' % opt.lr)
            print('# batch_size: %d' % opt.batch_size)
            print('# init_type: %s' % opt.init_type)
            print('# netD: %s' % opt.netD)
            print('# beta_mode: %s' % opt.beta_mode)
            print()

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
        real_A= x_A.detach()
        real_B = x_B.detach()
        with torch.no_grad():
            fake_B = self.gen_a2b(real_A).detach()  # G_A(A)
            cycle_A = self.gen_b2a(fake_B).detach()   # G_B(G_A(A))
            idt_A = self.gen_b2a(real_A).detach()
            fake_A = self.gen_b2a(real_B).detach()  # G_B(B)
            cycle_B = self.gen_a2b(fake_A).detach()   # G_A(G_B(B))
            idt_B  = self.gen_a2b(real_B).detach()
        return [real_A, fake_B, cycle_A, idt_A, real_B, fake_A, cycle_B, idt_B]

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
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A, beta_a) * lambda_A * lambda_idt
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B, beta_b) * lambda_B * lambda_idt

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.gan_criterion_a2b.gen_loss(self.fake_B, beta_a)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.gan_criterion_b2a.gen_loss(self.fake_A, beta_b)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.cycle_A, self.real_A, beta_a) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.cycle_B, self.real_B, beta_b) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        return self.loss_G

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # G_A and G_B
        self.set_requires_grad([self.dis_a, self.dis_b], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.fake_A_pool = []
        self.fake_B_pool = []
        self.beta_a_pool = []
        self.beta_b_pool = []
        batch_size = self.opt.batch_size
        # use gradient accumulation to mimic batch-wise training
        for i in range(batch_size):
            x_A = self.real_As[i:i+1]
            x_B = self.real_Bs[i:i+1]
            if self.opt.beta_mode == 'baseline':
                beta_as = torch.ones([batch_size, 1, 1, 1]).to(self.device)
                beta_bs = torch.ones([batch_size, 1, 1, 1]).to(self.device)
            else:
                beta_as = self.beta_net_a(self.real_As)
                beta_bs = self.beta_net_b(self.real_Bs)
            self.beta_a = beta_as[i]
            self.beta_b = beta_bs[i]
            self.forward(x_A, x_B)      # compute fake images and reconstruction images.
            self.fake_B_pool.append(self.fake_B.detach())
            self.fake_A_pool.append(self.fake_A.detach())
            self.beta_a_pool.append(self.beta_a.detach())
            self.beta_b_pool.append(self.beta_b.detach())
            loss_G = self.compute_loss_G(self.beta_a, self.beta_b)/batch_size             # calculate gradients for G_A and G_B
            loss_G.backward()
        beta_as = self.beta_net_a(self.real_As)
        beta_bs = self.beta_net_b(self.real_Bs)
        self.irw_loss_A = torch.norm(beta_as)
        self.irw_loss_B = torch.norm(beta_bs)
        irw_loss = self.opt.lambda_irw_A * self.irw_loss_A + self.opt.lambda_irw_B * self.irw_loss_B
        irw_loss.backward()
        self.optimizer_G.step()       # update G_A and G_B's weights


        # D_A and D_B
        self.set_requires_grad([self.dis_a, self.dis_b], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        for i in range(batch_size):
            loss_D = self.compute_loss_D(self.real_As[i:i+1], self.real_Bs[i:i+1],
                            self.fake_A_pool[i], self.fake_B_pool[i],
                            self.beta_a_pool[i], self.beta_b_pool[i]) / batch_size     # calculate gradients for D_B
            loss_D.backward()
        self.optimizer_D.step()  # update D_A and D_B's weights

        return to_data(beta_as.squeeze().detach()), to_data(beta_bs.squeeze().detach())

    def get_current_betas(self):
        half  = len(self.visual_names)//2
        betas = ['beta_%4.3f' % self.beta_a.item()]
        betas += [''] * (half-1)
        betas += ['beta_%4.3f' % self.beta_b.item()]
        betas += [''] * (half-1)
        assert len(betas) == len(self.visual_names)
        return betas
