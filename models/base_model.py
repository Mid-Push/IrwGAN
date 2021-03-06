import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import util.util as util

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.opt_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass
    @property
    def model_name(self):
        return ''
    @abstractmethod
    def print_information(self, opt):
        pass
    def setup(self, opt):
        opt.run_dir = os.path.join(opt.result_dir, self.model_name)
        self.run_dir = opt.run_dir
        util.mkdirs(self.run_dir)
        util.mkdirs(os.path.join(self.run_dir, 'fakeB'))
        util.mkdirs(os.path.join(self.run_dir, 'fakeA'))
        util.mkdirs(os.path.join(self.run_dir, 'img'))
        util.mkdirs(os.path.join(self.run_dir, 'model'))
        util.mkdirs(os.path.join(self.run_dir, 'log'))
        eval_log = os.path.join(self.run_dir, 'metric-fid.txt')
        if opt.phase == 'train':
            f = open(eval_log, 'w')
            f.writelines('\n###################################\n')
            f.writelines('########### training ##############\n')
            f.writelines('###################################\n')
            f.close()
        if opt.phase == 'test':
            f = open(eval_log, 'a')
            f.writelines('\n###################################\n')
            f.writelines('############## test ###############\n')
            f.writelines('###################################\n')
            f.close()

        if opt.phase == 'train' or opt.phase == 'resume':
            log_name = os.path.join(self.run_dir, 'log', 'training_log.txt')
            self.logger = util.Logger(log_name, append=(opt.phase == 'resume'))

        self.print_information(opt)
        self.setup_networks(opt)

    def setup_networks(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if opt.phase == 'train':
            # we already initialized the networks while constructing them
            pass

        if opt.phase == 'test':
            # load the latest model for testing
            if opt.epoch_count>0:
                print('[*] resuming from %d' %opt.epoch_count)
                load_path = os.path.join(self.run_dir, 'model', 'network-snapshot-%03d.pth' % opt.epoch_count)
            #load_path = os.path.join('results/selfie2anime-danbooru_lsgan_20_B_gl_1.0_thr0.1', 'model', 'network-snapshot-latest.pth')
                self.load_networks(load_path)

        if opt.phase == 'resume':
                # automatically load the model if resume flag is True
                latest_model_name = util.get_model_list(os.path.join(self.run_dir, 'model'), key='network', exclude='latest')
                self.load_networks(latest_model_name)
                opt.epoch_count = int(os.path.basename(latest_model_name).split('.')[0].split('-')[-1])  # setup the epoch_count to start with

        # put opt later as we load_networks will load optimizer as well
        if opt.phase == 'train' or opt.phase == 'resume':
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            # we do not have to save lr_scheduler as it will be reflected in opt.epoch_count
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch, used_time):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        model_dict = {}
        if isinstance(epoch, str):
            save_filename = 'network-snapshot-%s.pth' % (epoch)
        else:
            save_filename = 'network-snapshot-%03d.pth' % (epoch)
        save_path = os.path.join(self.run_dir, 'model', save_filename)
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    model_dict[name] = net.module.cpu().state_dict()
                    net.cuda(self.gpu_ids[0])
                else:
                    model_dict[name] = net.cpu().state_dict()
                    net.cuda(self.gpu_ids[0])

        # save optimizers
        for name in self.opt_names:
            if isinstance(name, str):
                opt = getattr(self, name)
                model_dict[name] = opt.state_dict()
        model_dict['used_time'] = used_time
        torch.save(model_dict, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, latest_model_name):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        load_path = latest_model_name
        model = torch.load(load_path, map_location=str(self.device))
        print('Loading from %s' % load_path)
        print('Loading models...')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(model[name])
        # load optimizers
        if len(self.opt_names) > 0:
            print('Loading optimizers...')
        for name in self.opt_names:
            if isinstance(name, str):
                opt = getattr(self, name)
                opt.load_state_dict(model[name])
        self.opt.used_time = model['used_time']

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        if self.opt.phase == 'train':
            print('---------- Networks initialized -------------')
        else:
            print('-------------- Networks loaded ----------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
