
import argparse
import utils 
import os


class BaseParser():
    """This class defines options used during both training and eval time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        #parser.add_argument('-phase', type=str,  default='train',  help='train, extract_features, vizualize')
        parser.add_argument('-data_root', type=str,  default='Datasets',  help='The repository where is the dataset')
        parser.add_argument('-dataset_name', type=str,  default='vtb-balanced-patients-202107091800',  help='The name of the dataset')
        parser.add_argument('-data_json', type=str,  default='vtb.balanced-patients.202107091800.json',  help='The file containing labels')

        # Dataset Configurations
        parser.add_argument('-num_classes', type=int,  default=2, help='Number of classes in the dataset.')
        parser.add_argument('-channel', type=int,  default=1, help='Number of channels')
        
        # Model Configurations
        parser.add_argument('-backbone',type=str, default='resnet', help='alexnet, resnet18, lightnet')

        parser.add_argument('-phase',type=str, default='train', help='train, test, extract_features, vizualize_saliency_map, all')

        parser.add_argument('-gpus', type=str, default='0')
        parser.add_argument('-log_dir', type=str, default='logs')

        parser.add_argument('-batch_size', type=int, default=4)
        parser.add_argument('-seed', type=int, default=42)
        parser.add_argument('-n_runs', type=int, default=1)

        # Optimization Configuration
        parser.add_argument('-l2_weight', type=float, default=0.0001,
                            help='L2 loss weight applied all the weights')
        parser.add_argument(
            '-momentum', type=float, default=0.9, help='The momentum of MomentumOptimizer')
        parser.add_argument('-init_lr', type=float , default=0.01, help='Initial learning rate')
        parser.add_argument('-step_size', type=int, default=7,
                            help='Epochs after which learing rate decays')
        parser.add_argument(
            '-lr_decay', type=float, default=0.1, help='Learning rate decay factor')
        parser.add_argument('-finetune', type=utils.boolean_string,  default=False,
                            help='Whether to finetune.')

        # Training Configuration
        parser.add_argument('-num_epochs', type=int,  default=20, help='Number of batches to run.')
        parser.add_argument('-load_checkpoint', type=int, default=19,  metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
        parser.add_argument('-checkpoint_path', type=str, default='checkpoint')

        self.initialized = True
        return parser


    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        self.opt, _ = parser.parse_known_args()


        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoint_path, opt.backbone)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()

        self.print_options(opt)

        return self.opt




