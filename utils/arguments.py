import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

data_args = add_argument_group('Dataset')
data_args.add_argument('--data_root', type=str, default='./dataset/Harmonization_dataset',
                       help='Path to the dataset root directory (default: ./dataset/Harmonization_dataset).')
data_args.add_argument('--vendor1', type=str, default='SIEMENS',
                       help='Name of the source/vendor A (e.g., SIEMENS). Used to select/source images.')
data_args.add_argument('--vendor2', type=str, default='Philips',
                       help='Name of the target/vendor B (e.g., Philips). Used for harmonization target selection.')
data_args.add_argument('--lower_bound', type=float, default=0.1,
                       help='Lower bound for usage of the input slice (range 0.0-1.0).')
data_args.add_argument('--upper_bound', type=float, default=0.9,
                       help='Upper bound for usage of the input slice (range 0.0-1.0).')

train_args = add_argument_group('Training')
train_args.add_argument('--total_epoch', type=int, default=50,
                        help='Total number of training epochs to run.')
train_args.add_argument('--decay_epoch', type=int, default=4,
                        help='Epoch interval (or starting epoch) for learning-rate decay. Adjust based on schedule used in training loop.')
train_args.add_argument('--batch_size', type=int, default=16,
                        help='Batch size used for training iterations.')
train_args.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Initial learning rate for the optimizer.')
train_args.add_argument('--ccl_weight', type=float, default=5.,
                        help='Weight for the Cycle Consistency loss term (controls contribution of contrastive/cycle consistency loss).')
train_args.add_argument('--identity_weight', type=float, default=1.,
                        help='Weight for the Identity loss term (encourages preserving input when appropriate).')
train_args.add_argument('--oml_weight', type=float, default=0,
                        help='Weight for the Original Matching loss term.')

misc_args = add_argument_group('Misc')
misc_args.add_argument('--save_dir', type=str, default='./results',
                       help='Directory where checkpoints, logs, and sample outputs are saved.')
misc_args.add_argument('-ex', '--experiment', type=str, required=True,
                       help='Experiment name or identifier (required). Used to name the experiment subfolder under `--save_dir`.')
misc_args.add_argument('--log_freq', type=int, default=200,
                       help='Iteration frequency to print training/validation logs.')
misc_args.add_argument('--save_freq', type=int, default=5,
                       help='Epoch frequency to save model state.')
misc_args.add_argument('--sample_save_freq', type=int, default=40,
                       help='Iteration frequency to save generated sample outputs.')
misc_args.add_argument('--random_seed', type=int, default=1234,
                       help='Random seed for reproducibility across runs.')
misc_args.add_argument('--num_workers', type=int, default=1)

def get_args():
    args = parser.parse_args()
    return args, arg_lists
