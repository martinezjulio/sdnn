'''
usage:
CUDA_VISIBLE_DEVICES=0 python train.py --config_file ./configs/test.yaml --num_gpus 1
'''

# -------------- IMPORTS

import argparse
from utils import helper_new as helper

# --------------- FLAGS

bools = ("True", "False")
parser = argparse.ArgumentParser(description='Deep Net Training')
parser.add_argument(
    '--config_file',
    default=None,
    type=str,
    help='path to config file')
parser.add_argument(
    '--num_gpus',
    default=1,
    type=int,
    help='num of gpus to use (0:cpu, default 1)')
parser.add_argument(
    '--pretrained',
    default="False",
    choices=bools,
    help='do previous checkpoints exist (default false)')
parser.add_argument('--restore_epoch', default=-1, type=int,
                    help='epoch to restore (default latest)')
parser.add_argument('--num_epochs', default=50, type=int,
                    help='num of epochs to train for (default 50)')
parser.add_argument('--valid_freq', default=1, type=float,
                    help='validate every x epochs (default 0.5)')
parser.add_argument(
    '--save_freq',
    default=1,
    type=int,
    help='USE CONFIG FILE, save every x epochs (default 5)')
parser.add_argument(
    '--workers',
    default=1,
    type=int,
    help='number of workers for read and write of data (default 1)')
parser.add_argument(
    '--maxout',
    default="True",
    choices=bools,
    help='read all data and then shuffle (default true)')
parser.add_argument(
    '--read_seed',
    default=None,
    type=int,
    help='seed type (default None)')
parser.add_argument(
    '--use_scheduler',
    default="False",
    choices=bools,
    help='USE CONFIG FILE, use a scheduler (default false)')
parser.add_argument(
    '--custom_learning_rate',
    default=0.001,
    type=float,
    help='force constant learning rate (default 1e-3)')
parser.add_argument(
    '--max_count',
    default=-1,
    type=int,
    help='max number of checkpoints to keep (default all)')
parser.add_argument(
    '--debug',
    default="False",
    choices=bools,
    help='log memory usage (default false)')
parser.add_argument(
    '--split',
    default="False",
    choices=bools,
    help='use split model (default false)')


FLAGS, FIRE_FLAGS = parser.parse_known_args()

flags = "pretrained", "maxout", "use_scheduler", "debug", "split"
# flags = "maxout", "use_scheduler", "debug", "split"
for flag in flags:
    setattr(FLAGS, flag, getattr(FLAGS, flag) == "True")

print(FLAGS)


# ---------------- MAIN

if __name__ == "__main__":
    if FLAGS.split:
        trainer = helper.debug_split if FLAGS.debug else helper.train_split
    else:
        trainer = helper.debug if FLAGS.debug else helper.train
    # Train Network
    trainer(config_file=FLAGS.config_file,
            pretrained=FLAGS.pretrained,
            restore_epoch=FLAGS.restore_epoch,
            epochs=FLAGS.num_epochs,
            valid_freq=FLAGS.valid_freq,
            save_freq=FLAGS.save_freq,
            workers=FLAGS.workers,
            ngpus=FLAGS.num_gpus,
            notebook=False,
            maxout=FLAGS.maxout,
            read_seed=FLAGS.read_seed,
            use_scheduler=FLAGS.use_scheduler,
            custom_learning_rate=FLAGS.custom_learning_rate,
            max_count=FLAGS.max_count)
