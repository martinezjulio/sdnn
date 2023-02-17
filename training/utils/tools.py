import os
import numpy as np
#import models
import torch

# def get_checkpoint(checkpoints_dir, epoch=-1):
#    '''
#    Description:
#        Assumes all checkpoints in checkpoints_dir are saved as epoch_X*X.pth.tar
#        where XX or XXX etc denotes the epoch of the checkpoint. Also assumes all
#        checkpoints are saved in strictly linearly increasing epoch values.
#
#    Inputs:
#        @checkpoints_dir - the path where all checkpoints are stored
#        @epoch           - default as the latest checkpoint
#     Returns:
#         The asbolute path of the checkpoint of the epoch closest to the value epoch
#    '''
#
#
#    checkpoints = np.array(os.listdir(checkpoints_dir))
#    assert(len(checkpoints) > 0), "No checkpoints found."
#    np.sort(checkpoints)[::-1]
#
#    if (epoch == -1):
#        checkpoint = checkpoints[-1]
#    else:
#        term0 = int(checkpoints[0].split('_')[1].split('.')[0])
#        term1 = int(checkpoints[1].split('_')[1].split('.')[0])
#        diff = term1 - term0
#        epoch_term = np.round(epoch/diff).astype(int)
#        assert(epoch_term-1 < len(checkpoints)), "Epoch not found."
#        checkpoint = checkpoints[epoch_term-1]
#
#    checkpoint = os.path.join(checkpoints_dir, checkpoint)
#    checkpoint = os.path.abspath(checkpoint)
#    return checkpoint


def get_checkpoint(checkpoints_dir, epoch=-1):
    '''
    Description:
        Assumes all checkpoints in checkpoints_dir are saved as epoch_X*X.pth.tar
        where XX or XXX etc denotes the epoch of the checkpoint. Also assumes all
        checkpoints are saved in strictly linearly increasing epoch values.

    Inputs:
        @checkpoints_dir - the path where all checkpoints are stored
        @epoch           - default as the latest checkpoint
     Returns:
         The asbolute path of the checkpoint of the epoch closest to the value epoch
    '''

    checkpoints = np.array(os.listdir(checkpoints_dir))
    assert (len(checkpoints) > 0), "No checkpoints found."
    np.sort(checkpoints)[::-1]

    # print(checkpoints)

    if (epoch == -1):
        checkpoint = checkpoints[-1]
    else:
        terms = np.array([checkpoints[i].split('_')[1].split('.')[0]
                         for i in range(len(checkpoints))]).astype(int)
        # print(terms)
        abs_diffs = np.abs(terms - epoch)
        # print(abs_diffs)
        epoch_term = np.argmin(abs_diffs)
        # print(epoch_term)
        checkpoint = checkpoints[epoch_term]

        if abs_diffs[epoch_term] > 0:
            print('Epoch ' +
                  str(epoch) +
                  ' doesnt exist. Returning epoch ' +
                  str(terms[epoch_term]) +
                  ' as closest match.')

    checkpoint = os.path.join(checkpoints_dir, checkpoint)
    checkpoint = os.path.abspath(checkpoint)
    return checkpoint


def get_num_classes(data_dir, islist=False):
    if islist:
        classes = []
        for i in range(len(data_dir)):
            dir = os.path.join(data_dir[i], 'train')
            classes.extend([os.path.abspath(d)
                           for d in os.scandir(dir) if d.is_dir()])
        num_classes = len(classes)
    else:
        data_dir = os.path.join(data_dir, 'train')
        num_classes = len([x for x in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, x))])

    return num_classes


def find_classes(dir):
    classes = sorted(os.listdir(dir))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def get_model(name, num_classes, ngpus, split_index=None, dataParallel=False):
    modelClass = getattr(models, name)
    if split_index is None:
        modelObject = modelClass(num_classes=num_classes)
    else:
        modelObject = modelClass(
            num_classes=num_classes,
            split_index=split_index)
    if dataParallel:
        modelObject = torch.nn.DataParallel(modelObject)
    if ngpus > 0:
        modelObject = modelObject.cuda()
    return modelObject


def get_rankings(x):
    temp = x.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    return ranks
