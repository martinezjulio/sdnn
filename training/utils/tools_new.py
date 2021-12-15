import os
import numpy as np
import models
import torch

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
    assert(len(checkpoints) > 0), "No checkpoints found."
    np.sort(checkpoints)[::-1]
    
    if (epoch == -1):
        checkpoint = checkpoints[-1]
    else:
        terms = np.array([checkpoints[i].split('_')[1].split('.')[0] for i in range(len(checkpoints))]).astype(int)
        #print(terms)
        abs_diffs = np.abs(terms - epoch)
        #print(abs_diffs)
        epoch_term = np.argmin(abs_diffs)
        #print(epoch_term)
        checkpoint = checkpoints[epoch_term]
        
        if abs_diffs[epoch_term] > 0:
            print('Epoch ' + str(epoch) + ' doesnt exist. Returning epoch ' + str(terms[epoch_term]) + ' as closest match.')
        
    checkpoint = os.path.join(checkpoints_dir, checkpoint)
    checkpoint = os.path.abspath(checkpoint)
    return checkpoint

def save_checkpoint(ckpt_data, ckpt_dir, max_count=-1, keep=None):
    '''
    Description:
        Saves checkpoint and maintains maximum checkpoint count if appropriate.
        Assumes all checkpoints in checkpoints_dir are saved as epoch_X*X.pth.tar 
        where XX or XXX etc denotes the epoch of the checkpoint. Also assumes all
        checkpoints are saved in strictly linearly increasing epoch values.
        
    Inputs: 
        @ckpt_data <dict>  - dictionary of values to save
        @ckpt_dir  <str>   - the path where all checkpoints are stored 
        @max_count <int>   - optional, max # of checkpoints allowed
        @keep      <set>   - optiona, checkpoints to never delete
    '''

    keep = set() if keep is None else set(f'epoch_{epoch:02d}.pth.tar' for epoch in keep)
    epoch = ckpt_data["epoch"]
    if max_count < -1: 
        print("Invalid max_count. Will not remove any checkpoints.")
        max_count = -1
    if max_count == -1: torch.save(ckpt_data, os.path.join(ckpt_dir,f'epoch_{epoch:02d}.pth.tar'))
    elif epoch in keep: torch.save(ckpt_data, os.path.join(ckpt_dir,f'epoch_{epoch:02d}.pth.tar'))
    else:
        checkpoints = os.listdir(ckpt_dir)
        if keep: checkpoints = [ckpt for ckpt in checkpoints if ckpt not in keep]
        removable = len(checkpoints)
        if removable + 1 <= max_count: torch.save(ckpt_data, os.path.join(ckpt_dir,f'epoch_{epoch:02d}.pth.tar'))
        elif max_count == 0:
            for ckpt in checkpoints: os.remove(os.path.join(ckpt_dir, ckpt))
        else:
            removed = 0
            while removable - removed >= max_count:
                os.remove(os.path.join(ckpt_dir, checkpoints[removed]))
                removed += 1
            torch.save(ckpt_data, os.path.join(ckpt_dir,f'epoch_{epoch:02d}.pth.tar'))

def get_num_classes(data_dir, islist=False):
    if islist:
        classes = []
        for i in range(len(data_dir)):
            dir = os.path.join(data_dir[i], 'train')
            classes.extend([os.path.abspath(d) for d in os.scandir(dir) if d.is_dir()])
        num_classes = len(classes)
    else:
        data_dir = os.path.join(data_dir, 'train')
        num_classes = len([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])

    return num_classes

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def get_model(name, num_classes, ngpus, split_index=None, dataParallel=False, scale=None, fc_affected=None):
    modelClass = getattr(models, name)
    
    if split_index is None:
        if scale is None: modelObject = modelClass(num_classes=num_classes)
        elif fc_affected is None: modelObject = modelClass(num_classes=num_classes, scale=scale)
        else: modelObject = modelClass(num_classes=num_classes, scale=scale, fc_scaling=fc_affected)
    else:
        if scale is None and fc_affected is None: modelObject = modelClass(num_classes=num_classes, split_index=split_index)
        elif scale is None: modelObject = modelClass(num_classes=num_classes, split_index=split_index, fc_doubling=fc_affected)
        else: modelObject = modelClass(num_classes=num_classes, split_index=split_index, scale=scale, fc_scaling=fc_affected)
    
    if dataParallel: modelObject = torch.nn.DataParallel(modelObject)
        
    if ngpus > 0: modelObject = modelObject.cuda()
        
    return modelObject

def get_rankings(x):
    temp = x.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    return ranks
