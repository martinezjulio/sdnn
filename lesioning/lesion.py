# imports
import os
import glob
import numpy as np
import copy
from utils import helper
from sklearn import metrics
import tqdm
import torch
import torchvision
import torchvision.models as models
import numpy as np
import copy
import utils
import scipy
import h5py
import argparse
import json
import jsonlines as jsonl
import time
from zipfile import BadZipFile
from filelock import Timeout
from filelock import SoftFileLock as FileLock
import random
import pprint
pp = pprint.PrettyPrinter(indent=1)
lock_timeout = 30*2
acquire_timeout = 30*2
fixed_seed_value = 0

IMAGE_RESIZE=256
IMAGE_SIZE=224
GRAYSCALE_PROBABILITY=0.2
resize_transform      = torchvision.transforms.Resize(IMAGE_RESIZE)
random_crop_transform = torchvision.transforms.RandomCrop(IMAGE_SIZE)
center_crop_transform = torchvision.transforms.CenterCrop(IMAGE_SIZE)
grayscale_transform   = torchvision.transforms.RandomGrayscale(p=GRAYSCALE_PROBABILITY)
normalize             = torchvision.transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)

class Validator(object):
    def __init__(self, name, model, batch_size, data_dir, ngpus, workers, 
                 max_samples=None, maxout=True, read_seed=None, 
                 shuffle=False, data_subdir='test', includePaths=False):
        self.name = name
        self.model = model
        self.max_samples = max_samples
        self.maxout=maxout
        self.read_seed=read_seed
        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.ngpus = ngpus
        self.batch_size = batch_size
        self.workers = workers
        self.shuffle = shuffle
        self.includePaths = includePaths
        self.dataset, self.data_loader = self.data()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        if self.ngpus > 0:
            self.criterion = self.criterion.cuda()
 
    def data(self):
        if type(self.data_dir) is list:
            ImageFolder = utils.folder_list.ImageFolder
            test_data_dir = []
            for i in range(len(self.data_dir)):
                test_data_dir.append(os.path.join(self.data_dir[i], self.data_subdir))
        else:
            ImageFolder = utils.folder.ImageFolder
            test_data_dir = os.path.join(self.data_dir, self.data_subdir)
        

        transform = torchvision.transforms.Compose([resize_transform, 
                                                    center_crop_transform, 
                                                    torchvision.transforms.ToTensor(),
                                                    normalize,
                                                   ])     
        
        dataset = ImageFolder(root=test_data_dir, 
                              max_samples=self.max_samples,
                              maxout=self.maxout,
                              read_seed=self.read_seed,
                              transform=transform,
                              includePaths=self.includePaths)

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=self.shuffle,
                                                  num_workers=self.workers,
                                                  pin_memory=True)
        return dataset, data_loader
    
    def __call__(self, x, y):
        if self.ngpus > 0:
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=True)
        output = self.model(x=x)
        prec_1, prec_5 = precision(output=output, target=y, topk=(1,5))
        prec_1 /= len(output)
        prec_5 /= len(output)
        loss = self.criterion(output,y)
        return loss.item(), prec_1, prec_5, output

def get_model(config_file, config, ngpus, dataParallel=False, pretrained=False, epoch=-1):
 
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    
    num_classes = utils.tools.get_num_classes(config["data_dirs"], islist=True)
    model = models.__dict__[config["arch"]](num_classes=num_classes)
    print("Initialized model with", config["arch"], "architecture.")
    
    if dataParallel:
        print('\nApplying DataParallel...')
        model = torch.nn.DataParallel(model)
    if ngpus > 0:
        print('Loading model onto gpu...')
        model = model.cuda()
    
    if pretrained:
        pretrained_path = utils.tools.get_checkpoint(
            epoch=epoch,
            checkpoints_dir=os.path.join(config["checkpoints"],os.path.basename(config_file)[:-5])
        )
        if ngpus > 0:
            ckpt_data = torch.load(pretrained_path)
        else:
            ckpt_data = torch.load(pretrained_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt_data['model_state_dict'])
        print('\nLoaded pretrained model from:', pretrained_path)
    return model

def load_record(filename):
    '''
    Description:
        Since npz file are ovewritten by several jobs at similar times may run into 
        corrupted files by opening at the same time resulting in an error. By using a while try
        loop you can avoid this until the coast is clear to load. 
    Input:
        filename: filename with extension .npz
    Return:
        record: an npz record. To see contents try record.files
    '''
    file_not_loaded=True
    attempts=1
    while file_not_loaded:
        try:
            record=np.load(filename, allow_pickle=True)
            file_not_loaded=False
        except:
            print('\nWHILE LOADING FILE: Failed Attempt', attempts, '.\n', flush=True)
            #time.sleep(5)
            attempts+=1
    return record

def get_latest_npz_filename(dir):
    '''
    Description:
        Returns the latest file in the directory dir
    '''
    list_of_files = glob.glob(os.path.join(dir, '*.npz')) 
    try:
        latest_file = max(list_of_files, key=os.path.getctime)
    except:
        latest_file = None
    return latest_file

def conclude_lesion_to_json(filename, sort_task, param_group_index):
    key = os.path.join('status','SORTEDBY_' + sort_task, str(param_group_index))
    lesion_data = get_lesion_data(filename)
    if key not in lesion_data.keys():
        count=1
        write_to_json(filename=filename, writer_method='a', keys=[key], values=['complete'])
    elif lesion_data[key]=='complete':
        count=2
    return count

def write_to_json(filename, writer_method, keys, values):
    num_keys = len(keys)
    assert(num_keys == len(values)), 'keys and values must have equal lengths'
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        with open(filename, writer_method) as outfile:
            for i in range(num_keys):
                key = keys[i]
                value = values[i]
                json.dump({key : value}, outfile)
                outfile.write('\n')
    finally:
        lock.release()
        
def get_lesion_data(filename):
    '''Description:
            returns a dictionary of all json objects in the specificed jsonlines file
            object occuring more than once with same key will appear uniquely in returned dictionary 
            with the last json object overwriting previous json objects of the same key
        Returns:
            lesion_data: python dciionary
    '''
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        lesion_data = {}
        with jsonl.open(filename) as reader:
            for obj in reader:
                key = list(obj.keys())[0]
                lesion_data[key]=obj[key]
    finally:
        lock.release()
    return lesion_data
        
def json_completion_status(filename, sort_task, param_group_index):
    '''
    Description: 
        True is status for group_index lesion is complete, False otherwise
    Returns:
        is_complete - boolean
    '''
    lesion_data = get_lesion_data(filename)
    
    key = os.path.join('status', 'SORTEDBY_' + sort_task, str(param_group_index))
    value='not submitted'
    if key in lesion_data:
        value = lesion_data[key]
    is_complete=False
    if value == 'complete':
        is_complete=True
    return is_complete

def get_predictions(pred_file, pred_key):
    with open(pred_file) as f:
        for line in f:
            obj = json.loads(line)
            obj_key = list(obj.keys())[0]
            if obj_key == pred_key:
                return obj[obj_key]

def write_obj2jsonl(filename, writer_method, key, obj):
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        with open(filename, writer_method) as outfile:
            json.dump({key : obj}, outfile)
            outfile.write('\n')
    finally:
        lock.release()
        
def randomize_classes(sort_task_index, seed, validator_sort_task, validator_nonsort_task=None):
    '''
    Description: randomly reassigns (swaps) half the classes (and data) of each validator to the other
    '''
    
    print('\nRandomizing Classes', flush=True)
    
    num_classes = np.sum(list(validator_sort_task.dataset.task_to_num_classes.values()))
    
    # random classes for task1
    np.random.seed(seed=seed)
    random_classes_task1 = np.random.choice(a=np.arange(num_classes), size=num_classes//2, replace=False, p=None)
    random_classes_task1 = np.sort(random_classes_task1)

    # random classes for task2 
    random_classes_task2 = [i for i in range(num_classes) if i not in random_classes_task1]   
    random_classes_task2 = np.array(random_classes_task2)
    
    # assign classes by sort and nonsort task
    if sort_task_index==0:
        random_classes_sort_task = random_classes_task1
        random_classes_nonsort_task = random_classes_task2
    else:
        random_classes_sort_task = random_classes_task2
        random_classes_nonsort_task = random_classes_task1
        
    # now gather samples for sort_task
    random_samples_sort_task = []
    for sample in validator_sort_task.dataset.samples:
        if sample[1] in random_classes_sort_task:
            random_samples_sort_task.append(sample)
    for sample in validator_nonsort_task.dataset.samples:
        if sample[1] in random_classes_sort_task:
            random_samples_sort_task.append(sample) 
              
    # now gather samples for nonsort_task
    random_samples_nonsort_task = []
    for sample in validator_sort_task.dataset.samples:
        if sample[1] in random_classes_nonsort_task:
            random_samples_nonsort_task.append(sample)
    for sample in validator_nonsort_task.dataset.samples:
        if sample[1] in random_classes_nonsort_task:
            random_samples_nonsort_task.append(sample)  
     
    # now modify/update the validators
    validator_sort_task.dataset.samples = random_samples_sort_task
    validator_nonsort_task.dataset.samples = random_samples_nonsort_task
    
    print('\nvalidator_sort_task:')
    print(validator_sort_task.dataset.samples[0])
    print(validator_sort_task.dataset.samples[-1])
    
    print('\nvalidator_nonsort_task:')
    print(validator_nonsort_task.dataset.samples[0])
    print(validator_nonsort_task.dataset.samples[-1])

def generate_unit(filename, num_units, progress_dir, next_iter=False, overwrite=False, iteration=0):
    '''
    Desription: picks the first unit from the remaining units if one exists and returns it, 
                along with an updated selected_units array only returned for convenience 
                of saving at end. If no remaining unit exists will return None with no update if next_iter=False
                otherwise with return Non with a new iteration file if next_iter=True
    '''
    
    if filename is None or overwrite==True:
        if not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
        filename = os.path.join(progress_dir,'progress_record_ITER_' + str(iteration)) + '.npz'
        lockname = filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            remaining_units = np.arange(num_units).astype(int)
            unit = remaining_units[0]
            unit_timestamp = time.time()
            remaining_units = np.delete(arr=remaining_units, obj=0) 
            pending_units = np.array([unit]).astype(int)
            pending_clock = np.array([unit_timestamp])
            seed_value = random.randint(0,1000)
            #print('calling random seed:', seed_value, flush=True)
            np.savez(file                = filename,
                     selected_units      = np.array([]).astype(int),
                     selected_losses     = np.array([]).astype(float),
                     selected_accuracies = np.array([]).astype(float),
                     selected_subperformances = np.array([]).astype(float),
                     selections_complete = np.array([False]),
                     remaining_units     = remaining_units, 
                     pending_units       = pending_units,
                     pending_clock       = pending_clock,
                     dropped_units       = np.array([]).astype(int),
                     dropped_losses      = np.array([]).astype(float),
                     dropped_accuracies = np.array([]).astype(float),
                     dropped_subperformances = np.array([]).astype(float),
                     next_iter_made      = np.array([False]),
                     selection_made      = np.array([False]),
                     seed                = np.array([seed_value]),
                     conclusion_count    = np.array([0]) 
                    )
        finally:
            lock.release()
    else:
        lockname = filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            progress_record     = load_record(filename)
            selected_units      = progress_record['selected_units']
            selected_losses     = progress_record['selected_losses']
            selected_accuracies = progress_record['selected_accuracies']
            selected_subperformances = progress_record['selected_subperformances']
            selections_complete = progress_record['selections_complete']
            remaining_units     = progress_record['remaining_units'] 
            pending_units       = progress_record['pending_units']
            pending_clock       = progress_record['pending_clock']
            dropped_units       = progress_record['dropped_units']
            dropped_losses      = progress_record['dropped_losses']
            dropped_accuracies = progress_record['dropped_accuracies']
            dropped_subperformances = progress_record['dropped_subperformances']
            next_iter_made      = progress_record['next_iter_made']
            selection_made      = progress_record['selection_made']
            seed                = progress_record['seed']
            conclusion_count    = progress_record['conclusion_count']

            # generate unit if there are any units remaining
            if remaining_units.shape[0] > 0:
                unit = remaining_units[0]
                unit_timestamp = time.time()
                seed_value = seed[0]
                #print('calling random seed:', seed_value, flush=True)
                remaining_units = np.delete(arr=remaining_units, obj=0)  # removes the zeroth index
                pending_units = np.append(pending_units, unit)
                pending_clock = np.append(pending_clock, unit_timestamp)
                # filename unchanged
                np.savez(file                = filename,
                         selected_units      = selected_units,
                         selected_losses     = selected_losses,
                         selected_accuracies = selected_accuracies,
                         selected_subperformances = selected_subperformances,
                         selections_complete = selections_complete,
                         remaining_units     = remaining_units,
                         pending_units       = pending_units,
                         pending_clock       = pending_clock,
                         dropped_units       = dropped_units,
                         dropped_losses      = dropped_losses,
                         dropped_accuracies  = dropped_accuracies,
                         dropped_subperformances = dropped_subperformances,
                         next_iter_made      = next_iter_made,
                         selection_made      = selection_made,
                         seed                = seed,
                         conclusion_count    = conclusion_count)
            
            # genetate empty signal (unit=None) if no units remain but do not create new iteration file
            elif (remaining_units.shape[0] == 0) and (next_iter == False):
                # filename unchanged
                unit = None
                seed_value = None
            
            # generate empty signal (unit=None) if no units remain and create new iteration file
            elif (remaining_units.shape[0] == 0) and (next_iter == True):
                unit = None
                seed_value = random.randint(0,1000)
                print('selection_made:', selection_made[0], flush=True)
                next_remaining_units = np.delete(arr=np.arange(num_units), obj=np.array(selected_units))            
                basename = os.path.basename(filename)
                next_iteration = int(basename.split('_')[3].strip('.npz')) + 1 # retrieves the iteration and adds 1
                next_filename = os.path.join(progress_dir,'progress_record_ITER_' + str(next_iteration)) + '.npz'
                next_lockname = next_filename + '.lock'
                next_lock = FileLock(next_lockname, timeout=lock_timeout)
                next_lock.acquire(timeout=acquire_timeout)
                
                # update so that next iter is true
                np.savez(file                = filename,
                         selected_units      = selected_units,
                         selected_losses     = selected_losses,
                         selected_accuracies = selected_accuracies,
                         selected_subperformances = selected_subperformances,
                         selections_complete = selections_complete,
                         remaining_units     = remaining_units, 
                         pending_units       = pending_units,
                         pending_clock       = pending_clock,
                         dropped_units       = dropped_units,
                         dropped_losses      = dropped_losses,
                         dropped_accuracies  = dropped_accuracies,
                         dropped_subperformances = dropped_subperformances,
                         next_iter_made      = np.array([True]),
                         selection_made      = selection_made,
                         seed                = seed,
                         conclusion_count    = conclusion_count
                        )
                try:
                    # filename changed
                    filename=next_filename
                    np.savez(file                = next_filename,
                             selected_units      = selected_units,
                             selected_losses     = selected_losses,
                             selected_accuracies = selected_accuracies,
                             selected_subperformances = selected_subperformances,
                             selections_complete = selections_complete,
                             remaining_units     = next_remaining_units, 
                             pending_units       = np.array([]).astype(int),
                             pending_clock       = np.array([]).astype(float),
                             dropped_units       = np.array([]).astype(int),
                             dropped_losses      = np.array([]).astype(float),
                             dropped_accuracies  = np.array([]).astype(float),
                             dropped_subperformances = np.array([]).astype(float),
                             next_iter_made      = np.array([False]),
                             selection_made      = np.array([False]),
                             seed                = np.array([seed_value]),
                             conclusion_count    = np.array([0])
                            )
                finally:
                    next_lock.release()
            else:
                print('next_iter has already been created!!!', flush=True)
                print('next_iter_made', next_iter_made[0], flush=True)
                unit=None
                seed_value=None
        finally:
            lock.release()
            
    return unit, seed_value, filename

def get_duplicate_ids(selected_units):
    selected_units_counts = {}
    duplicate_ids = []
    for i, unit in enumerate(selected_units):
        if unit not in selected_units_counts.keys():
            selected_units_counts[unit] = 1
        else:
            selected_units_counts[unit]+=1
            duplicate_ids.append(i)
    return duplicate_ids

def make_unique(units, losses, accuracies, subperformances):
    assert len(units) == len(losses), 'units and losses must have equal length'
    
    units = np.array(units)
    losses = np.array(losses)
    accuracies = np.array(accuracies)
    if subperformances is not None:
        subperformances = np.array(subperformances)
    
    duplicate_ids = get_duplicate_ids(units)
    if len(duplicate_ids) > 0:
        print('\nMaking Units Unique:', flush=True)
        print('--removing duplicate units:', units[duplicate_ids], flush=True)
        units = np.delete(units, duplicate_ids)
        losses = np.delete(losses, duplicate_ids)
        accuracies = np.delete(accuracies, duplicate_ids)
        if len(subperformances.shape)>1:
            subperformances = np.delete(subperformances, duplicate_ids, axis=0)
    else:
        pass
        #print('Values already_unique, no operation necessary.', flush=True)
    print(flush=True)
        
    return units, losses, accuracies, subperformances

def update_progress(filename, unit, loss, accuracy, subperformances):
    '''
    Description: appends the new unit and loss to the existing dropped units
    '''
    
    assert(unit is not None), 'unit must be an integer valued scalar'
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:

        # load data
        progress_record     = load_record(filename)
        selected_units      = progress_record['selected_units']
        selected_losses     = progress_record['selected_losses']
        selected_accuracies = progress_record['selected_accuracies']
        selected_subperformances = progress_record['selected_subperformances']
        selections_complete = progress_record['selections_complete']
        remaining_units     = progress_record['remaining_units']
        pending_units       = progress_record['pending_units']
        pending_clock       = progress_record['pending_clock']
        dropped_units       = progress_record['dropped_units']
        dropped_losses      = progress_record['dropped_losses']
        dropped_accuracies  = progress_record['dropped_accuracies']
        dropped_subperformances = progress_record['dropped_subperformances']
        next_iter_made      = progress_record['next_iter_made']
        selection_made      = progress_record['selection_made']
        seed                = progress_record['seed']
        conclusion_count    = progress_record['conclusion_count']
        
        # update drops
        dropped_units = np.append(dropped_units, unit)
        dropped_losses = np.append(dropped_losses, loss)
        dropped_accuracies = np.append(dropped_accuracies, accuracy)
        if subperformances is not None:
            if len(dropped_subperformances.shape) == 1:
                dropped_subperformances = subperformances
            elif len(dropped_subperformances.shape) == 2:
                dropped_subperformances = np.stack((dropped_subperformances, subperformances))
            elif len(dropped_subperformances.shape) == 3:
                dropped_subperformances = np.concatenate((dropped_subperformances, [subperformances])) 
        
        # maked unique
        if len(dropped_units) > 2:
            dropped_units, dropped_losses, dropped_accuracies, dropped_subperformances = make_unique(units=dropped_units, 
                                                                                 losses=dropped_losses, 
                                                                                 accuracies=dropped_accuracies,
                                                                                 subperformances=dropped_subperformances)
        
        # remove unit from pending 
        pending_removals = np.where(pending_units==unit)[0]
        pending_units    = np.delete(pending_units, pending_removals)
        pending_clock    = np.delete(pending_clock, pending_removals)

        # overwrite file
        np.savez(file                = filename,
                 selected_units      = selected_units,
                 selected_losses     = selected_losses,
                 selected_accuracies = selected_accuracies,
                 selected_subperformances = selected_subperformances,
                 selections_complete = selections_complete,
                 remaining_units     = remaining_units, 
                 pending_units       = pending_units,
                 pending_clock       = pending_clock,
                 dropped_units       = dropped_units,
                 dropped_losses      = dropped_losses,
                 dropped_accuracies  = dropped_accuracies,
                 dropped_subperformances = dropped_subperformances,
                 next_iter_made      = next_iter_made,
                 selection_made      = selection_made,
                 seed                = seed,
                 conclusion_count    = conclusion_count)
    finally:
        lock.release()

    return None

def conclude_progress(filename):
    '''
    Completes the progress record
    '''
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
    
        # load data
        progress_record     = load_record(filename)
        selected_units      = progress_record['selected_units']
        selected_losses     = progress_record['selected_losses']
        selected_accuracies = progress_record['selected_accuracies']
        selected_subperformances = progress_record['selected_subperformances']
        selections_complete = progress_record['selections_complete']
        remaining_units     = progress_record['remaining_units']
        pending_units       = progress_record['pending_units']
        pending_clock       = progress_record['pending_clock']
        dropped_units       = progress_record['dropped_units']
        dropped_losses      = progress_record['dropped_losses']
        dropped_accuracies  = progress_record['dropped_accuracies']
        dropped_subperformances = progress_record['dropped_subperformances']
        next_iter_made      = progress_record['next_iter_made']
        seed                = progress_record['seed']
        conclusion_count    = np.array([progress_record['conclusion_count'][0] + 1])
        selection_made      = np.array([True])

        # overwrite file
        np.savez(file                = filename,
                 selected_units      = selected_units,
                 selected_losses     = selected_losses,
                 selected_accuracies = selected_accuracies,
                 selected_subperformances = selected_subperformances,
                 selections_complete = selections_complete,
                 remaining_units     = remaining_units, 
                 pending_units       = pending_units,
                 pending_clock       = pending_clock,
                 dropped_units       = dropped_units,
                 dropped_losses      = dropped_losses,
                 dropped_accuracies  = dropped_accuracies,
                 dropped_subperformances = dropped_subperformances,
                 next_iter_made      = next_iter_made,
                 selection_made      = selection_made,
                 seed                = seed,
                 conclusion_count    = conclusion_count)
    finally:
        lock.release()
    
    return conclusion_count[0]

def get_pending_duration(pending_clock): 
    '''
    Description: returns the duration for each pending unit basedo on the current_timestamp and 
    the timestamps in pending_clock
    '''
    current_timestamp = time.time() 
    pending_durations = []
    for i, timestamp in enumerate(pending_clock):
        duration = (current_timestamp-timestamp)/60
        pending_durations.append(duration)
    return pending_durations

def restore_stagnant_pending_units(filename, duration_threshold=60):
    '''
    Description: Retreives progress record to restore any pending units that have been 
    pending for durations over duration_threshold. If found moves them back to remaining_units 
    and deletes them from pending_units and their corresponding pending_clock
    
    Inputs:
        filename: path to prorgress record file
        duration_threshold: considers a unit (element) in pending_units to be stagnant if 
                            its correspoinding timestamp in pending_clock yields a duration longer
                            than duration_threshold
    '''
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        progress_record     = load_record(filename)
        selected_units      = progress_record['selected_units']
        selected_losses     = progress_record['selected_losses']
        selected_accuracies = progress_record['selected_accuracies']
        selected_subperformances = progress_record['selected_subperformances']
        selections_complete = progress_record['selections_complete']
        remaining_units     = progress_record['remaining_units']
        pending_units       = progress_record['pending_units']
        pending_clock       = progress_record['pending_clock']
        dropped_units       = progress_record['dropped_units']
        dropped_losses      = progress_record['dropped_losses']
        dropped_accuracies  = progress_record['dropped_accuracies']
        dropped_subperformances = progress_record['dropped_subperformances']
        next_iter_made      = progress_record['next_iter_made']
        selection_made      = progress_record['selection_made']
        seed                = progress_record['seed']
        conclusion_count    = progress_record['conclusion_count']

        # find stagnant units and remove from pending and into remaining
        pending_durations = get_pending_duration(pending_clock)
        stagnant_pending_unit_indexes = np.array([u for (u, dur) in enumerate(pending_durations) if dur>duration_threshold])
        if len(stagnant_pending_unit_indexes) ==0:
            print('--  No stagnant pending units found.', flush=True)
            return None
        print('--  Found',len(stagnant_pending_unit_indexes), 'stagnant pending units. Restoring now.', flush=True)
        stagnant_pending_units = pending_units[stagnant_pending_unit_indexes]

        # updates
        pending_units = np.delete(pending_units, stagnant_pending_unit_indexes)
        pending_clock = np.delete(pending_clock, stagnant_pending_unit_indexes)
        remaining_units = np.sort(np.unique(np.append(remaining_units, stagnant_pending_units)))

        # overwrite file
        np.savez(file                = filename,
                 selected_units      = selected_units,
                 selected_losses     = selected_losses,
                 selected_accuracies = selected_accuracies,
                 selected_subperformances = selected_subperformances,
                 selections_complete = selections_complete,
                 remaining_units     = remaining_units, # updating
                 pending_units       = pending_units,   # updating
                 pending_clock       = pending_clock,   # updating
                 dropped_units       = dropped_units,
                 dropped_losses      = dropped_losses,
                 dropped_accuracies  = dropped_accuracies,
                 dropped_subperformances = dropped_subperformances,
                 next_iter_made      = next_iter_made,
                 selection_made      = selection_made,
                 seed                = seed,
                 conclusion_count    = conclusion_count)
    finally:
        lock.release()

    return None
    
def get_progress(filename):
    '''
    Description: retrieves the the dropped units and corresponding losses
    '''
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:  
        # load progress record
        progress_record  = load_record(filename)
        remaining_units  = progress_record['remaining_units']
        pending_units    = progress_record['pending_units']
        dropped_units    = progress_record['dropped_units']
        dropped_losses      = progress_record['dropped_losses']
        dropped_accuracies  = progress_record['dropped_accuracies']
        dropped_subperformances = progress_record['dropped_subperformances']
        next_iter_made   = progress_record['next_iter_made'][0]
        selection_made   = progress_record['selection_made'][0]
    finally:
        lock.release()
    
    return dropped_units, dropped_losses, dropped_accuracies, dropped_subperformances, pending_units, next_iter_made, selection_made, remaining_units

def get_selections(filename):
    
    if filename is None:
        selected_units      = np.array([]).astype(int)
        selected_losses     = np.array([]).astype(float)
        selected_accuracies = np.array([]).astype(float)
        selected_subperformances = np.array([]).astype(float)
        selections_complete = False
    
    else:
        lockname = filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            # load progress record
            progress_record     = load_record(filename)
            selected_units      = progress_record['selected_units']
            selected_losses     = progress_record['selected_losses']
            selected_accuracies = progress_record['selected_accuracies']
            selected_subperformances = progress_record['selected_subperformances']
            selections_complete = progress_record['selections_complete'][0]
        finally:
            lock.release()
        
    return selected_units, selected_losses, selected_accuracies, selected_subperformances, selections_complete
    

def update_selections(filename, new_selected_units, new_selected_losses, new_selected_accuracies, new_selected_subperformances):
    '''
    Description: appends the new selections to existing selections
    '''
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        progress_record     = load_record(filename)
        selected_units      = progress_record['selected_units']
        selected_losses     = progress_record['selected_losses']
        selected_accuracies = progress_record['selected_accuracies']
        selected_subperformances = progress_record['selected_subperformances']
        selections_complete = progress_record['selections_complete']
        remaining_units     = progress_record['remaining_units']
        pending_units       = progress_record['pending_units']
        pending_clock       = progress_record['pending_clock']
        dropped_units       = progress_record['dropped_units']
        dropped_losses      = progress_record['dropped_losses']
        dropped_accuracies  = progress_record['dropped_accuracies']
        dropped_subperformances  = progress_record['dropped_subperformances']
        next_iter_made      = progress_record['next_iter_made']
        selection_made      = progress_record['selection_made']
        seed                = progress_record['seed']
        conclusion_count    = progress_record['conclusion_count']

        # update selections
        selected_units = np.append(selected_units, new_selected_units)
        selected_losses = np.append(selected_losses, new_selected_losses)
        selected_accuracies = np.append(selected_accuracies, new_selected_accuracies)
        
        if new_selected_subperformances is not None:
            if len(selected_subperformances.shape) == 1:
                selected_subperformances = new_selected_subperformances
            elif len(selected_subperformances.shape) == 2:
                selected_subperformances = np.stack((selected_subperformances, new_selected_subperformances))
            elif len(selected_subperformances.shape) == 3:
                selected_subperformances = np.concatenate((selected_subperformances, [new_selected_subperformances]))
        
        #make unique    
        if len(selected_units) > 2:
            selected_units,selected_losses,selected_accuracies,selected_subperformances=make_unique(units=selected_units, 
                                                                                                    losses=selected_losses, 
                                                                                                 accuracies=selected_accuracies, 
                                                                                       subperformances=selected_subperformances)

        # overwrite with new udpates
        np.savez(file                = filename,
                 selected_units      = selected_units,                 # updating 
                 selected_losses     = selected_losses,                # updating 
                 selected_accuracies = selected_accuracies,            # updating
                 selected_subperformances = selected_subperformances,  # updating
                 selections_complete = selections_complete,
                 remaining_units     = remaining_units, 
                 pending_units       = pending_units,   
                 pending_clock       = pending_clock,   
                 dropped_units       = dropped_units,
                 dropped_losses      = dropped_losses,
                 dropped_accuracies  = dropped_accuracies,
                 dropped_subperformances = dropped_subperformances,
                 next_iter_made      = next_iter_made,
                 selection_made      = selection_made,
                 seed                = seed,
                 conclusion_count    = conclusion_count)
    finally:
        lock.release()
    
    return None

def conclude_selections(progress_filename, selections_dir):
    '''
    Description: concludes the selections record
    '''
    
    progress_lockname = progress_filename + '.lock'
    progress_lock = FileLock(progress_lockname, timeout=lock_timeout)
    progress_lock.acquire(timeout=acquire_timeout)
    try:
        progress_record     = load_record(progress_filename)
        selected_units      = progress_record['selected_units']
        selected_losses     = progress_record['selected_losses']
        selected_accuracies = progress_record['selected_accuracies']
        selected_subperformances = progress_record['selected_subperformances']
        remaining_units     = progress_record['remaining_units']
        pending_units       = progress_record['pending_units']
        pending_clock       = progress_record['pending_clock']
        dropped_units       = progress_record['dropped_units']
        dropped_losses      = progress_record['dropped_losses']
        dropped_accuracies  = progress_record['dropped_accuracies']
        dropped_subperformances = progress_record['dropped_subperformances']
        next_iter_made      = progress_record['next_iter_made']
        selection_made      = progress_record['selection_made']
        seed                = progress_record['seed']
        conclusion_count    = progress_record['conclusion_count']
        
        # conclude or make selections_complete
        selections_complete = np.array([True])    

        # overwrite with new udpates
        np.savez(file                = progress_filename,
                 selected_units      = selected_units,  
                 selected_losses     = selected_losses,               
                 selected_accuracies = selected_accuracies,           
                 selected_subperformances = selected_subperformances,   
                 selections_complete = selections_complete,# updating  
                 remaining_units     = remaining_units, 
                 pending_units       = pending_units,   
                 pending_clock       = pending_clock,   
                 dropped_units       = dropped_units,
                 dropped_losses      = dropped_losses,
                 dropped_accuracies  = dropped_accuracies,
                 dropped_subperformances = dropped_subperformances,
                 next_iter_made      = next_iter_made,
                 selection_made      = selection_made,
                 seed                = seed,
                 conclusion_count    = conclusion_count)
    finally:
        progress_lock.release()
        
        
    # write selections record finally!
    if not os.path.exists(selections_dir):
            os.makedirs(selections_dir)
    selections_filename = os.path.join(selections_dir, 'selections_record.npz')
    selections_lockname = selections_filename + '.lock'
    selections_lock = FileLock(selections_lockname, timeout=lock_timeout)
    selections_lock.acquire(timeout=acquire_timeout)
    try:
        np.savez(file                = selections_filename,
                 selected_units      = selected_units,  
                 selected_losses     = selected_losses,               
                 selected_accuracies = selected_accuracies,           
                 selected_subperformances = selected_subperformances,   
                 selections_complete = selections_complete)
    finally:
        selections_lock.release()
        
    return None

def get_drop_loss(selected_units, candidate_units, validator, weight, bias, cache, ngpus, max_batches, 
                  seed, subgroups_file=None):
    
    # drop units 
    # -------------------------
    drop_units = np.append(selected_units,candidate_units)
    for unit in drop_units:
        weight[unit] = 0.0
        if bias is not None:
            bias[unit]   = 0.0
        
    # get loss on prediction
    # -------------------------
    y_true, y_pred, _, _, loss, _ = helper.predict(model=validator.model, 
                                         data_loader=validator.data_loader, 
                                         ngpus=ngpus, 
                                         topk=1,
                                         max_batches=max_batches,
                                         reduce_loss=True,
                                         notebook=False,
                                         seed=seed,
                                         reduction='none')
    subperformances=None
    if subgroups_file is not None:
        subperformances = helper.get_subgroup_performances(subgroups_file=subgroups_file, 
                                                           y_true=y_true, 
                                                           y_pred=y_pred, 
                                                           losses=loss)
    
    # replace unit 
    # -------------------------
    for unit in drop_units:
        weight[unit] = torch.from_numpy(cache['W'][unit])
        if bias is not None:
            bias[unit]   = torch.from_numpy(cache['b'][unit]) 
    
    y_pred = np.squeeze(y_pred)
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    return np.mean(loss), accuracy, subperformances, y_pred, y_true

def get_base_performance(validator, ngpus, max_batches, seed, subgroups_file=None):
    
    # BASE PERFORMANCE
    # ---------------------------
    y_true, y_pred, _, _, loss, _ = helper.predict(model=validator.model, 
                                         data_loader=validator.data_loader, 
                                         ngpus=ngpus, 
                                         topk=1,
                                         max_batches=max_batches,
                                         reduce_loss=True,
                                         notebook=False,
                                         seed=seed,
                                         reduction='none')
    
    subperformances=None
    if subgroups_file is not None:
        subperformances = helper.get_subgroup_performances(subgroups_file=subgroups_file, 
                                                           y_true=y_true, 
                                                           y_pred=y_pred, 
                                                           losses=loss)
        
    
    y_pred = np.squeeze(y_pred)
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    return np.mean(loss), accuracy, subperformances, y_pred, y_true

def restore_missing_units(num_units, progress_filename, restore=False):
    '''
    Description: sometimes all units are not returned/skipped and waiting for jobs to return their units gets stuck 
    in an infinite loop. To avoid this the finally job has the option to restore any missing units in order to not wait forever.
    This function creates this functionality. It looks through the dropped_units and if there are any missing, it will restore
    these to the remaining units and continue lesioning. 
    
    Operates under the assumption that there are no pending units (all units accounted for are selections and dropped units)
    '''
    
    progress_lockname = progress_filename + '.lock'
    progress_lock = FileLock(progress_lockname, timeout=lock_timeout)
    progress_lock.acquire(timeout=acquire_timeout)
    try:
        # progress
        progress_record     = load_record(filename=progress_filename)   
        selected_units      = progress_record['selected_units']
        selected_losses     = progress_record['selected_losses']
        selected_accuracies = progress_record['selected_accuracies']
        selected_subperformances = progress_record['selected_subperformances']
        selections_complete = progress_record['selections_complete']
        remaining_units     = progress_record['remaining_units']
        pending_units       = progress_record['pending_units']
        pending_clock       = progress_record['pending_clock']
        dropped_units       = progress_record['dropped_units']
        dropped_losses      = progress_record['dropped_losses']
        dropped_accuracies  = progress_record['dropped_accuracies']
        dropped_subperformances = progress_record['dropped_subperformances']
        next_iter_made      = progress_record['next_iter_made']
        selection_made      = progress_record['selection_made']
        seed_value          = progress_record['seed']
        conclusion_count    = progress_record['conclusion_count']
        
        # restore missing ones but also removing duplicates
        dropped_units, dropped_losses, dropped_accuracies, dropped_subperformances = make_unique(units=dropped_units, 
                                                                                losses=dropped_losses,
                                                                                accuracies=dropped_accuracies,
                                                                                subperformances=dropped_subperformances)

        # print current
        print('\nBefore Restoring Missing Units:',flush=True)
        print('-------------------------------',flush=True)
        print('num selected_units:', len(selected_units),flush=True)
        print('num remaining_units:', len(remaining_units),flush=True)
        print('num pending_units:', len(pending_units),flush=True) 
        print('num pending_clock:', len(pending_clock),flush=True)
        print('num dropped_units:', len(dropped_units),flush=True)
        print('num dropped_losses:', len(dropped_losses),flush=True)
        print('next_iter_made:', next_iter_made,flush=True)
        print('selection_made:', selection_made,flush=True)
        print('seed_value:', seed_value,flush=True)
        print('conclusion_count:', conclusion_count,flush=True)
        print('selections_complete:', selections_complete, flush=True)
        print(flush=True)

        # get missing units
        missed_units = []
        for u in range(num_units):
            if (u not in dropped_units) and (u not in selected_units):
                missed_units.append(u)
        remaining_units = np.array(missed_units).astype(int)
        conclusion_count = np.array([0]).astype(int)
        selection_made = np.array([False])
        next_iter_made = np.array([False])
        print('missed_units:', missed_units, flush=True)
        print(flush=True)

        if restore:
            # overwrite with new udpates
            np.savez(file                = progress_filename,
                     selected_units      = selected_units,  
                     selected_losses     = selected_losses,               
                     selected_accuracies = selected_accuracies,           
                     selected_subperformances = selected_subperformances,   
                     selections_complete = selections_complete, 
                     remaining_units     = remaining_units, 
                     pending_units       = pending_units,   
                     pending_clock       = pending_clock,   
                     dropped_units       = dropped_units,
                     dropped_losses      = dropped_losses,
                     dropped_accuracies  = dropped_accuracies,
                     dropped_subperformances = dropped_subperformances,
                     next_iter_made      = next_iter_made,
                     selection_made      = selection_made,
                     seed                = seed,
                     conclusion_count    = conclusion_count)
    finally:
        progress_lock.release()
        
        # print current
        print('\nAfter Restoring Missing Units:')
        print('-------------------------------',flush=True)
        print('num selected_units:', len(selected_units),flush=True)
        print('num remaining_units:', len(remaining_units))
        print('num pending_units:', len(pending_units),flush=True)
        print('num pending_clock:', len(pending_clock),flush=True)
        print('num dropped_units:', len(dropped_units))
        print('num dropped_losses:', len(dropped_losses))
        print('next_iter_made:', next_iter_made)
        print('selection_made:', selection_made)
        print('seed_value:', seed_value)
        print('conclusion_count:', conclusion_count)
        print('selections_complete:', selections_complete, flush=True)
        
    return None
    
def greedy_lesion_layer(validator,index,layerMap,selections_dir,progress_dir,predictions_filename,predictions_key,sort_task, 
                        lesions_filename,iterator_seed,greedy_p=0.0,group_p=0.0,ngpus=0,max_batches=None, approx_method=None,
                        subgroups_file=None):
    '''
    Description:
        Runs greedy lesion on a specified layer. 
    Inputs:
        validator - validator object, see utils
        index     - index of parameter (so 0 will be 0th pair of weight and bias, and so on)
        layerMap  - layerMap object, see utils
        p         - percent of units to select
        group_p    - percent that determines number of selections at each iteration
        
    '''
    
    assert( (group_p >= 0.0) and (group_p <= 1.0) ), '0.0 <= group_p <= 1 does not hold'
    assert( (greedy_p >= 0.0) and (greedy_p <= 1.0) ), '0.0 <= greedy_p <= 1 does not hold'
    
    print('Starting Greedy Layer Lesion on Train Data',flush=True)
    print('------------------------------------------',flush=True)
    print(flush=True)
    
    print('Using Methods:')
    print('greedy_p = ',greedy_p)
    print('Approximation Method:', approx_method)
    print(flush=True)
    
    
    cache = {'W':{}, 'b':{}}
    weight, bias = helper.getWeightandBias(validator.model, layerMap, index)
    num_units = weight.shape[0]
    layer = layerMap['ParamGroupIndex2LayerIndex'][index]
    layerType = layerMap['ParamGroupIndex2LayerType'][index]
    print("{0:<40}: {1:}".format('validator.name',validator.name),flush=True)
    print("{0:<40}: {1:}".format('index',index),flush=True)
    print("{0:<40}: {1:}".format('layer',layer),flush=True)
    print("{0:<40}: {1:}".format('layerType',layerType),flush=True)
    print("{0:<40}: {1:}".format('num_units',num_units),flush=True)
    print(flush=True)
    num_units_to_drop = np.round(num_units*greedy_p).astype(int)
    if group_p <= (1.0/num_units): # effectively turns off grouping
        group_p = (1.0/num_units) +  1e-6 
    group_size = int(num_units*group_p)
    
       
    # cache all units
    # -------------------------------------------------------
    for unit in range(num_units):
        cache['W'][unit] = weight[unit].detach().cpu().numpy()
        if bias is not None:
            cache['b'][unit] = bias[unit].detach().cpu().numpy()
    
    # run until have reached the num_units required to select (if greedy_p=0 a break statement will occur)
    # ---------------------------------------------------------------------------------------------------
    
    progress_filename = get_latest_npz_filename(dir=progress_dir)
    selections = get_selections(filename=progress_filename)
    selected_units, selected_losses, selected_accuracies, selected_subperformances, _ = selections  # if none returns empty arrays
    
    seedTracker = []
        
    linear_complete=False
    greedy_complete=False
    while (selected_units.shape[0] <= num_units_to_drop):  
        progress_filename = get_latest_npz_filename(dir=progress_dir)
        selections = get_selections(filename=progress_filename)
        selected_units, selected_losses, selected_accuracies, selected_subperformances, _ = selections
        notNone_unit_count  = 0
        # run until a break statement (breaks out of nested while loop)
        while True:
            
            # filename changes only when created, otherwise returns same filename
            unit, seed_value, progress_filename = generate_unit(filename=progress_filename, 
                                                                num_units=num_units, 
                                                                progress_dir=progress_dir)
            
            if (seed_value not in seedTracker) and (unit is not None):
                seedTracker.append(seed_value)
                print('Getting Base Performance...', flush=True)
                base_loss, base_accuracy, base_subperformances, _, _ = get_base_performance(validator = validator, 
                                                                                            ngpus     =ngpus, 
                                                                                            max_batches=max_batches,
                                                                                            seed       =seed_value,
                                                                                            subgroups_file=subgroups_file)
                
                print("{0:<32}: ({1:.3f}, {2:.3f}, {3:})".format('(base loss, base accuracy, seed)\
                ',base_loss, base_accuracy*100.0, seed_value),flush=True)
            
                if base_subperformances is not None:
                    for i, item in enumerate(base_subperformances):
                        print("-----subgroup{0:<8}  (base loss, base accuracy) = ({1:.3f}, {2:.3f})\
                        ".format(':',item[0], item[1]*100.0),flush=True)  

            if (unit is not None) and (notNone_unit_count==0):
                notNone_unit_count+=1
                print('\n\n---------------------------------------------------------------------',flush=True)
                print('/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/',flush=True)                
                print('progress_filename:', progress_filename, flush=True)
                print('\nselected_units:',  selected_units,flush=True)
                print('\nselected_losses:', selected_losses,flush=True)
                print('\nLosses conditioned on ' + str(len(selected_units)) + ' selected units w/ seed value=' + str(seed_value),flush=True) 
                print('---------------------------------------------------------------------',flush=True)  
         
            if unit is None:
                if notNone_unit_count > 0:
                    print('Unit == None', flush=True)
                    print('\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/',flush=True) 
                    print('--------------------------------------------------------------------\n',flush=True)
                time.sleep(15)
                break
                
            if iterator_seed is None:
                seed_value = None
            elif iterator_seed == "fixed":
                seed_value = fixed_seed_value
            elif iterator_seed == "selection":
                pass # keep the seed_value from previous generate_unit call
            
            loss, accuracy, subperformances, y_pred, y_true = get_drop_loss(selected_units   =selected_units, 
                                                                         candidate_units  =unit, 
                                                                         validator        =validator, 
                                                                         weight           =weight,
                                                                         bias             =bias,
                                                                         cache            =cache,
                                                                         ngpus            =ngpus, 
                                                                         max_batches      =max_batches,
                                                                         seed             =seed_value,
                                                                         subgroups_file   =subgroups_file) 
            # if loss is matrix, average the loss
 
            print("{0:<32}: ({1:.3f}, {2:.3f})".format('(loss, accuracy) @unit ' + str(unit),loss, accuracy*100.0),flush=True)
            
            if subperformances is not None:
                for i, item in enumerate(subperformances):
                    loss_delta = -((base_subperformances[i][0] - item[0])/base_subperformances[i][0])*100
                    accuracy_delta = -((base_subperformances[i][1] - item[1])/base_subperformances[i][1])*100
                    print("-----subgroup{0:<2}  (loss, %delta) = ({1:.3f}, {2:.3f}), (accuracy, %delta) = ({3:.3f}, {4:.3f})".format(':',item[0], loss_delta, item[1]*100, accuracy_delta),flush=True)  
                subperformances = np.array(subperformances).astype(float)
    
            
            update_progress(filename=progress_filename, 
                            unit=unit, 
                            loss=loss, 
                            accuracy=accuracy, 
                            subperformances=subperformances)
        
            if (predictions_filename is not None) and (predictions_key is not None):
                key = os.path.join(predictions_key, 'ITER_' + str(len(selected_units)), 'LESION_UNIT_' + str(unit))
                obj = {'y_true': y_true.tolist(), 
                       'y_pred':y_pred.tolist(), 
                       'lesioned_units':selected_units.tolist()}
                write_obj2jsonl(filename=predictions_filename, writer_method='a', key=key, obj=obj)
        
        # should force loop until all candidates have been computed
        progress = get_progress(filename=progress_filename)
        candidate_units, candidate_losses, candidate_accuracies, candidate_subperformances, pending_units, next_iter_made, selection_made, remaining_units = progress
        num_remaining = len(remaining_units)
        num_pending = len(pending_units)
        missed_units = [u for u in range(num_units) if (u not in candidate_units) and (u not in selected_units)]
        num_missed_units = len(missed_units)
        
        if (num_remaining>0) or (num_pending>0):
            if num_remaining > 0:
                print(flush=True)
                print(num_remaining,  'remaining units. Continuing lesion for current interation.', flush=True)
            if num_pending > 0:
                print(flush=True)
                print(num_pending, 'pending units. Continuing lesion for current iteration.', flush=True)
                # check for stagnant pending units
                minutes = 15
                restore_stagnant_pending_units(filename=progress_filename, duration_threshold=minutes)
                
            time.sleep(15)
            selections = get_selections(filename=progress_filename)
            selected_units, selected_losses, selected_accuracies, selected_subperformances, _ = selections
            continue
             
        elif (num_remaining==0) and (num_pending==0) and (num_missed_units > 0):
            print('\nMissing Units found. Restoring missing units.', flush=True)
            restore_missing_units(num_units=num_units, 
                                  progress_filename=progress_filename,
                                  restore=True)
            time.sleep(15)
            selections = get_selections(filename=progress_filename)
            selected_units, selected_losses, selected_accuracies, selected_subperformances, _ = selections
            continue
       # ------------------------------------------
    
    
    
        # proceed with the following assumptions:  #
        # ---------------------------------------  #
        #    * num_remaining=0:                    #
        #    * num_pending=0:                      #
        #    * all_units_accounted_for=num_units   #
        
        
        # make selection if hasn't been done yet (usually the first job will get to this first)
        if (selection_made==False): 
            conclusion_count = conclude_progress(filename=progress_filename) 
            print('conclusion_count:', conclusion_count)
            
            if (conclusion_count > 1):
                print('ignoring conclusion test')
                time.sleep(30)
                continue
            
            # selections
            # ------------------------
            sorted_indexes      = np.argsort(candidate_losses)[::-1]
            new_selected_units  = candidate_units[sorted_indexes[:group_size]]
            new_selected_losses = candidate_losses[sorted_indexes[:group_size]]
            new_selected_accuracies = candidate_accuracies[sorted_indexes[:group_size]]
            
            new_selected_subperformances = None
            if (len(candidate_subperformances.shape))>2:
                new_selected_subperformances = candidate_subperformances[sorted_indexes[:group_size],:,:]
            
            if greedy_p>0:   

                # TEMP
                #selected_units_before, selected_losses_before, _ = get_selections(filename=progress_filename)
                #print('\nprogress_filename:',             progress_filename, flush=True)
                #print('\nselected_units_before_update:',  selected_units_before, flush=True)
                #print('\nselected_losses_before_update:', selected_losses_before,flush=True)
                # -------

                update_selections(filename                    =progress_filename, 
                                  new_selected_units          =new_selected_units, 
                                  new_selected_losses         =new_selected_losses,
                                  new_selected_accuracies     =new_selected_accuracies,
                                  new_selected_subperformances=new_selected_subperformances)
               
                selections = get_selections(filename=progress_filename)
                selected_units, selected_losses, selected_accuracies, selected_subperformances, _ = selections 
                
                # TEMP
                #print('\nprogress_filename:',            progress_filename, flush=True)
                #print('\nselected_units_after_update:',  selected_units, flush=True)
                #print('\nselected_losses_after_update:', selected_losses,flush=True)
                # -------

                generate_unit(filename=progress_filename,
                              num_units=num_units, 
                              progress_dir=progress_dir,
                              next_iter=True)
                # TEMP
                #selected_units, selected_losses, _ = get_selections(filename=progress_filename_temp)  
                #print('\nprogress_filename_temp:',           progress_filename_temp, flush=True)
                #print('\nselected_units_in_new_iter_file:',  selected_units,  flush=True)
                #print('\nselected_losses_in_new_iter_file:', selected_losses, flush=True)
                # -------
                
              
            else:
                # kill process if greedy_p=0 
                # -------------------------
                sorted_candidates   = np.argsort(candidate_losses)[::-1]
                new_selected_units  = candidate_units[sorted_candidates]
                new_selected_losses = candidate_losses[sorted_candidates]
                new_selected_accuracies = candidate_accuracies[sorted_candidates]
                
                new_selected_subperformances = None
                if (len(candidate_subperformances.shape))>2:
                    new_selected_subperformances = candidate_subperformances[sorted_candidates,:,:]
                
                # store selections
                update_selections(filename                    =progress_filename, 
                                  new_selected_units          =new_selected_units, 
                                  new_selected_losses         =new_selected_losses,
                                  new_selected_accuracies     =new_selected_accuracies,
                                  new_selected_subperformances=new_selected_subperformances)
                linear_complete=True
                break
             
        
        elif (selection_made==True and greedy_p==0):
            break
                       
        elif (selection_made==True and greedy_p>0):
            sleep_time=15
            total_wait = sleep_time
            time.sleep(sleep_time)
            next_iter_made = get_progress(filename=progress_filename)[5]
            selections = get_selections(filename=progress_filename)
            selected_units, selected_losses, selected_accuracies, selected_subperformances, selections_complete = selections
            while (next_iter_made==False and selections_complete==False):
                print('Waiting for selection to complete...', flush=True)
                print('\n--total waiting time:', total_wait, 'seconds', flush=True)
                total_wait += sleep_time
                time.sleep(sleep_time)
                next_iter_made = get_progress(filename=progress_filename)[5]
                selections = get_selections(filename=progress_filename)
                selected_units, selected_losses, selected_accuracies, selected_subperformances, selections_complete = selections
            continue
                
       
    if greedy_p>0.0:
        selections = get_selections(filename=progress_filename)
        selected_units, selected_losses, selected_accuracies, selected_subperformances, selections_complete = selections
        greedy_complete=(selected_units.shape[0] > num_units_to_drop)
        if greedy_complete and selections_complete==False:
            print('Writing selections record!', flush=True)
            conclude_selections(progress_filename=progress_filename, selections_dir=selections_dir)
    else:
        selections = get_selections(filename=progress_filename)
        selected_units, selected_losses, selected_accuracies, selected_subperformances, selections_complete = selections
        if linear_complete and selections_complete==False:
            print('Writing selections record!', flush=True)
            conclude_selections(progress_filename=progress_filename, selections_dir=selections_dir)
        
    print(flush=True)
    print('selected_units:', selected_units,flush=True)
    print('selected_losses:', selected_losses,flush=True)
    print(flush=True)
    return None


    
def run_lesion():
    # FLAGS 
    # ------------------------------------------------------
    parser = argparse.ArgumentParser(description='Lesion Filters')
    parser.add_argument('--config_file',          default=None,    type=str,  help='path to config file')
    parser.add_argument('--param_group_index',    default=0,       type=int,  help='param weight and bias group index')
    parser.add_argument('--greedy_p',             default=0.0,     type=float,  help='0.0 => linear lesion')
    parser.add_argument('--group_p',              default=0.0,     type=float,  help='0.0 => single-unit greedy version')
    parser.add_argument('--shuffle',              default="False", type=str,  help='shuffle data in dataloader')
    parser.add_argument('--random',               default="False", type=str,  help='random dropping')
    parser.add_argument('--ngpus',                default=1,            type=int,   help='number of gpus to use')
    parser.add_argument('--batch_size',           default=128,          type=int,   help='batch size')
    parser.add_argument('--max_batches',          default=5,            type=int,   help='batches to run on train losses')
    parser.add_argument('--workers',              default=4,            type=int,   help='read and write workers')
    parser.add_argument('--sort_task_index',      default=0,            type=int,   help='sort_task_index + 1')
    parser.add_argument('--nonsort_task_index',   default=1,            type=int,   help='nonsort_task_index + 1')
    parser.add_argument('--restore_epoch',        default=-1,           type=int,   help='epoch to restore from')
    parser.add_argument('--lesion_name',          default='',           type=str,   help='save suffix identifier')
    parser.add_argument('--read_suffix',          default='',           type=str,   help='read suffix identifier')
    parser.add_argument('--lesions_dir',          default='./lesions/', type=str,   help='where to read the losses from')
    parser.add_argument('--evaluate',              default="false",     type=str,  help='additionally run evaluation')
    parser.add_argument('--iterator_seed', choices=['fixed', 'selection', None], default=None, type=str, help='seed type')
    parser.add_argument('--read_seed',            default=None,         type=int,  help='seed type')
    parser.add_argument('--maxout',               default="False",      type=str, help='read all data and then shuffle')
    parser.add_argument('--randomize_classes',     default="False",     type=str, help='whether to randomly mix classes')
    parser.add_argument('--randomize_classes_seed', default=0,         type=int, help='how to mix the classes')
    parser.add_argument('--write_predictions',    default="False",     type=str, help='write y_true and y_pred for lesions')
    parser.add_argument('--subgroups_file',       default=None, type=str, help='array file for categ2subgroup')
    
    
    FLAGS, FIRE_FLAGS = parser.parse_known_args()
    
    # converts strings to corresponding boolean values
    FLAGS.shuffle = True if (FLAGS.shuffle == 'True') else False
    FLAGS.random = True if (FLAGS.random == 'True') else False
    FLAGS.evaluate = True if (FLAGS.evaluate == 'True') else False
    FLAGS.maxout = True if (FLAGS.maxout == 'True') else False
    FLAGS.randomize_classes = True if (FLAGS.randomize_classes == 'True') else False
    FLAGS.write_predictions = True if (FLAGS.write_predictions == 'True') else False
    
    if FLAGS.ngpus > 0:
        torch.backends.cudnn.benchmark = True
              
    # Get Model and tasks
    # --------------------------
    #config = helper.Config(config_file=FLAGS.config_file)
    #model, ckpt_data = config.get_model(pretrained=True, ngpus=FLAGS.ngpus, dataParallel=True, epoch=FLAGS.restore_epoch)
    
    # Get Config File
    with open(FLAGS.config_file) as f:
        config = json.load(f)
        print("Config File:\n-------------")
        pp.pprint(config)
    
    model = get_model(config_file=FLAGS.config_file, config=config, ngpus=FLAGS.ngpus, dataParallel=True, pretrained=True)
    
    layerMap = helper.getLayerMapping(model)
        
    config["batch_size"] = FLAGS.batch_size
    tasks = list(config["max_valid_samples"].keys())
    tasks.sort()
    sort_task = tasks[FLAGS.sort_task_index]
    nonsort_task = tasks[FLAGS.nonsort_task_index]
    print('sort_task:', sort_task)
    print('nonsort_task:', nonsort_task)
    
    ###########################################################################
    ###########################################################################
    #                               LESION                                    #
    ###########################################################################
    ###########################################################################
    
    # validator sort_task train data
    # --------------------------
    with open(FLAGS.config_file) as f:
        config_sort_task_train_data = json.load(f)
    #config_sort_task_train_data = helper.Config(config_file=FLAGS.config_file)
    config_sort_task_train_data["max_train_samples"][nonsort_task] = 0
    print("Config File:\n-------------")
    pp.pprint(config_sort_task_train_data)
    validator_sort_task_train_data = Validator(name='sort_task_train_data', 
                                               model=model, 
                                               batch_size=FLAGS.batch_size,
                                               data_dir=config_sort_task_train_data["data_dirs"], 
                                               data_subdir='train',
                                               max_samples=config_sort_task_train_data["max_train_samples"],
                                               maxout=FLAGS.maxout,
                                               read_seed=FLAGS.read_seed,
                                               ngpus=FLAGS.ngpus, 
                                               shuffle=FLAGS.shuffle,
                                               includePaths=True,
                                               workers=FLAGS.workers)
    
    print('Randomize Classes:', FLAGS.randomize_classes)
    
    if FLAGS.randomize_classes:
        print('----- run randomize_classes -----')
        with open(FLAGS.config_file) as f:
            config_sort_task_train_data = json.load(f)
        #config_sort_task_train_data = helper.Config(config_file=FLAGS.config_file)
        max_train_samples = copy.deepcopy(config_sort_task_train_data["max_train_samples"])
        max_train_samples[sort_task] = 0
        validator_nonsort_task_train_data = Validator(name='nonsort_task_train_data', 
                                                          model=model, 
                                                          batch_size=FLAGS.batch_size,
                                                          data_dir=config_sort_task_train_data["data_dirs"], 
                                                          data_subdir='train',
                                                          max_samples=max_train_samples,
                                                          maxout=FLAGS.maxout,
                                                          read_seed=FLAGS.read_seed,
                                                          ngpus=FLAGS.ngpus, 
                                                          shuffle=FLAGS.shuffle,
                                                          includePaths=True,
                                                          workers=FLAGS.workers)
        
        randomize_classes(sort_task_index        =FLAGS.sort_task_index, 
                          seed                   =FLAGS.randomize_classes_seed, 
                          validator_sort_task    =validator_sort_task_train_data, 
                          validator_nonsort_task =validator_nonsort_task_train_data)
        del validator_nonsort_task_train_data
        
    # print arguments
    # --------------------------
    #config.printAttributes()
    print(flush=True)
    print('--------MODEL-----------',flush=True)
    print('------------------------',flush=True)
    print(model,flush=True)
    print('------------------------',flush=True)
    print(flush=True)
    
    #helper.printArguments(config=config_sort_task_train_data, 
    #                      validator=validator_sort_task_train_data, 
    #                      mode='train', 
    #                      FLAGS=FLAGS)
    
    task_to_sort_by = sort_task
    if FLAGS.randomize_classes:
        task_to_sort_by = 'randomizedClasses_task_' \
        + str(FLAGS.sort_task_index) + '_seed_' + str(FLAGS.randomize_classes_seed)
    
    
    # Save File (HDF5)
    # --------------------------    
    print('-------SAVE FILE--------',flush=True)
    print('------------------------',flush=True)
    # OLD DIRECTORY STRUCTURE
    #network_dir = os.path.basename(os.path.dirname(FLAGS.config_file))
    #lesions_dir = os.path.join(FLAGS.lesions_dir, network_dir, config.name)
    #lesions_filename = 'lesions_LESION_NAME_' + FLAGS.lesion_name 
    #lesions_filename = os.path.join(lesions_dir, lesions_filename) + '.jsonl'
    #records_dir = ['LESION_NAME_'+ FLAGS.lesion_name,
    #                'SORTEDBY_' + task_to_sort_by, 
    #                'PARAM_GROUP_INDEX_'+ str(FLAGS.param_group_index)]
    #
    #records_dir = '_'.join(records_dir)
    #
    #selections_dir = os.path.join(lesions_dir, 'selections_records', records_dir)
    #progress_dir   = os.path.join(lesions_dir, 'progress_records',   records_dir)
    
    # NEW DIR STRUCTURE
    network_dir = os.path.basename(os.path.dirname(FLAGS.config_file))
    lesions_dir = os.path.join(FLAGS.lesions_dir, network_dir, os.path.basename(FLAGS.config_file)[:-5])
    records_dir=os.path.join(lesions_dir, 'LESION_NAME_'+ FLAGS.lesion_name)
    lesions_filename = os.path.join(records_dir, 'lesion.jsonl')
    selections_dir=os.path.join(records_dir,
                                'selections_records',
                                'SORTEDBY_'+task_to_sort_by,
                                'PARAM_GROUP_INDEX_'+ str(FLAGS.param_group_index))
    progress_dir=os.path.join(records_dir,
                              'progress_records',
                              'SORTEDBY_'+task_to_sort_by,
                              'PARAM_GROUP_INDEX_'+ str(FLAGS.param_group_index))
    
    print(flush=True)
    print('Results being saved to:', lesions_filename,flush=True)
    print(flush=True)
            
    print(flush=True)
    print('Record Files:', flush=True)
    print('\nSelections Records:', selections_dir,flush=True)
    print('\nProgress Records:', progress_dir,flush=True)
    print(flush=True)
                                                   
    if os.path.isfile(lesions_filename):
        print('Adding to existing jsonlines file...',flush=True)
        print(flush=True)
    else:
        print('Creating new jsonlines file...',flush=True)
        print(flush=True)
        if not os.path.exists(records_dir):
            os.makedirs(records_dir)

        keys = ['meta/greedy_p', 'meta/shuffle', 'meta/batch_size', 'meta/max_batches', 'meta/restore_epoch']
        values = [FLAGS.greedy_p, FLAGS.shuffle, FLAGS.batch_size, FLAGS.max_batches, FLAGS.restore_epoch]
        write_to_json(filename=lesions_filename, writer_method='w', keys=keys, values=values)
     
    # write predictions
    predictions_filename = None
    predictions_key = None
    if FLAGS.write_predictions:
        predictions_filename = os.path.join(records_dir, 'predictions.jsonl')
        predictions_key = os.path.join('SORTEDBY_'+task_to_sort_by,
                                       'PARAM_GROUP_INDEX_'+ str(FLAGS.param_group_index))
        
        if os.path.isfile(predictions_filename):
            print('Adding to existing jsonlines file...',flush=True)
            print('\npredictions file:\n', predictions_filename,flush=True)
            print(flush=True)
        else:
            print('Creating new jsonlines file...',flush=True)
            print('\npredictions file:\n', predictions_filename,flush=True)
            print(flush=True)
            key = 'dummy'
            obj = {'key':'value'}
            write_obj2jsonl(filename=predictions_filename, writer_method='w', key=key, obj=obj)
    
    
    
        
    # get latest selections
    progress_filename = get_latest_npz_filename(dir=progress_dir) 
    selections_complete = get_selections(filename=progress_filename)[4]
    
    if (progress_filename is None) or (selections_complete==False): 
         
        greedy_lesion_layer(validator=validator_sort_task_train_data, 
                            index=FLAGS.param_group_index,
                            layerMap=layerMap, 
                            selections_dir=selections_dir,
                            progress_dir=progress_dir,
                            predictions_filename=predictions_filename,
                            predictions_key=predictions_key,
                            sort_task=sort_task,
                            lesions_filename=lesions_filename,
                            iterator_seed=FLAGS.iterator_seed,
                            greedy_p=FLAGS.greedy_p, 
                            ngpus=FLAGS.ngpus, 
                            max_batches=FLAGS.max_batches,
                            group_p=FLAGS.group_p,
                            subgroups_file=FLAGS.subgroups_file)

    # wait for all processes to finish 
    while selections_complete==False: 
        progress_filename = get_latest_npz_filename(dir=progress_dir)
        selections_complete = get_selections(filename=progress_filename)[4]
        print('\nWaiting on other jobs to complete...', flush=True)
        time.sleep(10)
    
    # search json file for complete status
    # -------------------------
    status_is_complete = json_completion_status(filename=lesions_filename, 
                                                sort_task=task_to_sort_by, 
                                                param_group_index=FLAGS.param_group_index)
    
    if (status_is_complete == False):
        json_conclude_count = conclude_lesion_to_json(filename=lesions_filename, 
                                                      sort_task=task_to_sort_by, 
                                                      param_group_index=FLAGS.param_group_index)
        
        print('\njson_conclude_count:', json_conclude_count, flush=True)
        
        if json_conclude_count > 1:
            print('\nResults being written to JSON by another job!', flush=True)
            time.sleep(30)
            return None
                
        # write selections to json
        print('\nWriting lesion results to JSON file!', flush=True)
        print(lesions_filename, flush=True)
        keys = []
        values = []
        progress_filename = get_latest_npz_filename(dir=progress_dir)
        selections = get_selections(filename=progress_filename)
        selected_units, selected_losses, selected_accuracies, selected_subperformances, _ = selections
        
        group = os.path.join( 'selected_units', 'SORTEDBY_' + task_to_sort_by, str(FLAGS.param_group_index) )
        keys.append(group)
        values.append(selected_units.tolist())
        
        group = os.path.join('selected_losses', 'SORTEDBY_' + task_to_sort_by, str(FLAGS.param_group_index) )
        keys.append(group)
        values.append(selected_losses.tolist())
        
        group = os.path.join('selected_accuracies', 'SORTEDBY_' + task_to_sort_by, str(FLAGS.param_group_index) )
        keys.append(group)
        values.append(selected_accuracies.tolist())
        
        if len(selected_subperformances.shape)>2:
            group = os.path.join('selected_subperformances', 'SORTEDBY_' + task_to_sort_by, str(FLAGS.param_group_index) )
            keys.append(group)
            values.append(selected_subperformances.tolist())
        
        write_to_json(filename=lesions_filename, writer_method='a', keys=keys, values=values)
            
    print('---Lesion complete for current index!', flush=True)
    
if __name__ == "__main__": 
    run_lesion()
    print(flush=True)
    print('Lesion on Layer Complete.',flush=True)
    print(flush=True)
    
    