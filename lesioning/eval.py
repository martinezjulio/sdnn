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
import numpy as np
import copy
from utils import helper
import scipy
import h5py
import argparse
import json
import jsonlines as jsonl
import time
from zipfile import BadZipFile
from filelock import Timeout
from filelock import SoftFileLock as FileLock
import re
import lesion
import uuid
import transfer
from utils import tools
import torchvision.models as models
import pprint
pp = pprint.PrettyPrinter(indent=1)


lock_timeout = 5
acquire_timeout = 30
fixed_seed_value = 0

def make_random_selections(selections_dir, num_units, greedy_p):
    
    if greedy_p == 0.0:
        num_units_to_drop=num_units
    else:
        num_units_to_drop = np.round(num_units*greedy_p).astype(int)
    selected_units = np.random.choice(a=np.arange(num_units), size=num_units_to_drop, replace=False)
    
    print('\nNum Random Units:\n', len(selected_units), flush=True)
    print('\nRandomly generated selected_units:\n', selected_units, flush=True)
    print(flush=True)
    
    if not os.path.exists(selections_dir):
            os.makedirs(selections_dir)
            
    selections_filename = os.path.join(selections_dir, 'selections_record.npz')
    np.savez(selections_filename, 
             selected_units=selected_units, 
             selected_losses=np.array([None]), 
             selections_complete=np.array([None]))
    
    #print('Saved Selections @:', selections_filename)
    
    return None

def generate_drop_percent(filename, drop_percents_dir, beg=10, end=50, stepsize=10, overwrite=False):
    if filename is None or overwrite == True:
        
        
        if not os.path.exists(drop_percents_dir):
            os.makedirs(drop_percents_dir)
        
        filename = os.path.join(drop_percents_dir, 'drop_percent_record.npz')
        #os.mknod(filename)
        lockname = filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            drop_percent = beg
            remaining_drop_percents = np.delete(arr=np.arange(beg, end+1e-7, stepsize).astype(float), obj=0)
            #print('remaining_drop_percents', remaining_drop_percents)
            np.savez(file                    = filename,
                     remaining_drop_percents = remaining_drop_percents, 
                     completed_drop_percents = np.array([]).astype(float),
                     completed_accuracies    = np.array([]).astype(float),
                     completed_losses        = np.array([]).astype(float),
                     is_complete             = np.array([False]),
                     conclusion_count        = np.array([0]))
        finally:
            lock.release()
    else:
        lockname = filename + '.lock'
        lock = FileLock(lockname, timeout=lock_timeout)
        lock.acquire(timeout=acquire_timeout)
        try:
            # load data
            drop_percents_progress = lesion.load_record(filename)

            # get saved units
            remaining_drop_percents = drop_percents_progress['remaining_drop_percents'] 
            completed_drop_percents = drop_percents_progress['completed_drop_percents']
            completed_accuracies    = drop_percents_progress['completed_accuracies']
            completed_losses        = drop_percents_progress['completed_losses']
            is_complete             = drop_percents_progress['is_complete']
            conclusion_count        = drop_percents_progress['conclusion_count']

            if remaining_drop_percents.shape[0] > 0:
                drop_percent = remaining_drop_percents[0]
                remaining_drop_percents = np.delete(arr=remaining_drop_percents, obj=0)    
                np.savez(file                    = filename,
                         remaining_drop_percents = remaining_drop_percents, 
                         completed_drop_percents = completed_drop_percents,
                         completed_accuracies    = completed_accuracies,
                         completed_losses        = completed_losses,
                         is_complete             = is_complete,
                         conclusion_count        = conclusion_count)
            else:
                drop_percent = None
        finally:
            lock.release()
       
    return drop_percent, filename

def update_drop_percents(filename, drop_percent, accuracy, loss):
    assert(drop_percent is not None), 'unit must be an integer valued scalar'
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
        # load data
        drop_percents_progress = lesion.load_record(filename)

        # get saved units
        remaining_drop_percents = drop_percents_progress['remaining_drop_percents'] 
        completed_drop_percents = drop_percents_progress['completed_drop_percents']
        completed_accuracies    = drop_percents_progress['completed_accuracies']
        completed_losses        = drop_percents_progress['completed_losses']
        is_complete             = drop_percents_progress['is_complete']
        conclusion_count        = drop_percents_progress['conclusion_count']

        # make update
        completed_drop_percents = np.append(completed_drop_percents, drop_percent)
        completed_accuracies    = np.append(completed_accuracies, accuracy)
        completed_losses        = np.append(completed_losses, loss)

        # overwrite file
        np.savez(file                    = filename,
                 remaining_drop_percents = remaining_drop_percents, 
                 completed_drop_percents = completed_drop_percents,
                 completed_accuracies    = completed_accuracies,
                 completed_losses        = completed_losses,
                 is_complete             = is_complete,
                 conclusion_count        = conclusion_count)
    finally:
        lock.release()
    
    return None

def confirm_drop_percents_completion(filename):

    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
    
        # load data
        drop_percents_progress = lesion.load_record(filename)

        # get saved units
        remaining_drop_percents = drop_percents_progress['remaining_drop_percents'] 
        completed_drop_percents = drop_percents_progress['completed_drop_percents']
        completed_accuracies    = drop_percents_progress['completed_accuracies']
        completed_losses        = drop_percents_progress['completed_losses']
        is_complete             = np.array([True])
        conclusion_count = np.array([drop_percents_progress['conclusion_count'][0] + 1])

        # overwrite file
        np.savez(file                    = filename,
                 remaining_drop_percents = remaining_drop_percents, 
                 completed_drop_percents = completed_drop_percents,
                 completed_accuracies    = completed_accuracies,
                 completed_losses        = completed_losses,
                 is_complete             = is_complete,
                 conclusion_count        = conclusion_count)
    
    finally:
        lock.release()
    
    return conclusion_count[0]

def unconfirm_drop_percents_completion(filename):

    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:
    
        # load data
        drop_percents_progress = lesion.load_record(filename)

        # get saved units
        remaining_drop_percents = drop_percents_progress['remaining_drop_percents'] 
        completed_drop_percents = drop_percents_progress['completed_drop_percents']
        completed_accuracies    = drop_percents_progress['completed_accuracies']
        completed_losses        = drop_percents_progress['completed_losses']
        is_complete             = np.array([False])
        conclusion_count        = np.array([0])

        # overwrite file
        np.savez(file                    = filename,
                 remaining_drop_percents = remaining_drop_percents, 
                 completed_drop_percents = completed_drop_percents,
                 completed_accuracies    = completed_accuracies,
                 completed_losses        = completed_losses,
                 is_complete             = is_complete,
                 conclusion_count        = conclusion_count)
    
    finally:
        lock.release()
    
    return None

def get_drop_percents(filename):
    
    lockname = filename + '.lock'
    lock = FileLock(lockname, timeout=lock_timeout)
    lock.acquire(timeout=acquire_timeout)
    try:

        # load data
        drop_percents_progress = lesion.load_record(filename)

        # get saved units
        remaining_drop_percents = drop_percents_progress['remaining_drop_percents'] 
        completed_drop_percents = drop_percents_progress['completed_drop_percents']
        completed_accuracies    = drop_percents_progress['completed_accuracies']
        completed_losses        = drop_percents_progress['completed_losses']
        is_complete             = drop_percents_progress['is_complete']
        #conclusion_count       = drop_percents_progress['conclusion_count']

    finally:
        lock.release()
        
    return completed_drop_percents, completed_accuracies, completed_losses, is_complete

def lesion_layer_evaluator(validator, index, layerMap, drop_percent, selections_dir, iterator_seed, keep_units, ngpus=1):
    
    #print('Index:', index, flush=True)
    #print('Drop Percent:', drop_percent, flush=True)
    
    cache = {'W':{}, 'b':{}}
    weight, bias = helper.getWeightandBias(validator.model, layerMap, index)
    num_units = weight.shape[0]
    
    # Get Selected Units
    # ---------------------------
    latest_selections_record = lesion.get_latest_npz_filename(dir=selections_dir) # latest filename of selections record (or None)
    selected_units, _, _, _, _ = lesion.get_selections(filename=latest_selections_record)
    assert(len(selected_units) > 0), 'selected_units is empty!'
   
    #print('Num Selected Units:', len(selected_units), flush=True)  
       
    # get performances
    # -------------------------------------------------------
    if keep_units:
        keep_num = np.round(drop_percent*num_units).astype(int)
        if keep_num > len(selected_units):
            units_to_keep = np.arange(num_units)
        else:
            units_to_keep = selected_units[:keep_num]
    else:
        drop_num = np.round(drop_percent*num_units).astype(int)
        if drop_num > len(selected_units):
            drop_num = None
        drop_units = selected_units[:drop_num] 
        
    if keep_units:        
        for unit in range(num_units):
            if unit not in units_to_keep:
                cache['W'][unit] = weight[unit].detach().cpu().numpy()
                if bias is not None:
                    cache['b'][unit] = bias[unit].detach().cpu().numpy()
                weight[unit] = 0.0
                if bias is not None:
                    bias[unit]   = 0.0
    else:
        for unit in range(num_units):
            if unit in drop_units:
                cache['W'][unit] = weight[unit].detach().cpu().numpy()
                if bias is not None:
                    cache['b'][unit] = bias[unit].detach().cpu().numpy()
                weight[unit] = 0.0
                if bias is not None:
                    bias[unit]   = 0.0

    # get performance
    # ---------------
    y_true, y_pred, _, _, loss, _ = helper.predict(model=validator.model, 
                                                   data_loader=validator.data_loader, 
                                                   ngpus=ngpus,  
                                                   topk=1,
                                                   reduce_loss=True,
                                                   seed=iterator_seed)
    
    y_pred = np.squeeze(y_pred)
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    if keep_units:
        print("{0:<40}: {1:}".format('accuracy @keep ' + str(round(drop_percent,2)),accuracy),flush=True)
    else:
        print("{0:<40}: {1:}".format('accuracy @drop ' + str(round(drop_percent,2)),accuracy),flush=True)
        
    # replace selected_units
    # -------------------------------------------------------
    if keep_units:
        for unit in range(num_units):
            if unit not in units_to_keep:
                weight[unit] = torch.from_numpy(cache['W'][unit])
                if bias is not None:
                    bias[unit]   = torch.from_numpy(cache['b'][unit])
    else:
        for unit in drop_units:
            weight[unit] = torch.from_numpy(cache['W'][unit])
            if bias is not None:
                bias[unit]   = torch.from_numpy(cache['b'][unit]) 
    
    return loss, accuracy, y_true, y_pred

def array_lesion_layer_evaluator(validator, drop_percents_dir, drop_percents_range, param_group_index, 
                                 layerMap, selections_dir, ngpus, iterator_seed, keep_units):
    
    
    while True:
        latest_drop_percents_record = lesion.get_latest_npz_filename(dir=drop_percents_dir)
        drop_percent, latest_drop_percents_record = generate_drop_percent(filename=latest_drop_percents_record, 
                                                                          drop_percents_dir=drop_percents_dir, 
                                                                          beg=drop_percents_range[0], 
                                                                          end=drop_percents_range[1], 
                                                                          stepsize=drop_percents_range[2])
        if drop_percent is None:
            break
        
        loss, accuracy, y_true, y_pred = lesion_layer_evaluator(validator=validator, 
                                                                index=param_group_index, 
                                                                layerMap=layerMap, 
                                                                drop_percent=drop_percent,
                                                                selections_dir=selections_dir,
                                                                iterator_seed=iterator_seed,
                                                                keep_units=keep_units,
                                                                ngpus=ngpus)
        
        update_drop_percents(filename=latest_drop_percents_record, 
                             drop_percent=drop_percent, 
                             accuracy=accuracy, 
                             loss=loss)
        
        
        #####
        write_predictions(drop_percents_dir=drop_percents_dir, 
                          drop_percent=drop_percent, 
                          y_true=y_true, 
                          y_pred=y_pred)
        #####
    
    #time.sleep(30) # wait for other jobs to finish processing
    #_, _, _, is_complete = get_drop_percents(filename=latest_drop_percents_record)
    #if is_complete==False:
    #    confirm_drop_percents_completion(filename=latest_drop_percents_record)
 
    return None

def write_predictions(drop_percents_dir, drop_percent, y_true, y_pred):
    basename = os.path.basename(drop_percents_dir)
    save_suffix = "_".join(basename.split("_", 2)[:2])
    group = "_".join(basename.split("_", 2)[2:])
    sortedby = drop_percents_dir.split('/')[-2]
    predictions_group = os.path.join(group, sortedby, 'PERCENT_' + "{:.2f}".format(drop_percent))    
    predictions_dir = os.path.join(os.path.dirname(os.path.dirname(drop_percents_dir)), 'predictions', save_suffix)
    predictions_filename = os.path.join(predictions_dir, 'predictions.jsonl')

    #print("predictions_filename",predictions_filename)
    
    writer_method = 'a'
    if not os.path.exists(predictions_filename):
        writer_method = 'w'
        os.makedirs(predictions_dir)
        #print(predictions_dir)
        
    keys = [os.path.join(predictions_group, 'y_true'), os.path.join(predictions_group, 'y_pred')]
    values = [y_true.tolist(), y_pred.tolist()]
    
    lesion.write_to_json(filename=predictions_filename, writer_method=writer_method, keys=keys, values=values)
        
    return None

def get_base_performance(validator, ngpus, drop_percents_dir, iterator_seed):
    
    # BASE PERFORMANCE
    # ---------------------------
    y_true, y_pred, _, _, loss, _ = helper.predict(model=validator.model, 
                                                   data_loader=validator.data_loader, 
                                                   ngpus=ngpus,  
                                                   topk=1,
                                                   reduce_loss=True,
                                                   seed=iterator_seed)
    
    y_pred = np.squeeze(y_pred)
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    
    #####
    write_predictions(drop_percents_dir=drop_percents_dir, 
                      drop_percent=0.00, 
                      y_true=y_true, 
                      y_pred=y_pred)
    #####
    
    return loss, accuracy

def get_evaluation_objects(config_file,
                           model,
                           evaluation_task_index,
                           evaluation_task, 
                           nonevaluation_task, 
                           maxout, 
                           read_seed, 
                           shuffle, 
                           workers, 
                           param_group_index, 
                           batch_size, 
                           ngpus, 
                           drop_percents_dir,
                           layerMap,
                           drop_percents_beg,
                           drop_percents_end,
                           drop_percents_stepsize,
                           iterator_seed,
                           randomize_classes,
                           randomize_classes_seed):

    #config = helper.Config(config_file=config_file)
    with open(config_file) as f:
        config = json.load(f)
    max_valid_samples = copy.deepcopy(config["max_valid_samples"])
    max_valid_samples[nonevaluation_task] = 0
    validator = transfer.Validator(name=evaluation_task +'_validator', 
                                 model=model, 
                                 batch_size=batch_size,
                                 data_dir=config["data_dirs"], 
                                 data_subdir='test',
                                 max_samples=max_valid_samples,
                                 maxout=maxout,
                                 read_seed=read_seed,
                                 ngpus=ngpus, 
                                 shuffle=shuffle,
                                 includePaths=True,
                                 workers=workers)
    
    if randomize_classes:
        
        max_valid_samples = copy.deepcopy(config["max_valid_samples"])
        max_valid_samples[evaluation_task] = 0
        validator_nonevaluation = transfer.Validator(name=nonevaluation_task +'_validator', 
                                                   model=model, 
                                                   batch_size=batch_size,
                                                   data_dir=config["data_dirs"], 
                                                   data_subdir='test',
                                                   max_samples=max_valid_samples,
                                                   maxout=maxout,
                                                   read_seed=read_seed,
                                                   ngpus=ngpus, 
                                                   shuffle=shuffle,
                                                   includePaths=True,
                                                   workers=workers)
        
        lesion.randomize_classes(sort_task_index        =evaluation_task_index, 
                                 seed                   =randomize_classes_seed, 
                                 validator_sort_task    =validator, 
                                 validator_nonsort_task =validator_nonevaluation)
        
        del validator_nonevaluation
    

    #print(flush=True)
    #helper.printArguments(config=config, 
    #                      validator=validator, 
    #                      mode='valid', 
    #                      FLAGS=None)

    weight, _ = helper.getWeightandBias(validator.model, layerMap, param_group_index)
    num_units = weight.shape[0]
    layer = layerMap['ParamGroupIndex2LayerIndex'][param_group_index]
    layerType = layerMap['ParamGroupIndex2LayerType'][param_group_index]
    drop_percents_range = (drop_percents_beg,drop_percents_end,drop_percents_stepsize)

    #base_loss, base_accuracy = get_base_performance(validator=validator, 
    #                                                ngpus=ngpus,
    #                                                drop_percents_dir=drop_percents_dir,
    #                                                iterator_seed=iterator_seed)
    
    print("{0:<40}: {1:}".format('validator.name',validator.name),flush=True)
    print("{0:<40}: {1:}".format('index',param_group_index),flush=True)
    print("{0:<40}: {1:}".format('layer',layer),flush=True)
    print("{0:<40}: {1:}".format('layerType',layerType),flush=True)
    print("{0:<40}: {1:}".format('num_units',num_units),flush=True)
    print(flush=True)
    #print("{0:<40}: {1:}".format('accuracy @base',base_accuracy),flush=True)
    #print("{0:<40}: {1:}".format('loss     @base',base_loss),    flush=True)
    #print(flush=True)

    return validator, layerMap, weight, num_units, drop_percents_range

def write_lesion_performances_to_file(lesions_filename,
                                      sort_task,
                                      evaluation_task,
                                      param_group_index,
                                      drop_percents_dir,
                                      drop_percents_beg,
                                      drop_percents_end,
                                      drop_percents_stepsize):
    
    latest_drop_percents_record = lesion.get_latest_npz_filename(dir=drop_percents_dir)
    drop_percents_record_data = get_drop_percents(filename=latest_drop_percents_record)
    drop_percents, accuracies, losses, is_complete = drop_percents_record_data

    num_drop_percents_sofar = len(drop_percents)
    num_drop_percents_total = len(np.arange(drop_percents_beg, drop_percents_end+1e-7, drop_percents_stepsize))
    
    if (is_complete==False): # and (num_drop_percents_sofar == num_drop_percents_total):
        conclusion_count = confirm_drop_percents_completion(filename=latest_drop_percents_record)
        print('conclusion_count:', conclusion_count)
        if (conclusion_count > 1):
            print('\nResults written in another job.\n', flush=True)
            return None
            
        print('Num drop percents total:', num_drop_percents_total, flush=True)
        while num_drop_percents_sofar < num_drop_percents_total:
            print('\nWaiting for all selection candidates...', flush=True)
            print('Num drop percents so far:', num_drop_percents_sofar,flush=True)
            time.sleep(30)
            drop_percents_record_data = get_drop_percents(filename=latest_drop_percents_record)
            drop_percents, accuracies, losses, is_complete = drop_percents_record_data
            num_drop_percents_sofar = len(drop_percents)
            
            
        print('\nAll drop percents complete!\n', flush=True)
        print('Num drop percents so far:', num_drop_percents_sofar,flush=True)
                 
        keys = []
        values = []
        
        # Drop Percents
        # -------------------------- 
        sorted_indexes = np.argsort(drop_percents)
        drop_percents = drop_percents[sorted_indexes]
        keys.append('meta/drop_percents')
        values.append(drop_percents.tolist())
        
        # save losses_sort_task
        # --------------------------
        group = os.path.join('validated_losses_SORTEDBY_' + sort_task, evaluation_task, str(param_group_index) )
        losses = losses[sorted_indexes]
        keys.append(group)
        values.append(losses.tolist())
        
        # save accuracies_sort_task
        # --------------------------
        group = os.path.join('validated_accuracies_SORTEDBY_' + sort_task, evaluation_task, str(param_group_index) )
        accuracies = accuracies[sorted_indexes]
        keys.append(group)
        values.append(accuracies.tolist())
        
        # write to JSON file
        lesion.write_to_json(filename=lesions_filename, writer_method='a', keys=keys, values=values)

        print(flush=True)
        print('\nWriting results below to file!')
        print('drop_percents', np.round(drop_percents, decimals=3), flush=True)
        print('losses_nonsort_task', losses,flush=True)
        print('accuracies_nonsort_task', accuracies, flush=True)
    else:
        print('\nResults written in another job.\n', flush=True)

    return None
        
    
def run_evaluator():
    # FLAGS 
    # ------------------------------------------------------
    parser = argparse.ArgumentParser(description='Lesion Filters')
    parser.add_argument('--config_file',          default=None,    type=str,  help='path to config file')
    parser.add_argument('--greedy_p',             default=0.0,     type=float,  help='greedy value 0 turns off greedy')
    parser.add_argument('--param_group_index',    default=0,       type=int,  help='param weight and bias group index')
    parser.add_argument('--shuffle',              default="False",   type=str, help='shuffle data in dataloader')
    parser.add_argument('--random_lesion',        default="False",   type=str, help='random dropping')
    parser.add_argument('--ngpus',                default=1,       type=int,  help='number of gpus to use')
    parser.add_argument('--batch_size',           default=128,     type=int,  help='batch size')
    parser.add_argument('--max_batches',          default=5,       type=int,  help='batches to run on train losses')
    parser.add_argument('--workers',              default=1,       type=int,  help='read and write workers')
    parser.add_argument('--sort_task_index',      default=0,       type=int,  help='sort_task_index + 1')
    parser.add_argument('--nonsort_task_index',   default=1,       type=int,  help='nonsort_task_index + 1')
    parser.add_argument('--restore_epoch',        default=-1,      type=int,  help='epoch to restore from')
    parser.add_argument('--lesion_name',          default='',      type=str,  help='save suffix identifier')
    parser.add_argument('--eval_version',         default='',      type=str,  help='eval version',nargs='?',const='')
    parser.add_argument('--lesions_dir',          default='./lesions/', type=str,  help='where to read the losses from')
    parser.add_argument('--drop_percents_beg',    default=0.10,     type=float,  help='begin percentage drop')
    parser.add_argument('--drop_percents_end',    default=0.50,     type=float,  help='begin percentage drop')
    parser.add_argument('--drop_percents_stepsize',default=0.10,    type=float,  help='percentage drop stepsize')
    parser.add_argument('--read_seed',            default=None,     type=int,  help='seed type')
    parser.add_argument('--maxout',               default="False",  type=str, help='read all data and then shuffle')
    parser.add_argument('--iterator_seed',        default=0,        type=int, help='read all data and then shuffle')
    parser.add_argument('--randomize_classes',    default="False",  type=str, help='wether to randomly mix classes')
    parser.add_argument('--randomize_classes_seed', default=0,      type=int, help='how to mix the classes')
    parser.add_argument('--keep_units',            default="False", type=str, help='how to mix the classes')
    
    FLAGS, FIRE_FLAGS = parser.parse_known_args()
    
    # converts strings to corresponding boolean values
    FLAGS.shuffle = True if (FLAGS.shuffle == 'True') else False
    FLAGS.random_lesion = True if (FLAGS.random_lesion == 'True') else False
    FLAGS.maxout = True if (FLAGS.maxout == 'True') else False
    FLAGS.randomize_classes = True if (FLAGS.randomize_classes == 'True') else False
    FLAGS.keep_units = True if (FLAGS.keep_units == 'True') else False
    
    helper.printArguments(FLAGS=FLAGS)
    
    if FLAGS.ngpus > 0:
        torch.backends.cudnn.benchmark = True
        
        
    # Get Model and tasks
    # --------------------------
    with open(FLAGS.config_file) as f:
        config = json.load(f)
        print("Config File:\n-------------")
        pp.pprint(config)
    config_name = os.path.basename(FLAGS.config_file)[:-5]
    
    model = lesion.get_model(config_file=FLAGS.config_file, 
                      config=config, 
                      ngpus=FLAGS.ngpus, 
                      dataParallel=True, 
                      pretrained=True)
    
    layerMap = helper.getLayerMapping(model)
    
    # get task names
    config["batch_size"] = FLAGS.batch_size
    tasks = list(config["max_valid_samples"].keys())
    tasks.sort()
    sort_task = tasks[FLAGS.sort_task_index]
    nonsort_task = tasks[FLAGS.nonsort_task_index]
    
    # print arguments
    # --------------------------
    print(flush=True)
    print('--------MODEL-----------',flush=True)
    print('------------------------',flush=True)
    print(model,flush=True)
    print('------------------------',flush=True)
    print(flush=True)
    
    
    # Save File (HDF5)
    # -------------------------- 
    print('Save Files:',flush=True)
    print('-------------------------------------------',flush=True)
    network_dir = os.path.basename(os.path.dirname(FLAGS.config_file))
    lesions_dir = os.path.join(FLAGS.lesions_dir, network_dir, config_name)
    records_dir=os.path.join(lesions_dir, 'LESION_NAME_'+FLAGS.lesion_name)
    lesions_filename = os.path.join(records_dir, 'eval'+FLAGS.eval_version+'.json')
         
    if FLAGS.random_lesion:
        id = uuid.uuid1()
        task_to_sort_by = 'random' + id.hex   
        evaluation_sort_task = sort_task
        evaluation_nonsort_task = nonsort_task
    elif FLAGS.randomize_classes:
        task_to_sort_by = 'randomizedClasses_task_' + str(FLAGS.sort_task_index) \
        + '_seed_' + str(FLAGS.randomize_classes_seed)
        evaluation_sort_task = task_to_sort_by
        evaluation_nonsort_task = 'randomizedClasses_task_' + str(FLAGS.nonsort_task_index) \
        + '_seed_' + str(FLAGS.randomize_classes_seed)
    else:
        task_to_sort_by = sort_task
        evaluation_sort_task = sort_task
        evaluation_nonsort_task = nonsort_task
    
    selections_dir=os.path.join(records_dir,
                                'selections_records','SORTEDBY_'+task_to_sort_by,
                                'PARAM_GROUP_INDEX_'+ str(FLAGS.param_group_index))
    drop_percents_dir_sort_task = os.path.join(records_dir,
                                               'drop_percents_records', 
                                               'VERSION_' + FLAGS.eval_version ,
                                               'EVALUATION_TASK_' + evaluation_sort_task,
                                               'SORTEDBY_' + task_to_sort_by,
                                               'PARAM_GROUP_INDEX_'+ str(FLAGS.param_group_index))
    drop_percents_dir_nonsort_task = os.path.join(records_dir,
                                                  'drop_percents_records', 
                                                  'VERSION_' + FLAGS.eval_version ,
                                                  'EVALUATION_TASK_' + evaluation_nonsort_task,
                                                  'SORTEDBY_' + task_to_sort_by,
                                                  'PARAM_GROUP_INDEX_'+ str(FLAGS.param_group_index))
       
    print(flush=True)
    print('Results file:', lesions_filename,flush=True)
    print(flush=True)
    
    print(flush=True)
    print('Record Files:', flush=True)
    print('-------------------------------------------',flush=True)
    print('\nSelections Records:', selections_dir,flush=True)
    print('\nDrop Percent Records task 1   :\n', drop_percents_dir_sort_task,flush=True)
    print('\nDrop Percent Records task 2   :\n', drop_percents_dir_nonsort_task,flush=True)
    print(flush=True)                                            
            
    if os.path.isfile(lesions_filename):
        print('Adding to existing file...',flush=True)
        print(flush=True)
    else:
        print('Creating new jsonlines file...',flush=True)
        print(flush=True)
        if not os.path.exists(lesions_dir):
            os.makedirs(lesions_dir)
        
        keys = ['meta/shuffle','meta/batch_size', 'meta/max_batches', 'meta/restore_epoch']
        values = [FLAGS.shuffle, FLAGS.batch_size, FLAGS.max_batches, FLAGS.restore_epoch]
        lesion.write_to_json(filename=lesions_filename, writer_method='w', keys=keys, values=values)
    
    ###########################################################################
    ###########################################################################
    #                               EVALUATE                                  #
    ###########################################################################
    ###########################################################################
    
    
    
    # SORT TASK
    # ---------------------------------
    # ---------------------------------
    
    print('\n\n\n--------------------------------------------',flush=True)
    print('Starting TASK 1:',flush=True)
    print('----TASK 1: ' + evaluation_sort_task,flush=True)
    print('----SORTED BY: ' + task_to_sort_by,flush=True)
    print('-------------------------------------------',flush=True)

    # get evaluation objects for evaluation task
    evaluation_objects = get_evaluation_objects(config_file             =FLAGS.config_file,
                                                model                   =model,
                                                evaluation_task_index   =FLAGS.sort_task_index,
                                                evaluation_task         =sort_task, 
                                                nonevaluation_task      =nonsort_task, 
                                                maxout                  =FLAGS.maxout, 
                                                read_seed               =FLAGS.read_seed, 
                                                shuffle                 =FLAGS.shuffle, 
                                                workers                 =FLAGS.workers, 
                                                param_group_index       =FLAGS.param_group_index, 
                                                batch_size              =FLAGS.batch_size, 
                                                ngpus                   =FLAGS.ngpus, 
                                                drop_percents_dir       =drop_percents_dir_sort_task,
                                                layerMap                =layerMap,
                                                drop_percents_beg       =FLAGS.drop_percents_beg,
                                                drop_percents_end       =FLAGS.drop_percents_end,
                                                drop_percents_stepsize  =FLAGS.drop_percents_stepsize,
                                                iterator_seed           =FLAGS.iterator_seed,
                                                randomize_classes       =FLAGS.randomize_classes,
                                                randomize_classes_seed  =FLAGS.randomize_classes_seed)
    
    validator, layerMap, weight, num_units, drop_percents_range = evaluation_objects
    
    # Also counts for selections from below
    if FLAGS.random_lesion:
        make_random_selections(selections_dir=selections_dir, num_units=num_units, greedy_p=FLAGS.greedy_p)
    
    # starts evluation on percentage drops
    array_lesion_layer_evaluator(validator                              =validator, 
                                 drop_percents_dir                      =drop_percents_dir_sort_task, 
                                 drop_percents_range                    =drop_percents_range, 
                                 param_group_index                      =FLAGS.param_group_index,
                                 layerMap                               =layerMap, 
                                 selections_dir                         =selections_dir, 
                                 ngpus                                  =FLAGS.ngpus,
                                 iterator_seed                          =FLAGS.iterator_seed,
                                 keep_units                             =FLAGS.keep_units)
    
    # reads evaluation npz files and writes to final json file
    write_lesion_performances_to_file(lesions_filename                  =lesions_filename,
                                      sort_task                         =task_to_sort_by,
                                      evaluation_task                   =evaluation_sort_task,
                                      param_group_index                 =FLAGS.param_group_index,
                                      drop_percents_dir                 =drop_percents_dir_sort_task,
                                      drop_percents_beg                 =FLAGS.drop_percents_beg,
                                      drop_percents_end                 =FLAGS.drop_percents_end,
                                      drop_percents_stepsize            =FLAGS.drop_percents_stepsize)
    
    
    
    # NONSORT TASK
    # ---------------------------------
    # ---------------------------------
    print('\n\n\n--------------------------------------------',flush=True)
    print('Starting TASK 2:',flush=True)
    print('----TASK 2: ' + evaluation_nonsort_task,flush=True)
    print('----SORTED BY: ' + task_to_sort_by,flush=True)
    print('-------------------------------------------',flush=True)
  
    evaluation_objects = get_evaluation_objects(config_file             =FLAGS.config_file,
                                                model                   =model,
                                                evaluation_task_index   =FLAGS.nonsort_task_index,
                                                evaluation_task         =nonsort_task, 
                                                nonevaluation_task      =sort_task, 
                                                maxout                  =FLAGS.maxout, 
                                                read_seed               =FLAGS.read_seed, 
                                                shuffle                 =FLAGS.shuffle, 
                                                workers                 =FLAGS.workers, 
                                                param_group_index       =FLAGS.param_group_index, 
                                                batch_size              =FLAGS.batch_size, 
                                                ngpus                   =FLAGS.ngpus, 
                                                drop_percents_dir       =drop_percents_dir_nonsort_task,
                                                layerMap                =layerMap,
                                                drop_percents_beg       =FLAGS.drop_percents_beg,
                                                drop_percents_end       =FLAGS.drop_percents_end,
                                                drop_percents_stepsize  =FLAGS.drop_percents_stepsize,
                                                iterator_seed           =FLAGS.iterator_seed,
                                                randomize_classes       =FLAGS.randomize_classes,
                                                randomize_classes_seed  =FLAGS.randomize_classes_seed)
    
    validator, layerMap, weight, num_units, drop_percents_range = evaluation_objects
    
    array_lesion_layer_evaluator(validator                              =validator, 
                                 drop_percents_dir                      =drop_percents_dir_nonsort_task, 
                                 drop_percents_range                    =drop_percents_range, 
                                 param_group_index                      =FLAGS.param_group_index,
                                 layerMap                               =layerMap, 
                                 selections_dir                         =selections_dir, 
                                 iterator_seed                          =FLAGS.iterator_seed,
                                 ngpus                                  =FLAGS.ngpus,
                                 keep_units                             =FLAGS.keep_units)
              
    write_lesion_performances_to_file(lesions_filename                  =lesions_filename,
                                      sort_task                         =task_to_sort_by,
                                      evaluation_task                   =evaluation_nonsort_task,
                                      param_group_index                 =FLAGS.param_group_index,
                                      drop_percents_dir                 =drop_percents_dir_nonsort_task,
                                      drop_percents_beg                 =FLAGS.drop_percents_beg,
                                      drop_percents_end                 =FLAGS.drop_percents_end,
                                      drop_percents_stepsize            =FLAGS.drop_percents_stepsize)
                        
                         
if __name__ == "__main__": 
    run_evaluator()
    print(flush=True)
    print('Lesion Evaluator on Layer Complete.',flush=True)
    print(flush=True)                        

    
    
    

        

    
    
    
    
    
    
    
    
