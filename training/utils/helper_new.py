# std modules
import time
import os
import sys
import copy
import random
import itertools
from math import isclose

# custom modules
from . import tools_new,folder,helper,folder_list

# extra modules
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import yaml
import numpy as np
import tqdm
from sklearn import metrics
import pandas as pd
from tabulate import tabulate


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    pass  # to collapse comments above


class Config(object):
    def __init__(self, config_file):

        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self._name = config['project']['name']
        self._description = config['project']['description']
        self._model_type = config['project']['model']

        self._batch_size = config['hyperparameters']['batch_size']
        self._optimizer = config['hyperparameters']['optimizer']
        self._momentum = config['hyperparameters']['momentum'] if self._optimizer == 'sgd' else None
        self._learning_rate = config['hyperparameters']['learning_rate']
        self._step_size = config['hyperparameters'].get('step_size', None)
        self._weight_decay = config['hyperparameters']['weight_decay']

        self._use_scheduler = config['hyperparameters'].get(
            'use_scheduler', False)
        self._scheduler = config['hyperparameters'].get('scheduler', 'StepLR')
        self._scheduler_params = config['hyperparameters'].get(
            'scheduler_params', {'step_size': 50})
        self._reset_learning = config['hyperparameters'].get(
            'reset_learning', False)

        self._scale = config['hyperparameters'].get('scale', None)
        self._fc_affected = config['hyperparameters'].get('fc_affected', None)

        self._data_dir = list(config['data_directories'].values())
        self._num_classes = self._get_num_classes(task=None)
        self._split = config['hyperparameters']['split']

        self._checkpoints_dir = os.path.join(
            config['save_directories']['checkpoints_dir'], self.name)
        self._log_dir = os.path.join(
            config['save_directories']['log_dir'], self.name)

        self._max_train_samples = {}
        for key, value in config['max_train_samples'].items():
            dir = config['data_directories'][key]
            task_name = os.path.basename(dir)
            self._max_train_samples[task_name] = value
        self._max_valid_samples = {}
        for key, value in config['max_valid_samples'].items():
            dir = config['data_directories'][key]
            task_name = os.path.basename(dir)
            self._max_valid_samples[task_name] = value

        if 'saving' in config:
            self._keep_epochs = config['saving'].get('keep', [])
            self._saving_function = config['saving'].get('function', 'step')
            self._saving_params = config['saving'].get(
                'params', {'step': 5, 'offset': 0})
            self._max_count = config['saving'].get('max_count', -1)
        else:
            self._keep_epochs = []
            self._saving_function = 'step'
            self._saving_params = {'step': 5, 'offset': 0}
            self._max_count = -1

        if self._split:
            self._data_dir_task1 = copy.deepcopy(self.data_dir)
            self._num_classes_task1 = copy.deepcopy(self._num_classes)
            self._max_train_samples_task1 = copy.deepcopy(
                self._max_train_samples)
            self._max_valid_samples_task1 = copy.deepcopy(
                self._max_valid_samples)

            self._split_index = config['hyperparameters']['split_index']
            self._data_dir_task2 = list(config['data_directories_2'].values())
            self._num_classes_task2 = self._get_num_classes(task=2)

            self._max_train_samples_task2 = {}
            for key, value in config['max_train_samples_2'].items():
                dir = config['data_directories_2'][key]
                task_name = os.path.basename(dir)
                self._max_train_samples_task2[task_name] = value
            self._max_valid_samples_task2 = {}
            for key, value in config['max_valid_samples_2'].items():
                dir = config['data_directories_2'][key]
                task_name = os.path.basename(dir)
                self._max_valid_samples_task2[task_name] = value

            self._data_dir = None
            self._num_classes = None
            self._max_train_samples = None
            self._max_valid_samples = None

        else:
            self._split_index = None
            self._data_dir_task1 = None
            self._data_dir_task2 = None
            self._num_classes_task1 = None
            self._num_classes_task2 = None
            self._max_train_samples_task1 = None
            self._max_valid_samples_task1 = None
            self._max_train_samples_task2 = None
            self._max_valid_samples_task2 = None

    @property
    def name(self): return self._name

    @property
    def description(self): return self._description

    @property
    def model_type(self): return self._model_type

    @property
    def num_classes(self):
        assert (self._num_classes is not None), 'num_classes does not exist.'
        return self._num_classes

    @property
    def num_classes_task1(self):
        assert (
            self._num_classes_task1 is not None), 'num_classes_task1 does not exist.'
        return self._num_classes_task1

    @property
    def num_classes_task2(self):
        assert (
            self._num_classes_task2 is not None), 'num_classes_task2 does not exist.'
        return self._num_classes_task2

    @property
    def split(self): return self._split

    @property
    def split_index(self):
        assert (self._split_index is not None), 'split_index does not exist.'
        return self._split_index

    @property
    def scale(self):
        assert (self._scale is not None), 'scale does not exist.'
        return self._scale

    @property
    def fc_affected(self):
        assert (self._fc_affected is not None), 'fc_affected does not exist.'
        return self._fc_affected

    @property
    def batch_size(self): return self._batch_size

    @property
    def optimizer(self): return self._optimizer

    @property
    def momentum(self): return self._momentum

    @property
    def learning_rate(self): return self._learning_rate

    @property
    def step_size(self): return self._step_size

    @property
    def weight_decay(self): return self._weight_decay

    @property
    def use_scheduler(self):
        assert (self._use_scheduler is not None), 'use_scheduler does not exist.'
        return self._use_scheduler

    @property
    def scheduler(self):
        assert (self._scheduler is not None), 'scheduler does not exist.'
        return self._scheduler

    @property
    def scheduler_params(self):
        assert (
            self._scheduler_params is not None), 'scheduler_params does not exist.'
        return self._scheduler_params

    @property
    def reset_learning(self):
        assert (self._reset_learning is not None), 'reset_learning does not exist.'
        return self._reset_learning

    @property
    def data_dir(self):
        assert (self._data_dir is not None), 'data_dir does not exist.'
        return self._data_dir

    @property
    def data_dir_task1(self):
        assert (self._data_dir_task1 is not None), 'data_dir_task1 does not exist.'
        return self._data_dir_task1

    @property
    def data_dir_task2(self):
        assert (self._data_dir_task2 is not None), 'data_dir_task2 does not exist.'
        return self._data_dir_task2

    @property
    def checkpoints_dir(self): return self._checkpoints_dir

    @property
    def log_dir(self): return self._log_dir

    @property
    def keep_epochs(self): return self._keep_epochs

    @property
    def saving_function(self):
        assert (self._saving_function is not None), 'saving_function does not exist.'
        return self._saving_function

    @property
    def saving_params(self): return self._saving_params

    @property
    def max_count(self):
        assert (self._max_count is not None), 'max_count does not exist.'
        return self._max_count

    @property
    def max_train_samples(self):
        assert (
            self._max_train_samples is not None), 'max_train_samples does not exist.'
        return self._max_train_samples

    @property
    def max_valid_samples(self):
        assert (
            self._max_valid_samples is not None), 'max_valid_samples does not exist.'
        return self._max_valid_samples

    @property
    def max_train_samples_task1(self):
        assert (
            self._max_train_samples_task1 is not None), 'max_train_samples_task1 does not exist.'
        return self._max_train_samples_task1

    @property
    def max_valid_samples_task1(self):
        assert (
            self._max_valid_samples_task1 is not None), 'max_valid_samples_task1 does not exist.'
        return self._max_valid_samples_task1

    @property
    def max_train_samples_task2(self):
        assert (
            self._max_train_samples_task2 is not None), 'max_train_samples_task2 does not exist.'
        return self._max_train_samples_task2

    @property
    def max_valid_samples_task2(self):
        assert (
            self._max_valid_samples_task2 is not None), 'max_valid_samples_task2 does not exist.'
        return self._max_valid_samples_task2

    @batch_size.setter
    def batch_size(self, batch_size): self._batch_size = batch_size

    @data_dir.setter
    def data_dir(self, dir_):
        assert (self._data_dir is not None), "data_dir_task1 does not exist."
        self._data_dir = dir_

    @data_dir_task1.setter
    def data_dir_task1(self, dir_):
        assert (self._data_dir_task1 is not None), "data_dir_task1 does not exist."
        self._data_dir_task1 = dir_

    @data_dir_task2.setter
    def data_dir_task2(self, dir_):
        assert (self._data_dir_task2 is not None), "data_dir_task2 does not exist."
        self._data_dir_task2 = dir_

    @max_train_samples.setter
    def max_train_samples(self, max_samples):
        assert (
            self._max_train_samples is not None), 'max_train_samples does not exist.'
        self._max_train_samples = max_samples

    @max_valid_samples.setter
    def max_valid_samples(self, max_samples):
        assert (
            self._max_valid_samples is not None), 'max_valid_samples does not exist.'
        self._max_valid_samples = max_samples

    @max_train_samples_task1.setter
    def max_train_samples_task1(self, max_samples):
        assert (
            self._max_train_samples_task1 is not None), 'max_train_samples_task1 does not exist.'
        self._max_train_samples_task1 = max_samples

    @max_valid_samples_task1.setter
    def max_valid_samples_task1(self, max_samples):
        assert (
            self._max_valid_samples_task1 is not None), 'max_valid_samples_task1 does not exist.'
        self._max_valid_samples_task1 = max_samples

    @max_train_samples_task2.setter
    def max_train_samples_task2(self, max_samples):
        assert (
            self._max_train_samples_task2 is not None), 'max_train_samples_task2 does not exist.'
        self._max_train_samples_task2 = max_samples

    @max_valid_samples_task2.setter
    def max_valid_samples_task2(self, max_samples):
        assert (
            self._max_valid_samples_task2 is not None), 'max_valid_samples_task2 does not exist.'
        self._max_valid_samples_task2 = max_samples

    def _get_num_classes(self, task):
        # num classes
        if task is None:
            data_dir = self._data_dir
        elif task == 1:
            data_dir = self._data_dir_task1
        elif task == 2:
            data_dir = self._data_dir_task2
        islist = (list == type(data_dir))
        num_classes = tools_new.get_num_classes(data_dir, islist=islist)
        return num_classes

    def get_model(self, ngpus, pretrained=False, epoch=-1, dataParallel=False):
        if self.split:
            num_classes = (self._num_classes_task1, self._num_classes_task2)
        else:
            num_classes = self._num_classes

        model = tools_new.get_model(name=self._model_type,
                                          num_classes=num_classes,
                                          ngpus=ngpus,
                                          split_index=self._split_index,
                                          scale=self._scale,
                                          fc_affected=self._fc_affected,
                                          dataParallel=dataParallel)
        if pretrained:
            restore_path = tools_new.get_checkpoint(
                epoch=epoch, checkpoints_dir=self._checkpoints_dir)
            if ngpus > 0:
                ckpt_data = torch.load(restore_path)
            else:
                print('Loading model onto cpu...')
                ckpt_data = torch.load(
                    restore_path, map_location=torch.device('cpu'))

            model.load_state_dict(ckpt_data['state_dict'])
            print('Restored from: ' + os.path.relpath(restore_path))

            return model, ckpt_data
        else:
            return model, None

    def printAttributes(self):
        '''
        Description: prints all attributes/properties of the Config object
        '''
        def printFormat(name, var):
            print("{0:<30}: {1:}".format(name, var), flush=True)

        print(flush=True)
        print('---CONFIGURATION---', flush=True)
        print('------------------------', flush=True)
        if self._name is not None:
            printFormat('config.name', self.name)
        if self._description is not None:
            printFormat('config.description', self.description)
        if self._model_type is not None:
            printFormat('config.model_type', self.model_type)
        if self._num_classes is not None:
            printFormat('config.num_classes', self.num_classes)
        if self._num_classes_task1 is not None:
            printFormat('config.num_classes_task1', self.num_classes_task1)
        if self._num_classes_task2 is not None:
            printFormat('config.num_classes_task2', self.num_classes_task2)
        if self._split is not None:
            printFormat('config.split', self.split)
        if self._split_index is not None:
            printFormat('config.split_index', self.split_index)
        if self._scale is not None:
            printFormat('config.scale', self.scale)
        if self._fc_affected is not None:
            printFormat('config.fc_affected', self.fc_affected)
        if self._batch_size is not None:
            printFormat('config.batch_size', self.batch_size)
        if self._optimizer is not None:
            printFormat('config.optimizer', self.optimizer)
        if self._momentum is not None:
            printFormat('config.momentum', self.momentum)
        if self._learning_rate is not None:
            printFormat('config.learning_rate', self.learning_rate)
        if self._step_size is not None:
            printFormat('config.step_size', self.step_size)
        if self._weight_decay is not None:
            printFormat('config.weight_decay', self.weight_decay)
        if self._use_scheduler is not None:
            printFormat('config.use_scheduler', self.use_scheduler)
        if self._scheduler is not None:
            printFormat('config.scheduler', self.scheduler)
        if self._scheduler_params is not None:
            printFormat('config.scheduler_params', self.scheduler_params)
        if self._reset_learning is not None:
            printFormat('config.reset_learning', self.reset_learning)
        if self._data_dir is not None:
            printFormat('config.data_dir', '[')
            for directory in self.data_dir:
                printFormat('', directory)
            printFormat('', ']')
        if self._data_dir_task1 is not None:
            printFormat('config.data_dir_task1', self.data_dir_task1)
        if self._data_dir_task2 is not None:
            printFormat('config.data_dir_task2', self.data_dir_task2)
        if self._checkpoints_dir is not None:
            printFormat('config.checkpoints_dir', self.checkpoints_dir)
        if self._log_dir is not None:
            printFormat('config.log_dir', self.log_dir)
        if self._keep_epochs is not None:
            printFormat('config.keep_epochs', self.keep_epochs)
        if self._saving_function is not None:
            printFormat('config.saving_function', self.saving_function)
        if self._saving_params is not None:
            printFormat('config.saving_params', self.saving_params)
        if self._max_count is not None:
            printFormat('config.max_count', self.max_count)
        if self._max_train_samples is not None:
            printFormat('config.max_train_samples', self.max_train_samples)
        if self._max_valid_samples is not None:
            printFormat('config.max_valid_samples', self.max_valid_samples)
        if self._max_train_samples_task1 is not None:
            printFormat(
                'config.max_train_samples_task1',
                self.max_train_samples_task1)
        if self._max_valid_samples_task1 is not None:
            printFormat(
                'config.max_valid_samples_task1',
                self.max_valid_samples_task1)
        if self._max_train_samples_task2 is not None:
            printFormat(
                'config.max_train_samples_task2',
                self.max_train_samples_task2)
        if self._max_valid_samples_task2 is not None:
            printFormat(
                'config.max_valid_samples_task2',
                self.max_valid_samples_task2)
        print('------------------------', flush=True)
        print(flush=True)


if 'Image':  # "virtual" closure for image preprocessing steps
    IMAGE_RESIZE = 256
    IMAGE_SIZE = 224
    GRAYSCALE_PROBABILITY = 0.2
    resize_transform = torchvision.transforms.Resize(IMAGE_RESIZE)
    random_crop_transform = torchvision.transforms.RandomCrop(IMAGE_SIZE)
    center_crop_transform = torchvision.transforms.CenterCrop(IMAGE_SIZE)
    grayscale_transform = torchvision.transforms.RandomGrayscale(
        p=GRAYSCALE_PROBABILITY)
    normalize = torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)


class Trainer(object):

    def __init__(self, name, model, batch_size, learning_rate, weight_decay, data_dir, ngpus, workers, use_scheduler,
                 ckpt_data, task=None, max_samples=None, shuffle=True, data_subdir='train', includePaths=False, optim='adam',
                 momentum=None, maxout=True, read_seed=None, reset_learning=False, scheduler='StepLR', scheduler_params=None,
                 step_size=None):
        assert (task in [None, 1, 2]), "Task must one of {None, 1, 2}"
        assert (
            not use_scheduler or step_size is None), "step_size is deprecated, please use scheduler_params"
        assert (
            not use_scheduler or scheduler_params is not None), "scheduler_params is None but use_scheduler is True"
        self.name = name
        self.model = model
        self.learning_rate = learning_rate
        self.task = task
        self.max_samples = max_samples
        self.maxout = maxout
        self.read_seed = read_seed
        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.ngpus = ngpus
        self.batch_size = batch_size
        self.workers = workers
        self.shuffle = shuffle
        self.includePaths = includePaths
        self.dataset, self.data_loader = self.data()

        if optim == 'adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=learning_rate,
                                              weight_decay=weight_decay)
        if optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=learning_rate,
                                             momentum=momentum,
                                             weight_decay=weight_decay)
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau if scheduler == "ReduceLROnPlateau" else torch.optim.lr_scheduler.StepLR
            self.scheduler = scheduler(self.optimizer, **scheduler_params)
        if not reset_learning and ckpt_data:
            self.optimizer.load_state_dict(ckpt_data['optimizer'])
            if use_scheduler and 'scheduler' in ckpt_data:
                self.scheduler.load_state_dict(ckpt_data['scheduler'])

        self.criterion = torch.nn.CrossEntropyLoss()
        if self.ngpus > 0:
            self.criterion = self.criterion.cuda()

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def data(self):
        if type(self.data_dir) is list:
            ImageFolder = folder_list.ImageFolder
            train_data_dir = []
            for i in range(len(self.data_dir)):
                train_data_dir.append(
                    os.path.join(
                        self.data_dir[i],
                        self.data_subdir))
        else:
            ImageFolder = folder.ImageFolder
            train_data_dir = os.path.join(self.data_dir, self.data_subdir)

        # transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224),
        #                                           torchvision.transforms.RandomHorizontalFlip(),
        #                                           torchvision.transforms.ToTensor(),
        #                                           normalize,
        #                                          ])

        transform = torchvision.transforms.Compose([resize_transform,
                                                    random_crop_transform,
                                                    grayscale_transform,
                                                    torchvision.transforms.ToTensor(),
                                                    normalize,
                                                    ])
        dataset = ImageFolder(root=train_data_dir,
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

    def __call__(self, x, y, update_weights=True, activations=False, gradients=False, debug=None):  # DEBUG
        self.model.train()
        self.optimizer.zero_grad()

        if self.ngpus > 0 and debug:  # DEBUG
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            metrics.log_memory(debug)
        elif self.ngpus > 0:
            y = y.cuda(non_blocking=True)
        if self.task is None:
            output = self.model(x=x, activations=activations)
        else:
            output = self.model(
                x=x,
                task=self.task,
                activations=activations,
                gradients=gradients)
        prec_1, prec_5 = metrics.precision(
            output=output, target=y, topk=(1, 5))
        prec_1 /= len(output)
        prec_5 /= len(output)
        loss = self.criterion(output, y)

        if debug is None:  # DEBUG
            loss.backward()  # compute gradients
            if update_weights:
                self.optimizer.step()  # update weights
            return loss.item(), prec_1, prec_5, output
        # DEBUG
        torch.cuda.empty_cache()
        metrics.log_memory(debug)
        loss.backward()
        torch.cuda.empty_cache()
        metrics.log_memory(debug)
        if update_weights:
            self.optimizer.step()
            torch.cuda.empty_cache()
            metrics.log_memory(debug)
        return loss.item(), prec_1, prec_5, output
        # DEBUG


class Validator(object):
    def __init__(self, name, model, batch_size, data_dir, ngpus, workers,
                 task=None, max_samples=None, maxout=True, read_seed=None,
                 shuffle=False, data_subdir='test', includePaths=False):
        assert (task in [None, 1, 2]), "Task must one of {None, 1, 2}"
        self.name = name
        self.model = model
        self.task = task
        self.max_samples = max_samples
        self.maxout = maxout
        self.read_seed = read_seed
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
            ImageFolder = folder_list.ImageFolder
            test_data_dir = []
            for i in range(len(self.data_dir)):
                test_data_dir.append(
                    os.path.join(
                        self.data_dir[i],
                        self.data_subdir))
        else:
            ImageFolder = folder.ImageFolder
            test_data_dir = os.path.join(self.data_dir, self.data_subdir)

        # normalize = torchvision.transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)
        # transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
        #                                torchvision.transforms.CenterCrop(224),
        #                                torchvision.transforms.ToTensor(),
        #                                normalize,])

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

    def __call__(self, x, y, activations=False, gradients=False, debug=None):  # DEBUG
        if self.ngpus > 0 and debug:  # DEBUG
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=True)
            metrics.log_memory(debug)
        elif self.ngpus > 0:
            y = y.cuda(non_blocking=True)
        if self.task is None:
            output = self.model(
                x=x,
                activations=activations,
                gradients=gradients)
        else:
            output = self.model(
                x=x,
                task=self.task,
                activations=activations,
                gradients=gradients)
        prec_1, prec_5 = metrics.precision(
            output=output, target=y, topk=(1, 5))
        prec_1 /= len(output)
        prec_5 /= len(output)
        loss = self.criterion(output, y)
        if debug:
            torch.cuda.empty_cache()  # DEBUG
            metrics.log_memory(debug)  # DEBUG
        return loss.item(), prec_1, prec_5, output


def train(config_file, pretrained=False, restore_epoch=-1, epochs=50,
          valid_freq=1, save_freq=1, workers=1, ngpus=1, notebook=False,
          maxout=False, read_seed=None, use_scheduler=None, custom_learning_rate=None,
          max_count=-1):

    # assert(use_scheduler is False), "please set use_scheduler in the config file"
    assert (max_count == -1 or max_count >=
            0), "--max_count must be 0 or positive."
    np.random.seed(0)
    torch.manual_seed(0)

    print("Pretrained = " + str(pretrained))

    if ngpus > 0:
        torch.backends.cudnn.benchmark = True

    # progress bars
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        from tqdm import tnrange as trange
    else:
        from tqdm import tqdm as tqdm
        from tqdm import trange as trange

    # Get Model
    config = Config(config_file=config_file)
    config.printAttributes()
    model, ckpt_data = config.get_model(
        ngpus=ngpus, pretrained=pretrained, epoch=restore_epoch, dataParallel=True)

    checkpoints_exist = os.path.exists(config.checkpoints_dir)

    # assert((checkpoints_exist==True and pretrained==True) or \
    #       (checkpoints_exist==False and pretrained==False)), "If no checkpoints exist train from scratch, otherwise restore"

    if not os.path.exists(config.checkpoints_dir):
        os.makedirs(config.checkpoints_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # Saving setup
    keep_epochs = set()
    max_count = max_count if config.max_count is None else config.max_count
    save_epoch = saving_setup(config, save_freq, keep_epochs)
    force_save = False

    # for tensorboard
    writer = SummaryWriter(log_dir=config.log_dir)

    # validator
    validator = Validator(name='valid',
                          model=model,
                          batch_size=config.batch_size,
                          data_dir=config.data_dir,
                          ngpus=ngpus,
                          workers=workers,
                          max_samples=config.max_valid_samples,
                          maxout=maxout,
                          read_seed=read_seed)

    # trainer
    learning_rate = config.learning_rate
    use_scheduler = config.use_scheduler
    print(use_scheduler)
    print(use_scheduler == False)

    if use_scheduler == False:
        print(
            '\nSetting Custom Learning Rate:',
            custom_learning_rate,
            flush=True)
        learning_rate = custom_learning_rate

    trainer = Trainer(name='train',
                      model=model,
                      batch_size=config.batch_size,
                      learning_rate=learning_rate,
                      optim=config.optimizer,
                      momentum=config.momentum,
                      weight_decay=config.weight_decay,
                      data_dir=config.data_dir,
                      ngpus=ngpus,
                      workers=workers,
                      max_samples=config.max_train_samples,
                      maxout=maxout,
                      read_seed=read_seed,
                      ckpt_data=ckpt_data,
                      use_scheduler=use_scheduler,
                      scheduler=config.scheduler,
                      scheduler_params=config.scheduler_params,
                      reset_learning=config.reset_learning)

    if pretrained:
        start_epoch = ckpt_data['epoch'] + 1
        keep_epochs.update(ckpt_data.get('keep_epochs', set()))
        walltime = ckpt_data['walltime']
        model.load_state_dict(ckpt_data['state_dict'])
        if use_scheduler == False:
            print(
                '\nSetting Custom Learning Rate:',
                custom_learning_rate,
                flush=True)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = custom_learning_rate

        print()
        print('RESTORED model from epoch ' + str(start_epoch - 1))
        print()
    else:
        start_epoch = 1
        walltime = 0.0

    start_time = time.time()
    train_num_steps = len(trainer.data_loader)
    valid_num_steps = len(validator.data_loader)
    valid_step_freq = int(train_num_steps * valid_freq)
    stop_epoch = start_epoch + epochs - 1

    # print training arguments
    print('---TRAINING ARGUMENTS---')
    print('------------------------')
    print("{0:<30}: {1:}".format('config_file', config_file))
    print("{0:<30}: {1:}".format('pretrained', pretrained))
    print("{0:<30}: {1:}".format('ngpus', ngpus))
    print("{0:<30}: {1:}".format('workers', workers))
    print("{0:<30}: {1:}".format('restore_epoch', restore_epoch))
    print("{0:<30}: {1:}".format('epochs', epochs))
    print("{0:<30}: {1:}".format('valid_freq', valid_freq))
    print("{0:<30}: {1:}".format('save_freq', save_freq))
    print("{0:<30}: {1:}".format('keep_epochs', keep_epochs))
    print('------------------------')
    print()

    # print training arguments
    print('---NUMBER OF SAMPLES---')
    print('------------------------')
    print('Training Samples by torch.dataset:')
    print("{0:<30}: {1:}".format('train_num_steps', train_num_steps))
    print(
        "{0:<30}: {1:}".format(
            '~num_train_samples',
            train_num_steps *
            config.batch_size))
    print()
    print('Validation Samples by toch.dataset:')
    print("{0:<30}: {1:}".format('valid_num_steps', valid_num_steps))
    print(
        "{0:<30}: {1:}".format(
            '~num_valid_samples',
            valid_num_steps *
            config.batch_size))
    print()
    print(
        "{0:<30}: {1:}".format(
            'num_classes',
            trainer.dataset.task_to_num_classes))
    print()
    print('Training Samples by Configuration:')
    trainCount = 0
    for key in trainer.dataset.task_to_num_classes.keys():
        count = trainer.dataset.task_to_num_classes[key] * \
            config.max_train_samples[key]
        trainCount += count
        print("{0:<30}: {1:}".format(key, count))
    print("{0:<30}: {1:}".format('total', trainCount))
    print()
    print('Validation Samples by Configuration:')
    validCount = 0
    for key in validator.dataset.task_to_num_classes.keys():
        count = validator.dataset.task_to_num_classes[key] * \
            config.max_valid_samples[key]
        validCount += count
        print("{0:<30}: {1:}".format(key, count))
    print("{0:<30}: {1:}".format('total', validCount))
    print('------------------------')
    print()

    print('---OPTIMIZER------------')
    print('------------------------')
    print(trainer.optimizer)
    print('------------------------')
    print()

    if use_scheduler:
        print('---SCHEDULER------------')
        print('------------------------')
        print(trainer.scheduler.state_dict())
        print('------------------------')
        print()

    print('---MODEL----------------')
    print('------------------------')
    print(model)
    print('------------------------')
    print(flush=True)

    # save the random initial model
    if pretrained == False:
        print('saving initial model')
        ckpt_data = {}
        ckpt_data['batch_size'] = config.batch_size
        ckpt_data['learning_rate'] = trainer.learning_rate
        ckpt_data['momentum'] = config.momentum
        ckpt_data['step_size'] = config.step_size
        ckpt_data['weight_decay'] = config.weight_decay
        ckpt_data['walltime'] = walltime
        ckpt_data['epoch'] = 0
        ckpt_data['state_dict'] = model.state_dict()
        ckpt_data['optimizer'] = trainer.optimizer.state_dict()
        torch.save(
            ckpt_data,
            os.path.join(
                config.checkpoints_dir,
                f'epoch_0.pth.tar'))

    # for epoch in trange(start_epoch, stop_epoch+1, initial=start_epoch, desc='epoch'): # for usiing tqdm
        # for train_step, train_batch in enumerate(tqdm(trainer.data_loader,
        # desc='train')):
    for epoch in range(start_epoch, stop_epoch):
        if use_scheduler:
            learning_rate = trainer.lr
        force_save = False
        for train_step, train_batch in enumerate(trainer.data_loader):

            global_step = (epoch - 1) * train_num_steps + train_step
            print("global_step", global_step)

            # validate epoch
            if ((train_step) % valid_step_freq) == 0:
                validator.model.eval()
                with torch.no_grad():
                    avg_valid_loss = []
                    avg_valid_prec_1 = []
                    avg_valid_prec_5 = []
                    # for valid_step, valid_batch in
                    # enumerate(tqdm(validator.data_loader, desc='valid')):
                    for valid_step, valid_batch in enumerate(
                            validator.data_loader):
                        print('valid_step', valid_step)
                        x_valid, y_valid = valid_batch
                        valid_start_time = time.time()
                        valid_loss, valid_prec_1, valid_prec_5, _ = validator(
                            x=x_valid, y=y_valid)
                        valid_sec_per_iter = time.time() - valid_start_time
                        avg_valid_loss.append(valid_loss)
                        # print('valid_prec_1', valid_prec_1)
                        # print('valid_prec_5', valid_prec_5)
                        avg_valid_prec_1.append(valid_prec_1)
                        avg_valid_prec_5.append(valid_prec_5)
                    avg_valid_loss = np.mean(avg_valid_loss)
                    avg_valid_prec_1 = np.mean(avg_valid_prec_1)
                    avg_valid_prec_5 = np.mean(avg_valid_prec_5)

                    writer.add_scalar(
                        'valid/loss', avg_valid_loss, global_step)
                    writer.add_scalar(
                        'valid/precision1', avg_valid_prec_1, global_step)
                    writer.add_scalar(
                        'valid/precision5', avg_valid_prec_5, global_step)

                    print()
                    print('---------VALID EPOCH---------')
                    print("{0:<30}: {1:}".format('epoch', epoch))
                    print("{0:<30}: {1:}".format('global_step', global_step))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_loss',
                            avg_valid_loss))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec_1',
                            avg_valid_prec_1))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec_5',
                            avg_valid_prec_5))
                    print('-----------------------------')
                    print()

            # train step
            x_train, y_train = train_batch
            train_start_time = time.time()
            train_loss, train_prec_1, train_prec_5, _ = trainer(
                x=x_train, y=y_train)
            # print('train_prec_1', train_prec_1)
            # print('train_prec_5', train_prec_5)
            train_sec_per_iter = time.time() - train_start_time
            walltime += (time.time() - start_time) / (60.0**2)
            start_time = time.time()

            progress = (train_step + 1.0) / train_num_steps
            writer.add_scalar('train/loss', train_loss, global_step)
            writer.add_scalar('train/precision1', train_prec_1, global_step)
            writer.add_scalar('train/precision5', train_prec_5, global_step)
            writer.add_scalar('meta/learning_rate', learning_rate, global_step)
            writer.add_scalar('meta/progress', progress, global_step)
            writer.add_scalar(
                'meta/train-sec-per-iter',
                train_sec_per_iter,
                global_step)
            writer.add_scalar(
                'meta/valid-sec-per-iter',
                valid_sec_per_iter,
                global_step)
            writer.add_scalar('meta/walltime', walltime, global_step)

            print()
            print('---------TRAIN STEP---------')
            print("{0:<30}: {1:}".format('epoch', epoch))
            print("{0:<30}: {1:}".format('global_step', global_step))
            print("{0:<30}: {1:}".format('progress', progress))
            print("{0:<30}: {1:}".format('walltime', walltime))
            print("{0:<30}: {1:}".format('train_loss', train_loss))
            print("{0:<30}: {1:}".format('train_prec_1', train_prec_1))
            print("{0:<30}: {1:}".format('train_prec_5', train_prec_5))
            print("{0:<30}: {1:}".format('learning_rate', learning_rate))
            print('----------------------------')
            print()

        if use_scheduler:
            if config.scheduler == "ReduceLROnPlateau":
                trainer.scheduler.step(avg_valid_loss)
            else:
                trainer.scheduler.step()
            if learning_rate != trainer.lr:
                force_save = True
                keep_epochs.add(epoch)
            learning_rate = trainer.lr

        # Save Model
        if force_save or save_epoch(epoch):
            ckpt_data = {}
            if config.step_size:
                ckpt_data['step_size'] = config.step_size
            ckpt_data['batch_size'] = config.batch_size
            ckpt_data['learning_rate'] = learning_rate
            ckpt_data['momentum'] = config.momentum
            ckpt_data['weight_decay'] = config.weight_decay
            ckpt_data['walltime'] = walltime
            ckpt_data['epoch'] = epoch
            ckpt_data['keep_epochs'] = keep_epochs
            ckpt_data['state_dict'] = model.state_dict()
            ckpt_data['optimizer'] = trainer.optimizer.state_dict()
            ckpt_data['scheduler'] = trainer.scheduler.state_dict()
            tools_new.save_checkpoint(
                ckpt_data,
                config.checkpoints_dir,
                max_count=max_count,
                keep=keep_epochs)

        writer.add_scalar('meta/epoch', epoch, global_step)

    writer.close()


def debug(config_file, pretrained=False, restore_epoch=-1, epochs=50,
          valid_freq=0.5, save_freq=5, workers=1, ngpus=1, notebook=False,
          maxout=False, read_seed=None, use_scheduler=True, custom_learning_rate=None,
          max_count=-1):  # DEBUG

    assert (use_scheduler is None), "please set use_scheduler in the config file"
    assert (max_count == -1 or max_count >=
            0), "--max_count must be 0 or positive."
    np.random.seed(0)
    torch.manual_seed(0)

    if ngpus > 0:
        torch.backends.cudnn.benchmark = True

    # progress bars
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        from tqdm import tnrange as trange
    else:
        from tqdm import tqdm as tqdm
        from tqdm import trange as trange

    # Get Model
    config = Config(config_file=config_file)
    config.printAttributes()
    model, ckpt_data = config.get_model(
        ngpus=ngpus, pretrained=pretrained, epoch=restore_epoch, dataParallel=True)

    checkpoints_exist = os.path.exists(config.checkpoints_dir)

    assert ((checkpoints_exist == True and pretrained == True) or
            (checkpoints_exist == False and pretrained == False)), "If no checkpoints exist train from scratch, otherwise restore"

    if not os.path.exists(config.checkpoints_dir):
        os.makedirs(config.checkpoints_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # Saving setup
    keep_epochs = set()
    max_count = max_count if config.max_count is None else config.max_count
    save_epoch = saving_setup(config, save_freq, keep_epochs)
    force_save = False

    # validator
    validator = Validator(name='valid',
                          model=model,
                          batch_size=config.batch_size,
                          data_dir=config.data_dir,
                          ngpus=ngpus,
                          workers=workers,
                          max_samples=config.max_valid_samples,
                          maxout=maxout,
                          read_seed=read_seed)

    # trainer
    learning_rate = config.learning_rate
    use_scheduler = config.use_scheduler
    if use_scheduler == False:
        print(
            '\nSetting Custom Learning Rate:',
            custom_learning_rate,
            flush=True)
        learning_rate = custom_learning_rate

    trainer = Trainer(name='train',
                      model=model,
                      batch_size=config.batch_size,
                      learning_rate=learning_rate,
                      optim=config.optimizer,
                      momentum=config.momentum,
                      weight_decay=config.weight_decay,
                      data_dir=config.data_dir,
                      ngpus=ngpus,
                      workers=workers,
                      max_samples=config.max_train_samples,
                      maxout=maxout,
                      read_seed=read_seed,
                      ckpt_data=ckpt_data,
                      use_scheduler=use_scheduler,
                      scheduler=config.scheduler,
                      scheduler_params=config.scheduler_params,
                      reset_learning=config.reset_learning)
    if pretrained:
        start_epoch = ckpt_data['epoch'] + 1
        keep_epochs.update(ckpt_data.get('keep_epochs', set()))
        walltime = ckpt_data['walltime']
        model.load_state_dict(ckpt_data['state_dict'])
        if use_scheduler == False:
            print(
                '\nSetting Custom Learning Rate:',
                custom_learning_rate,
                flush=True)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = custom_learning_rate

        print()
        print('RESTORED model from epoch ' + str(start_epoch - 1))
        print()
    else:
        start_epoch = 1
        walltime = 0.0

    start_time = time.time()
    train_num_steps = len(trainer.data_loader)
    valid_num_steps = len(validator.data_loader)
    valid_step_freq = int(train_num_steps * valid_freq)
    stop_epoch = start_epoch + epochs

    # print training arguments
    print('---TRAINING ARGUMENTS---')
    print('------------------------')
    print("{0:<30}: {1:}".format('config_file', config_file))
    print("{0:<30}: {1:}".format('pretrained', pretrained))
    print("{0:<30}: {1:}".format('ngpus', ngpus))
    print("{0:<30}: {1:}".format('workers', workers))
    print("{0:<30}: {1:}".format('restore_epoch', restore_epoch))
    print("{0:<30}: {1:}".format('epochs', epochs))
    print("{0:<30}: {1:}".format('valid_freq', valid_freq))
    print("{0:<30}: {1:}".format('save_freq', save_freq))
    print("{0:<30}: {1:}".format('keep_epochs', keep_epochs))
    print('------------------------')
    print()

    # print training arguments
    print('---NUMBER OF SAMPLES---')
    print('------------------------')
    print('Training Samples by torch.dataset:')
    print("{0:<30}: {1:}".format('train_num_steps', train_num_steps))
    print(
        "{0:<30}: {1:}".format(
            '~num_train_samples',
            train_num_steps *
            config.batch_size))
    print()
    print('Validation Samples by toch.dataset:')
    print("{0:<30}: {1:}".format('valid_num_steps', valid_num_steps))
    print(
        "{0:<30}: {1:}".format(
            '~num_valid_samples',
            valid_num_steps *
            config.batch_size))
    print()
    print(
        "{0:<30}: {1:}".format(
            'num_classes',
            trainer.dataset.task_to_num_classes))
    print()
    print('Training Samples by Configuration:')
    trainCount = 0
    for key in trainer.dataset.task_to_num_classes.keys():
        count = trainer.dataset.task_to_num_classes[key] * \
            config.max_train_samples[key]
        trainCount += count
        print("{0:<30}: {1:}".format(key, count))
    print("{0:<30}: {1:}".format('total', trainCount))
    print()
    print('Validation Samples by Configuration:')
    validCount = 0
    for key in validator.dataset.task_to_num_classes.keys():
        count = validator.dataset.task_to_num_classes[key] * \
            config.max_valid_samples[key]
        validCount += count
        print("{0:<30}: {1:}".format(key, count))
    print("{0:<30}: {1:}".format('total', validCount))
    print('------------------------')
    print()

    print('---OPTIMIZER------------')
    print('------------------------')
    print(trainer.optimizer)
    print('------------------------')
    print()

    if use_scheduler:
        print('---SCHEDULER------------')
        print('------------------------')
        print(trainer.scheduler.state_dict())
        print('------------------------')
        print()

    print('---MODEL----------------')
    print('------------------------')
    print(model)
    print('------------------------')
    print(flush=True)

    debug = {
        'writer': SummaryWriter(
            log_dir=config.log_dir),
        'debug_step': 0,
        'global_step': 0}  # DEBUG
    # for epoch in trange(start_epoch, stop_epoch+1, initial=start_epoch, desc='epoch'): # for usiing tqdm
    # for train_step, train_batch in enumerate(tqdm(trainer.data_loader,
    # desc='train')):
    for epoch in range(start_epoch, stop_epoch):
        if use_scheduler:
            learning_rate = trainer.lr
        force_save = False
        for train_step, train_batch in enumerate(trainer.data_loader):

            debug['global_step'] = (
                epoch - 1) * train_num_steps + train_step  # DEBUG
            print("global_step", debug['global_step'])  # DEBUG

            # validate epoch
            if ((train_step) % valid_step_freq) == 0:
                validator.model.eval()
                with torch.no_grad():
                    avg_valid_loss = []
                    avg_valid_prec_1 = []
                    avg_valid_prec_5 = []
                    # for valid_step, valid_batch in
                    # enumerate(tqdm(validator.data_loader, desc='valid')):
                    for valid_step, valid_batch in enumerate(
                            validator.data_loader):
                        print('valid_step', valid_step)
                        x_valid, y_valid = valid_batch
                        valid_start_time = time.time()
                        valid_loss, valid_prec_1, valid_prec_5, _ = validator(
                            x=x_valid, y=y_valid, debug=debug)  # DEBUG
                        valid_sec_per_iter = time.time() - valid_start_time
                        avg_valid_loss.append(valid_loss)
                        # print('valid_prec_1', valid_prec_1)
                        # print('valid_prec_5', valid_prec_5)
                        avg_valid_prec_1.append(valid_prec_1)
                        avg_valid_prec_5.append(valid_prec_5)
                    avg_valid_loss = np.mean(avg_valid_loss)
                    avg_valid_prec_1 = np.mean(avg_valid_prec_1)
                    avg_valid_prec_5 = np.mean(avg_valid_prec_5)

                    debug['writer'].add_scalar(
                        'valid/loss', avg_valid_loss, debug['global_step'])  # DEBUG
                    debug['writer'].add_scalar(
                        'valid/precision1',
                        avg_valid_prec_1,
                        debug['global_step'])  # DEBUG
                    debug['writer'].add_scalar(
                        'valid/precision5',
                        avg_valid_prec_5,
                        debug['global_step'])  # DEBUG

                    print()
                    print('---------VALID EPOCH---------')
                    print("{0:<30}: {1:}".format('epoch', epoch))
                    print(
                        "{0:<30}: {1:}".format(
                            'global_step',
                            debug['global_step']))  # DEBUG
                    print(
                        "{0:<30}: {1:}".format(
                            'debug_step',
                            debug['debug_step']))  # DEBUG
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_loss',
                            avg_valid_loss))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec_1',
                            avg_valid_prec_1))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec_5',
                            avg_valid_prec_5))
                    print('-----------------------------')
                    print()

            # train step
            x_train, y_train = train_batch
            train_start_time = time.time()
            train_loss, train_prec_1, train_prec_5, _ = trainer(
                x=x_train, y=y_train, debug=debug)  # DEBUG
            # print('train_prec_1', train_prec_1)
            # print('train_prec_5', train_prec_5)
            train_sec_per_iter = time.time() - train_start_time
            walltime += (time.time() - start_time) / (60.0**2)
            start_time = time.time()

            progress = (train_step + 1.0) / train_num_steps
            debug['writer'].add_scalar(
                'train/loss', train_loss, debug['global_step'])  # DEBUG
            debug['writer'].add_scalar(
                'train/precision1',
                train_prec_1,
                debug['global_step'])  # DEBUG
            debug['writer'].add_scalar(
                'train/precision5',
                train_prec_5,
                debug['global_step'])  # DEBUG
            debug['writer'].add_scalar(
                'meta/learning_rate',
                learning_rate,
                debug['global_step'])  # DEBUG
            debug['writer'].add_scalar(
                'meta/progress', progress, debug['global_step'])  # DEBUG
            debug['writer'].add_scalar(
                'meta/train-sec-per-iter',
                train_sec_per_iter,
                debug['global_step'])  # DEBUG
            debug['writer'].add_scalar(
                'meta/valid-sec-per-iter',
                valid_sec_per_iter,
                debug['global_step'])  # DEBUG
            debug['writer'].add_scalar(
                'meta/walltime', walltime, debug['global_step'])  # DEBUG

            print()
            print('---------TRAIN STEP---------')
            print("{0:<30}: {1:}".format('epoch', epoch))
            print(
                "{0:<30}: {1:}".format(
                    'global_step',
                    debug['global_step']))  # DEBUG
            print(
                "{0:<30}: {1:}".format(
                    'debug_step',
                    debug['debug_step']))  # DEBUG
            print("{0:<30}: {1:}".format('progress', progress))
            print("{0:<30}: {1:}".format('train_loss', train_loss))
            print("{0:<30}: {1:}".format('train_prec_1', train_prec_1))
            print("{0:<30}: {1:}".format('train_prec_5', train_prec_5))
            print("{0:<30}: {1:}".format('learning_rate', learning_rate))
            print('----------------------------')
            print()

        if use_scheduler:  # DEBUG
            torch.cuda.empty_cache()
            metrics.log_memory(debug)
            if config.scheduler == "ReduceLROnPlateau":
                trainer.scheduler.step(avg_valid_loss)
            else:
                trainer.scheduler.step()
            torch.cuda.empty_cache()
            metrics.log_memory(debug)
            if learning_rate != trainer.lr:
                force_save = True
                keep_epochs.add(epoch)
            learning_rate = trainer.lr

        # Save Model
        if force_save or save_epoch(epoch):
            ckpt_data = {}
            if config.step_size:
                ckpt_data['step_size'] = config.step_size
            ckpt_data['batch_size'] = config.batch_size
            ckpt_data['learning_rate'] = learning_rate
            ckpt_data['momentum'] = config.momentum
            ckpt_data['weight_decay'] = config.weight_decay
            ckpt_data['walltime'] = walltime
            ckpt_data['epoch'] = epoch
            ckpt_data['keep_epochs'] = keep_epochs
            ckpt_data['state_dict'] = model.state_dict()
            ckpt_data['optimizer'] = trainer.optimizer.state_dict()
            ckpt_data['scheduler'] = trainer.scheduler.state_dict()
            tools_new.save_checkpoint(
                ckpt_data,
                config.checkpoints_dir,
                max_count=max_count,
                keep=keep_epochs)

        debug['writer'].add_scalar(
            'meta/epoch', epoch, debug['global_step'])  # DEBUG

    debug['writer'].close()  # DEBUG


def train_split(config_file, pretrained=False, restore_epoch=-1, epochs=50,
                valid_freq=0.5, save_freq=5, workers=1, ngpus=1, notebook=False,
                maxout=False, read_seed=None, use_scheduler=None, custom_learning_rate=None,
                max_count=-1):
    # TODO UPDATE WITH NEW SCHEDULER AND SAVING MECHANISM
    assert (use_scheduler is None), "please set use_scheduler in the config file"
    assert (max_count == -1 or max_count >=
            0), "--max_count must be 0 or positive."
    np.random.seed(0)
    torch.manual_seed(0)

    if ngpus > 0:
        torch.backends.cudnn.benchmark = True

    # progress bars
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        from tqdm import tnrange as trange
    else:
        from tqdm import tqdm as tqdm
        from tqdm import trange as trange

    # Get Model
    config = Config(config_file=config_file)
    config.printAttributes()
    model, ckpt_data = config.get_model(
        ngpus=ngpus, pretrained=pretrained, epoch=restore_epoch, dataParallel=True)

    checkpoints_exist = os.path.exists(config.checkpoints_dir)

    assert ((checkpoints_exist == True and pretrained == True) or
            (checkpoints_exist == False and pretrained == False)), "If no checkpoints exist train from scratch, otherwise restore"

    if not os.path.exists(config.checkpoints_dir):
        os.makedirs(config.checkpoints_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # Saving setup
    keep_epochs = set()
    max_count = max_count if config.max_count is None else config.max_count
    save_epoch = saving_setup(config, save_freq, keep_epochs)
    force_save = False

    # for tensorboard
    writer = SummaryWriter(log_dir=config.log_dir)

    validator1 = Validator(name='valid1',
                           model=model,
                           task=1,
                           batch_size=config.batch_size,
                           data_dir=config.data_dir_task1,
                           ngpus=ngpus,
                           workers=workers,
                           max_samples=config.max_valid_samples_task1,
                           maxout=maxout,
                           read_seed=read_seed)

    validator2 = Validator(name='valid2',
                           model=model,
                           task=2,
                           batch_size=config.batch_size,
                           data_dir=config.data_dir_task2,
                           ngpus=ngpus,
                           workers=workers,
                           max_samples=config.max_valid_samples_task2,
                           maxout=maxout,
                           read_seed=read_seed)

    # trainer
    learning_rate = config.learning_rate
    use_scheduler = config.use_scheduler
    if use_scheduler == False:
        print(
            '\nSetting Custom Learning Rate:',
            custom_learning_rate,
            flush=True)
        learning_rate = custom_learning_rate

    trainer1 = Trainer(name='train1',
                       model=model,
                       task=1,
                       batch_size=config.batch_size,
                       learning_rate=learning_rate,
                       optim=config.optimizer,
                       momentum=config.momentum,
                       weight_decay=config.weight_decay,
                       data_dir=config.data_dir_task1,
                       ngpus=ngpus,
                       workers=workers,
                       max_samples=config.max_train_samples_task1,
                       maxout=maxout,
                       read_seed=read_seed,
                       ckpt_data=ckpt_data,
                       use_scheduler=use_scheduler,
                       scheduler=config.scheduler,
                       scheduler_params=config.scheduler_params,
                       reset_learning=config.reset_learning)

    trainer2 = Trainer(name='train2',
                       model=model,
                       task=2,
                       batch_size=config.batch_size,
                       learning_rate=learning_rate,
                       optim=config.optimizer,
                       momentum=config.momentum,
                       step_size=config.step_size,
                       weight_decay=config.weight_decay,
                       data_dir=config.data_dir_task2,
                       ngpus=ngpus,
                       workers=workers,
                       max_samples=config.max_train_samples_task2,
                       maxout=maxout,
                       read_seed=read_seed,
                       ckpt_data=ckpt_data,
                       use_scheduler=use_scheduler,
                       scheduler=config.scheduler,
                       scheduler_params=config.scheduler_params,
                       reset_learning=config.reset_learning)

    if pretrained:
        start_epoch = ckpt_data['epoch'] + 1
        keep_epochs.update(ckpt_data.get('keep_epochs', set()))
        walltime = ckpt_data['walltime']
        model.load_state_dict(ckpt_data['state_dict'])
        if use_scheduler == False:
            print(
                '\nSetting Custom Learning Rate:',
                custom_learning_rate,
                flush=True)
            for param_group in trainer1.optimizer.param_groups:
                param_group['lr'] = custom_learning_rate
            for param_group in trainer2.optimizer.param_groups:
                param_group['lr'] = custom_learning_rate

        print()
        print('RESTORED model from epoch ' + str(start_epoch - 1))
        print()
    else:
        start_epoch = 1
        walltime = 0.0

    start_time = time.time()
    train_num_steps = max(len(trainer1.data_loader), len(trainer2.data_loader))
    valid_num_steps = max(len(validator1.data_loader),
                          len(validator2.data_loader))
    valid_step_freq = int(train_num_steps * valid_freq)
    stop_epoch = start_epoch + epochs

    if "Print Args":  # "virtual" closure for printing args
        # print training arguments
        print('---TRAINING ARGUMENTS---')
        print('------------------------')
        print("{0:<30}: {1:}".format('config_file', config_file))
        print("{0:<30}: {1:}".format('pretrained', pretrained))
        print("{0:<30}: {1:}".format('ngpus', ngpus))
        print("{0:<30}: {1:}".format('workers', workers))
        print("{0:<30}: {1:}".format('restore_epoch', restore_epoch))
        print("{0:<30}: {1:}".format('epochs', epochs))
        print("{0:<30}: {1:}".format('valid_freq', valid_freq))
        print("{0:<30}: {1:}".format('save_freq', save_freq))
        print("{0:<30}: {1:}".format('keep_epochs', keep_epochs))
        print('------------------------')
        print()

        # TODO make print_args function (ask how Julio uses current function)
        for task, trainer, validator in (
                (1, trainer1, validator1), (2, trainer2, validator2)):
            print(f'---NUMBER OF SAMPLES: TASK {task}---')
            print('------------------------')
            print('Training Samples by torch.dataset:')
            print(
                "{0:<30}: {1:}".format(
                    'train_num_steps', len(
                        trainer.data_loader)))
            print(
                "{0:<30}: {1:}".format(
                    '~num_train_samples', len(
                        trainer.data_loader) * config.batch_size))
            print()
            print('Validation Samples by toch.dataset:')
            print(
                "{0:<30}: {1:}".format(
                    'valid_num_steps', len(
                        validator.data_loader)))
            print(
                "{0:<30}: {1:}".format(
                    '~num_valid_samples', len(
                        validator.data_loader) * config.batch_size))
            print()
            print(
                "{0:<30}: {1:}".format(
                    'num_classes',
                    trainer.dataset.task_to_num_classes))
            print()
            print('Training Samples by Configuration:')
            trainCount = 0
            for key in trainer.dataset.task_to_num_classes.keys():
                count = trainer.dataset.task_to_num_classes[key] * getattr(
                    config, f'max_train_samples_task{task}')[key]
                trainCount += count
                print("{0:<30}: {1:}".format(key, count))
            print("{0:<30}: {1:}".format('total', trainCount))
            print()
            print('Validation Samples by Configuration:')
            validCount = 0
            for key in validator.dataset.task_to_num_classes.keys():
                count = validator.dataset.task_to_num_classes[key] * getattr(
                    config, f'max_valid_samples_task{task}')[key]
                validCount += count
                print("{0:<30}: {1:}".format(key, count))
            print("{0:<30}: {1:}".format('total', validCount))
            print('------------------------')
            print()

        print('---OPTIMIZER------------')
        print('------------------------')
        print('trainer1.optimizer: ')
        print()
        print(trainer1.optimizer)
        print()
        print('trainer2.optimizer: ')
        print()
        print(trainer2.optimizer)
        print('------------------------')
        print()

        print('---MODEL----------------')
        print('------------------------')
        print(model)
        print('------------------------')
        print(flush=True)

    # for epoch in trange(start_epoch, stop_epoch+1, initial=start_epoch,
    # desc='epoch'):
    for epoch in range(start_epoch, stop_epoch + 1):
        it1 = iter(trainer1.data_loader)
        it2 = iter(trainer2.data_loader)

        # for train_step in trange(train_num_steps, desc='train'):
        for train_step in range(train_num_steps):

            global_step = (epoch - 1) * train_num_steps + train_step
            print("global_step", global_step)

            # validate epoch
            if ((train_step) % valid_step_freq) == 0:
                valid_start_time = time.time()

                # task 1
                validator1.model.eval()
                with torch.no_grad():
                    avg_valid_loss_task1 = []
                    avg_valid_prec1_task1 = []
                    avg_valid_prec5_task1 = []
                    # for valid_step, valid_batch in
                    # enumerate(tqdm(validator1.data_loader,
                    # desc=validator1.name)):
                    for valid_step, valid_batch in enumerate(
                            validator1.data_loader):
                        print("valid_step1", valid_step)
                        x_valid, y_valid = valid_batch

                        valid_loss_task1, valid_prec1_task1, valid_prec5_task1, _ = validator1(
                            x=x_valid, y=y_valid)

                        avg_valid_loss_task1.append(valid_loss_task1)
                        avg_valid_prec1_task1.append(valid_prec1_task1)
                        avg_valid_prec5_task1.append(valid_prec5_task1)
                    avg_valid_loss_task1 = np.mean(avg_valid_loss_task1)
                    avg_valid_prec1_task1 = np.mean(avg_valid_prec1_task1)
                    avg_valid_prec5_task1 = np.mean(avg_valid_prec5_task1)
                    writer.add_scalar(
                        'valid/loss_task1',
                        avg_valid_loss_task1,
                        global_step)
                    writer.add_scalar(
                        'valid/precision1_task1',
                        avg_valid_prec1_task1,
                        global_step)
                    writer.add_scalar(
                        'valid/precision5_task1',
                        avg_valid_prec5_task1,
                        global_step)

                    print()
                    print('---------VALID TASK 1 STEP---------')
                    print("{0:<30}: {1:}".format('epoch', epoch))
                    print("{0:<30}: {1:}".format('global_step', global_step))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_loss_task1',
                            avg_valid_loss_task1))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec1_task1',
                            avg_valid_prec1_task1))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec5_task1',
                            avg_valid_prec5_task1))
                    print('-----------------------------------')
                    print()

                # task 2
                validator2.model.eval()
                with torch.no_grad():
                    avg_valid_loss_task2 = []
                    avg_valid_prec1_task2 = []
                    avg_valid_prec5_task2 = []
                    # for valid_step, valid_batch in
                    # enumerate(tqdm(validator2.data_loader,
                    # desc=validator2.name)):
                    for valid_step, valid_batch in enumerate(
                            validator2.data_loader):
                        print("valid_step2", valid_step)
                        x_valid, y_valid = valid_batch

                        valid_loss_task2, valid_prec1_task2, valid_prec5_task2, _ = validator2(
                            x=x_valid, y=y_valid)

                        avg_valid_loss_task2.append(valid_loss_task2)
                        avg_valid_prec1_task2.append(valid_prec1_task2)
                        avg_valid_prec5_task2.append(valid_prec5_task2)
                    avg_valid_loss_task2 = np.mean(avg_valid_loss_task2)
                    avg_valid_prec1_task2 = np.mean(avg_valid_prec1_task2)
                    avg_valid_prec5_task2 = np.mean(avg_valid_prec5_task2)
                    writer.add_scalar(
                        'valid/loss_task2',
                        avg_valid_loss_task2,
                        global_step)
                    writer.add_scalar(
                        'valid/precision1_task2',
                        avg_valid_prec1_task2,
                        global_step)
                    writer.add_scalar(
                        'valid/precision5_task2',
                        avg_valid_prec5_task2,
                        global_step)

                    print()
                    print('---------VALID TASK 2 STEP---------')
                    print("{0:<30}: {1:}".format('epoch', epoch))
                    print("{0:<30}: {1:}".format('global_step', global_step))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_loss_task2',
                            avg_valid_loss_task2))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec1_task2',
                            avg_valid_prec1_task2))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec5_task2',
                            avg_valid_prec5_task2))
                    print('-----------------------------------')
                    print()

                valid_sec_per_validation = time.time() - valid_start_time
                writer.add_scalar(
                    'meta/sec-per-validation',
                    valid_sec_per_validation,
                    epoch)

            train_start_time = time.time()

            # train step task1
            try:
                train_batch1 = next(it1)
            except StopIteration:
                train_batch1 = None

            if train_batch1 is not None:
                x_train1, y_train1 = train_batch1
                train_loss_task1, train_prec1_task1, train_prec5_task1, _ = trainer1(
                    x=x_train1, y=y_train1)

                writer.add_scalar(
                    'train/loss_task1',
                    train_loss_task1,
                    global_step)
                writer.add_scalar(
                    'train/precision1_task1',
                    train_prec1_task1,
                    global_step)
                writer.add_scalar(
                    'train/precision5_task1',
                    train_prec5_task1,
                    global_step)

                print()
                print('---------TRAIN TASK 1 STEP---------')
                print("{0:<30}: {1:}".format('epoch', epoch))
                print("{0:<30}: {1:}".format('global_step', global_step))
                print(
                    "{0:<30}: {1:}".format(
                        'train_loss_task1',
                        train_loss_task1))
                print(
                    "{0:<30}: {1:}".format(
                        'train_prec1_task1',
                        train_prec1_task1))
                print(
                    "{0:<30}: {1:}".format(
                        'train_prec5_task1',
                        train_prec5_task1))
                print('-----------------------------------')
                print()

            # train step task2
            try:
                train_batch2 = next(it2)
            except StopIteration:
                train_batch2 = None

            if train_batch2 is not None:
                x_train2, y_train2 = train_batch2
                train_loss_task2, train_prec1_task2, train_prec5_task2, _ = trainer2(
                    x=x_train2, y=y_train2)

                writer.add_scalar(
                    'train/loss_task2',
                    train_loss_task2,
                    global_step)
                writer.add_scalar(
                    'train/precision1_task2',
                    train_prec1_task2,
                    global_step)
                writer.add_scalar(
                    'train/precision5_task2',
                    train_prec5_task2,
                    global_step)

                print()
                print('---------TRAIN TASK 2 STEP---------')
                print("{0:<30}: {1:}".format('epoch', epoch))
                print("{0:<30}: {1:}".format('global_step', global_step))
                print(
                    "{0:<30}: {1:}".format(
                        'train_loss_task2',
                        train_loss_task2))
                print(
                    "{0:<30}: {1:}".format(
                        'train_prec1_task2',
                        train_prec1_task2))
                print(
                    "{0:<30}: {1:}".format(
                        'train_prec5_task2',
                        train_prec5_task2))
                print('-----------------------------------')
                print()

            train_sec_per_iter = time.time() - train_start_time
            walltime += (time.time() - start_time) / (60.0**2)
            start_time = time.time()
            progress = (train_step + 1.0) / train_num_steps
            if use_scheduler:
                lr = trainer1.scheduler.get_lr()[0]
            else:
                lr = trainer1.learning_rate

            writer.add_scalar('meta/learning_rate', lr, global_step)
            writer.add_scalar('meta/progress', progress, global_step)
            writer.add_scalar(
                'meta/train-sec-per-iter',
                train_sec_per_iter,
                global_step)
            writer.add_scalar('meta/walltime', walltime, global_step)

            print()
            print('---------STEP COMPLETE---------')
            print("{0:<30}: {1:}".format('epoch', epoch))
            print("{0:<30}: {1:}".format('global_step', global_step))
            print("{0:<30}: {1:}".format('walltime', walltime))
            print("{0:<30}: {1:}".format('progress', progress))
            print("{0:<30}: {1:}".format('learning_rate', lr))
            print('----------------------------')
            print()

        # Save Model
        if ((epoch % save_freq) == 0):
            ckpt_data = {}
            ckpt_data['batch_size'] = config.batch_size
            ckpt_data['learning_rate'] = trainer1.learning_rate
            ckpt_data['momentum'] = config.momentum
            ckpt_data['step_size'] = config.step_size
            ckpt_data['weight_decay'] = config.weight_decay
            ckpt_data['walltime'] = walltime
            ckpt_data['epoch'] = epoch
            ckpt_data['state_dict'] = model.state_dict()
            ckpt_data['optimizer1'] = trainer1.optimizer.state_dict()
            ckpt_data['optimizer2'] = trainer2.optimizer.state_dict()
            tools_new.save_checkpoint(
                ckpt_data, config.checkpoints_dir, max_count=max_count)

        if use_scheduler:
            trainer1.scheduler.step(epoch=epoch)
            trainer2.scheduler.step(epoch=epoch)
        writer.add_scalar('meta/epoch', epoch, global_step)

    writer.close()


def debug_split(config_file, pretrained=False, restore_epoch=-1,
                epochs=50, valid_freq=0.5, save_freq=5, workers=1,
                ngpus=1, notebook=False, maxout=False, read_seed=None,
                use_scheduler=True, custom_learning_rate=None,
                max_count=-1):
    # TODO UPDATE WITH NEW SAVING AND SCHEDULER MECHANISM
    assert (max_count == -1 or max_count >=
            0), "--max_count must be 0 or positive."
    np.random.seed(0)
    torch.manual_seed(0)

    if ngpus > 0:
        torch.backends.cudnn.benchmark = True

    # progress bars
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        from tqdm import tnrange as trange
    else:
        from tqdm import tqdm as tqdm
        from tqdm import trange as trange

    # Get Model
    config = helper.Config(config_file=config_file)
    config.printAttributes()
    model, ckpt_data = config.get_model(
        ngpus=ngpus, pretrained=pretrained, epoch=restore_epoch, dataParallel=True)

    checkpoints_exist = os.path.exists(config.checkpoints_dir)

    assert ((checkpoints_exist == True and pretrained == True) or
            (checkpoints_exist == False and pretrained == False)), "If no checkpoints exist train from scratch, otherwise restore"

    if not os.path.exists(config.checkpoints_dir):
        os.makedirs(config.checkpoints_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # for tensorboard
    writer = SummaryWriter(log_dir=config.log_dir)

    validator1 = helper.Validator(name='valid1',
                                  model=model,
                                  task=1,
                                  batch_size=config.batch_size,
                                  data_dir=config.data_dir_task1,
                                  ngpus=ngpus,
                                  workers=workers,
                                  max_samples=config.max_valid_samples_task1,
                                  maxout=maxout,
                                  read_seed=read_seed)

    validator2 = helper.Validator(name='valid2',
                                  model=model,
                                  task=2,
                                  batch_size=config.batch_size,
                                  data_dir=config.data_dir_task2,
                                  ngpus=ngpus,
                                  workers=workers,
                                  max_samples=config.max_valid_samples_task2,
                                  maxout=maxout,
                                  read_seed=read_seed)

    # trainer
    learning_rate = config.learning_rate
    if use_scheduler == False:
        print(
            '\nSetting Custom Learning Rate:',
            custom_learning_rate,
            flush=True)
        learning_rate = custom_learning_rate

    trainer1 = helper.Trainer(name='train1',
                              model=model,
                              task=1,
                              batch_size=config.batch_size,
                              learning_rate=learning_rate,
                              optim=config.optimizer,
                              momentum=config.momentum,
                              step_size=config.step_size,
                              weight_decay=config.weight_decay,
                              data_dir=config.data_dir_task1,
                              ngpus=ngpus,
                              workers=workers,
                              max_samples=config.max_train_samples_task1,
                              maxout=maxout,
                              read_seed=read_seed,
                              use_scheduler=use_scheduler)

    trainer2 = helper.Trainer(name='train2',
                              model=model,
                              task=2,
                              batch_size=config.batch_size,
                              learning_rate=learning_rate,
                              optim=config.optimizer,
                              momentum=config.momentum,
                              step_size=config.step_size,
                              weight_decay=config.weight_decay,
                              data_dir=config.data_dir_task2,
                              ngpus=ngpus,
                              workers=workers,
                              max_samples=config.max_train_samples_task2,
                              maxout=maxout,
                              read_seed=read_seed,
                              use_scheduler=use_scheduler)

    if pretrained:
        start_epoch = ckpt_data['epoch'] + 1
        walltime = ckpt_data['walltime']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer1.optimizer.load_state_dict(ckpt_data['optimizer1'])
        trainer2.optimizer.load_state_dict(ckpt_data['optimizer2'])
        if use_scheduler == False:
            print(
                '\nSetting Custom Learning Rate:',
                custom_learning_rate,
                flush=True)
            for param_group in trainer1.optimizer.param_groups:
                param_group['lr'] = custom_learning_rate
            for param_group in trainer2.optimizer.param_groups:
                param_group['lr'] = custom_learning_rate

        print()
        print('RESTORED model from epoch ' + str(start_epoch - 1))
        print()
    else:
        start_epoch = 1
        walltime = 0.0

    start_time = time.time()
    train_num_steps = max(len(trainer1.data_loader), len(trainer2.data_loader))
    valid_num_steps = max(len(validator1.data_loader),
                          len(validator2.data_loader))
    valid_step_freq = int(train_num_steps * valid_freq)
    stop_epoch = start_epoch + epochs

    if "Print Args":  # "virtual" closure for printing args
        # print training arguments
        print('---TRAINING ARGUMENTS---')
        print('------------------------')
        print("{0:<30}: {1:}".format('config_file', config_file))
        print("{0:<30}: {1:}".format('pretrained', pretrained))
        print("{0:<30}: {1:}".format('ngpus', ngpus))
        print("{0:<30}: {1:}".format('workers', workers))
        print("{0:<30}: {1:}".format('restore_epoch', restore_epoch))
        print("{0:<30}: {1:}".format('epochs', epochs))
        print("{0:<30}: {1:}".format('valid_freq', valid_freq))
        print("{0:<30}: {1:}".format('save_freq', save_freq))
        print('------------------------')
        print()

        # print training arguments
        print('---NUMBER OF SAMPLES: TASK 1---')
        print('------------------------')
        print('Training Samples by torch.dataset:')
        print(
            "{0:<30}: {1:}".format(
                'train_num_steps', len(
                    trainer1.data_loader)))
        print(
            "{0:<30}: {1:}".format(
                '~num_train_samples', len(
                    trainer1.data_loader) * config.batch_size))
        print()
        print('Validation Samples by toch.dataset:')
        print(
            "{0:<30}: {1:}".format(
                'valid_num_steps', len(
                    validator1.data_loader)))
        print(
            "{0:<30}: {1:}".format(
                '~num_valid_samples', len(
                    validator1.data_loader) * config.batch_size))
        print()
        print(
            "{0:<30}: {1:}".format(
                'num_classes',
                trainer1.dataset.task_to_num_classes))
        print()
        print('Training Samples by Configuration:')
        trainCount = 0
        for key in trainer1.dataset.task_to_num_classes.keys():
            count = trainer1.dataset.task_to_num_classes[key] * \
                config.max_train_samples_task1[key]
            trainCount += count
            print("{0:<30}: {1:}".format(key, count))
        print("{0:<30}: {1:}".format('total', trainCount))
        print()
        print('Validation Samples by Configuration:')
        validCount = 0
        for key in validator1.dataset.task_to_num_classes.keys():
            count = validator1.dataset.task_to_num_classes[key] * \
                config.max_valid_samples_task1[key]
            validCount += count
            print("{0:<30}: {1:}".format(key, count))
        print("{0:<30}: {1:}".format('total', validCount))
        print('------------------------')
        print()

        # print training arguments
        print('---NUMBER OF SAMPLES: TASK 2---')
        print('------------------------')
        print('Training Samples by torch.dataset:')
        print(
            "{0:<30}: {1:}".format(
                'train_num_steps', len(
                    trainer2.data_loader)))
        print(
            "{0:<30}: {1:}".format(
                '~num_train_samples', len(
                    trainer2.data_loader) * config.batch_size))
        print()
        print('Validation Samples by toch.dataset:')
        print(
            "{0:<30}: {1:}".format(
                'valid_num_steps', len(
                    validator2.data_loader)))
        print(
            "{0:<30}: {1:}".format(
                '~num_valid_samples', len(
                    validator2.data_loader) * config.batch_size))
        print()
        print(
            "{0:<30}: {1:}".format(
                'num_classes',
                trainer2.dataset.task_to_num_classes))
        print()
        print('Training Samples by Configuration:')
        trainCount = 0
        for key in trainer2.dataset.task_to_num_classes.keys():
            count = trainer2.dataset.task_to_num_classes[key] * \
                config.max_train_samples_task2[key]
            trainCount += count
            print("{0:<30}: {1:}".format(key, count))
        print("{0:<30}: {1:}".format('total', trainCount))
        print()
        print('Validation Samples by Configuration:')
        validCount = 0
        for key in validator2.dataset.task_to_num_classes.keys():
            count = validator2.dataset.task_to_num_classes[key] * \
                config.max_valid_samples_task2[key]
            validCount += count
            print("{0:<30}: {1:}".format(key, count))
        print("{0:<30}: {1:}".format('total', validCount))
        print('------------------------')
        print()

        print('---OPTIMIZER------------')
        print('------------------------')
        print('trainer1.optimizer: ')
        print()
        print(trainer1.optimizer)
        print()
        print('trainer2.optimizer: ')
        print()
        print(trainer2.optimizer)
        print('------------------------')
        print()

        print('---MODEL----------------')
        print('------------------------')
        print(model)
        print('------------------------')
        print(flush=True)

    # for epoch in trange(start_epoch, stop_epoch+1, initial=start_epoch,
    # desc='epoch'):
    for epoch in range(start_epoch, stop_epoch + 1):
        it1 = iter(trainer1.data_loader)
        it2 = iter(trainer2.data_loader)

        # for train_step in trange(train_num_steps, desc='train'):
        for train_step in range(train_num_steps):

            global_step = (epoch - 1) * train_num_steps + train_step
            print("global_step", global_step)

            # validate epoch
            if ((train_step) % valid_step_freq) == 0:
                valid_start_time = time.time()

                # task 1
                validator1.model.eval()
                with torch.no_grad():
                    avg_valid_loss_task1 = []
                    avg_valid_prec1_task1 = []
                    avg_valid_prec5_task1 = []
                    # for valid_step, valid_batch in
                    # enumerate(tqdm(validator1.data_loader,
                    # desc=validator1.name)):
                    for valid_step, valid_batch in enumerate(
                            validator1.data_loader):
                        print("valid_step1", valid_step)
                        x_valid, y_valid = valid_batch

                        valid_loss_task1, valid_prec1_task1, valid_prec5_task1, _ = validator1(
                            x=x_valid, y=y_valid)

                        avg_valid_loss_task1.append(valid_loss_task1)
                        avg_valid_prec1_task1.append(valid_prec1_task1)
                        avg_valid_prec5_task1.append(valid_prec5_task1)
                    avg_valid_loss_task1 = np.mean(avg_valid_loss_task1)
                    avg_valid_prec1_task1 = np.mean(avg_valid_prec1_task1)
                    avg_valid_prec5_task1 = np.mean(avg_valid_prec5_task1)
                    writer.add_scalar(
                        'valid/loss_task1',
                        avg_valid_loss_task1,
                        global_step)
                    writer.add_scalar(
                        'valid/precision1_task1',
                        avg_valid_prec1_task1,
                        global_step)
                    writer.add_scalar(
                        'valid/precision5_task1',
                        avg_valid_prec5_task1,
                        global_step)

                    print()
                    print('---------VALID TASK 1 STEP---------')
                    print("{0:<30}: {1:}".format('epoch', epoch))
                    print("{0:<30}: {1:}".format('global_step', global_step))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_loss_task1',
                            avg_valid_loss_task1))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec1_task1',
                            avg_valid_prec1_task1))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec5_task1',
                            avg_valid_prec5_task1))
                    print('-----------------------------------')
                    print()

                # task 2
                validator2.model.eval()
                with torch.no_grad():
                    avg_valid_loss_task2 = []
                    avg_valid_prec1_task2 = []
                    avg_valid_prec5_task2 = []
                    # for valid_step, valid_batch in
                    # enumerate(tqdm(validator2.data_loader,
                    # desc=validator2.name)):
                    for valid_step, valid_batch in enumerate(
                            validator2.data_loader):
                        print("valid_step2", valid_step)
                        x_valid, y_valid = valid_batch

                        valid_loss_task2, valid_prec1_task2, valid_prec5_task2, _ = validator2(
                            x=x_valid, y=y_valid)

                        avg_valid_loss_task2.append(valid_loss_task2)
                        avg_valid_prec1_task2.append(valid_prec1_task2)
                        avg_valid_prec5_task2.append(valid_prec5_task2)
                    avg_valid_loss_task2 = np.mean(avg_valid_loss_task2)
                    avg_valid_prec1_task2 = np.mean(avg_valid_prec1_task2)
                    avg_valid_prec5_task2 = np.mean(avg_valid_prec5_task2)
                    writer.add_scalar(
                        'valid/loss_task2',
                        avg_valid_loss_task2,
                        global_step)
                    writer.add_scalar(
                        'valid/precision1_task2',
                        avg_valid_prec1_task2,
                        global_step)
                    writer.add_scalar(
                        'valid/precision5_task2',
                        avg_valid_prec5_task2,
                        global_step)

                    print()
                    print('---------VALID TASK 2 STEP---------')
                    print("{0:<30}: {1:}".format('epoch', epoch))
                    print("{0:<30}: {1:}".format('global_step', global_step))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_loss_task2',
                            avg_valid_loss_task2))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec1_task2',
                            avg_valid_prec1_task2))
                    print(
                        "{0:<30}: {1:}".format(
                            'avg_valid_prec5_task2',
                            avg_valid_prec5_task2))
                    print('-----------------------------------')
                    print()

                valid_sec_per_validation = time.time() - valid_start_time
                writer.add_scalar(
                    'meta/sec-per-validation',
                    valid_sec_per_validation,
                    epoch)

            train_start_time = time.time()

            # train step task1
            try:
                train_batch1 = next(it1)
            except StopIteration:
                train_batch1 = None

            if train_batch1 is not None:
                x_train1, y_train1 = train_batch1
                train_loss_task1, train_prec1_task1, train_prec5_task1, _ = trainer1(
                    x=x_train1, y=y_train1)

                writer.add_scalar(
                    'train/loss_task1',
                    train_loss_task1,
                    global_step)
                writer.add_scalar(
                    'train/precision1_task1',
                    train_prec1_task1,
                    global_step)
                writer.add_scalar(
                    'train/precision5_task1',
                    train_prec5_task1,
                    global_step)

                print()
                print('---------TRAIN TASK 1 STEP---------')
                print("{0:<30}: {1:}".format('epoch', epoch))
                print("{0:<30}: {1:}".format('global_step', global_step))
                print(
                    "{0:<30}: {1:}".format(
                        'train_loss_task1',
                        train_loss_task1))
                print(
                    "{0:<30}: {1:}".format(
                        'train_prec1_task1',
                        train_prec1_task1))
                print(
                    "{0:<30}: {1:}".format(
                        'train_prec5_task1',
                        train_prec5_task1))
                print('-----------------------------------')
                print()

            # train step task2
            try:
                train_batch2 = next(it2)
            except StopIteration:
                train_batch2 = None

            if train_batch2 is not None:
                x_train2, y_train2 = train_batch2
                train_loss_task2, train_prec1_task2, train_prec5_task2, _ = trainer2(
                    x=x_train2, y=y_train2)

                writer.add_scalar(
                    'train/loss_task2',
                    train_loss_task2,
                    global_step)
                writer.add_scalar(
                    'train/precision1_task2',
                    train_prec1_task2,
                    global_step)
                writer.add_scalar(
                    'train/precision5_task2',
                    train_prec5_task2,
                    global_step)

                print()
                print('---------TRAIN TASK 2 STEP---------')
                print("{0:<30}: {1:}".format('epoch', epoch))
                print("{0:<30}: {1:}".format('global_step', global_step))
                print(
                    "{0:<30}: {1:}".format(
                        'train_loss_task2',
                        train_loss_task2))
                print(
                    "{0:<30}: {1:}".format(
                        'train_prec1_task2',
                        train_prec1_task2))
                print(
                    "{0:<30}: {1:}".format(
                        'train_prec5_task2',
                        train_prec5_task2))
                print('-----------------------------------')
                print()

            train_sec_per_iter = time.time() - train_start_time
            walltime += (time.time() - start_time) / (60.0**2)
            start_time = time.time()
            progress = (train_step + 1.0) / train_num_steps
            if use_scheduler:
                lr = trainer1.scheduler.get_lr()[0]
            else:
                lr = trainer1.learning_rate

            writer.add_scalar('meta/learning_rate', lr, global_step)
            writer.add_scalar('meta/progress', progress, global_step)
            writer.add_scalar(
                'meta/train-sec-per-iter',
                train_sec_per_iter,
                global_step)
            writer.add_scalar('meta/walltime', walltime, global_step)

            print()
            print('---------STEP COMPLETE---------')
            print("{0:<30}: {1:}".format('epoch', epoch))
            print("{0:<30}: {1:}".format('global_step', global_step))
            print("{0:<30}: {1:}".format('walltime', walltime))
            print("{0:<30}: {1:}".format('progress', progress))
            print("{0:<30}: {1:}".format('learning_rate', lr))
            print('----------------------------')
            print()

        # Save Model
        if ((epoch % save_freq) == 0):
            ckpt_data = {}
            ckpt_data['batch_size'] = config.batch_size
            ckpt_data['learning_rate'] = trainer1.learning_rate
            ckpt_data['momentum'] = config.momentum
            ckpt_data['step_size'] = config.step_size
            ckpt_data['weight_decay'] = config.weight_decay
            ckpt_data['walltime'] = walltime
            ckpt_data['epoch'] = epoch
            ckpt_data['state_dict'] = model.state_dict()
            ckpt_data['optimizer1'] = trainer1.optimizer.state_dict()
            ckpt_data['optimizer2'] = trainer2.optimizer.state_dict()
            helper.utils.tools_new.save_checkpoint(
                ckpt_data, config.checkpoints_dir, max_count=max_count)

        if use_scheduler:
            trainer1.scheduler.step(epoch=epoch)
            trainer2.scheduler.step(epoch=epoch)
        writer.add_scalar('meta/epoch', epoch, global_step)

    writer.close()


def predict(model, data_loader, ngpus, task=None, topk=1,
            notebook=False, max_batches=None, reduce_loss=True, seed=None):
    '''
    # todo: make dataloader includePaths optional
    '''

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    if ngpus > 0:
        criterion = criterion.cuda()
        softmax = softmax.cuda()

    model.eval()
    with torch.no_grad():
        losses = []
        nsteps = len(data_loader)

        if seed is not None:
            set_seed(seed)

        for step, batch in enumerate(data_loader):

            if max_batches is not None:
                if step == max_batches:
                    break

            x, y, paths = batch
            paths = np.array(paths)
            if ngpus > 0:
                y = y.cuda(non_blocking=True)
            if task is None:
                output = model(x=x)
                softmax_output = softmax(output)
            else:
                output = model(x=x, task=task)
                softmax_output = softmax(output)
            loss = criterion(output, y)

            y_prob, y_pred = torch.topk(input=softmax_output.data, k=topk)
            y, y_pred, y_prob = y.cpu().numpy(), y_pred.cpu().numpy(), y_prob.cpu().numpy()
            softmax_output = softmax_output.cpu().numpy()
            losses.append(loss.item())

            if step == 0:
                y_true_all = y
                y_pred_all = y_pred
                y_prob_all = y_prob
                paths_all = paths
                softmax_output_all = softmax_output
            else:
                y_true_all = np.concatenate((y_true_all, y), axis=0)
                y_pred_all = np.concatenate((y_pred_all, y_pred), axis=0)
                y_prob_all = np.concatenate((y_prob_all, y_prob), axis=0)
                paths_all = np.concatenate((paths_all, paths), axis=0)
                softmax_output_all = np.concatenate(
                    (softmax_output_all, softmax_output), axis=0)

        loss_ = np.array(losses)
        if reduce_loss:
            loss_ = np.mean(loss_)

    return y_true_all, y_pred_all, y_prob_all, paths_all, loss_, softmax_output_all


def predict_tqdm(model, data_loader, ngpus, task=None, topk=1,
                 notebook=False, max_batches=None, reduce_loss=True):

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    if ngpus > 0:
        criterion = criterion.cuda()
        softmax = softmax.cuda()

    if notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm

    model.eval()
    with torch.no_grad():
        losses = []
        nsteps = len(data_loader)

        for step, batch in enumerate(tqdm(data_loader, desc='predict')):

            if max_batches is not None:
                if step == max_batches:
                    break

            x, y, paths = batch
            paths = np.array(paths)
            if ngpus > 0:
                y = y.cuda(non_blocking=True)
            if task is None:
                output = model(x=x)
                softmax_output = softmax(output)
            else:
                output = model(x=x, task=task)
                softmax_output = softmax(output)
            loss = criterion(output, y)

            y_prob, y_pred = torch.topk(input=softmax_output.data, k=topk)
            y, y_pred, y_prob = y.cpu().numpy(), y_pred.cpu().numpy(), y_prob.cpu().numpy()
            softmax_output = softmax_output.cpu().numpy()
            losses.append(loss.item())

            if step == 0:
                y_true_all = y
                y_pred_all = y_pred
                y_prob_all = y_prob
                paths_all = paths
                softmax_output_all = softmax_output
            else:
                y_true_all = np.concatenate((y_true_all, y), axis=0)
                y_pred_all = np.concatenate((y_pred_all, y_pred), axis=0)
                y_prob_all = np.concatenate((y_prob_all, y_prob), axis=0)
                paths_all = np.concatenate((paths_all, paths), axis=0)
                softmax_output_all = np.concatenate(
                    (softmax_output_all, softmax_output), axis=0)

        loss_ = np.array(losses)
        if reduce_loss:
            loss_ = np.mean(loss_)

    return y_true_all, y_pred_all, y_prob_all, paths_all, loss_, softmax_output_all


def eval(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred)
    return acc, bal_acc, confusion_matrix, report


def drop_units(validator, features_layer=None, classifier_layer=None,
               num_gpus=0, notebook=False, max_batches=None, task=None, verbose=False):
    assert (features_layer is None or classifier_layer is None), "features_layer xor classifier_layer must be None"

    # progress bars
    if notebook:
        from tqdm import tnrange as trange

    else:
        from tqdm import trange as trange

    # Check that layers are from Conv or Linear layers only
    if features_layer is not None:
        assert (type(validator.model.module.features[features_layer]) is torch.nn.modules.conv.Conv2d), \
            "Must be of type Conv2D or Linear"
        num_units = validator.model.module.features[features_layer].weight.detach(
        ).cpu().numpy().shape[0]

    if classifier_layer is not None:
        assert (type(validator.model.module.classifier[classifier_layer]) is torch.nn.modules.linear.Linear), \
            "Must be of type Conv2D or Linear"
        num_units = validator.model.module.classifier[classifier_layer].weight.detach(
        ).cpu().numpy().shape[0]

    print("num_units:", num_units)

    losses = []

    for unit in trange(num_units, desc='drop_analysis'):

        if classifier_layer is not None:
            # cache unit
            cached_classifier_weights = validator.model.module.classifier[classifier_layer].weight[unit].detach(
            ).cpu().numpy()
            cached_classifier_bias = validator.model.module.classifier[classifier_layer].bias[unit].detach(
            ).cpu().numpy()
            # drop unit
            validator.model.module.classifier[classifier_layer].weight[unit] = 0.0
            validator.model.module.classifier[classifier_layer].bias[unit] = 0.0

        if features_layer is not None:
            # cache unit
            cached_features_weights = validator.model.module.features[features_layer].weight[unit].detach(
            ).cpu().numpy()
            cached_features_bias = validator.model.module.features[features_layer].bias[unit].detach(
            ).cpu().numpy()
            # drop unit
            validator.model.module.features[features_layer].weight[unit] = 0.0
            validator.model.module.features[features_layer].bias[unit] = 0.0

        # predict w/ out unit
        _, _, _, _, loss, _ = predict(model=validator.model,
                                      data_loader=validator.data_loader,
                                      ngpus=num_gpus,
                                      task=task,
                                      topk=1,
                                      notebook=notebook,
                                      max_batches=max_batches,
                                      reduce_loss=False)

        if unit == 0:
            losses_all = np.expand_dims(loss, axis=0)
        else:
            losses_all = np.concatenate(
                (losses_all, np.expand_dims(loss, axis=0)), axis=0)

        if verbose:
            print('avg loss:', np.mean(loss))

        if classifier_layer is not None:
            # replace unit
            validator.model.module.classifier[classifier_layer].weight[unit] = torch.from_numpy(
                cached_classifier_weights)
            validator.model.module.classifier[classifier_layer].bias[unit] = torch.from_numpy(
                cached_classifier_bias)

        if features_layer is not None:
            # replace unit
            validator.model.module.features[features_layer].weight[unit] = torch.from_numpy(
                cached_features_weights)
            validator.model.module.features[features_layer].bias[unit] = torch.from_numpy(
                cached_features_bias)

    return np.transpose(losses_all)


def keep_unit(validator, num_classes_task1, features_layer=None, classifier_layer=None,
              num_gpus=0, notebook=False, max_batches=None, task=None, verbose=False):

    if notebook:
        from tqdm import tnrange as trange
    else:
        from tqdm import trange as trange

    assert (features_layer is None or classifier_layer is None), "features_layer xor classifier_layer must be None"

    # Check that layers are from Conv or Linear layers only
    if features_layer is not None:
        assert (type(validator.model.module.features[features_layer]) is torch.nn.modules.conv.Conv2d), \
            "Must be of type Conv2D or Linear"
        print('dropping conv unit')
        model_shape = validator.model.module.features[features_layer].weight.detach(
        ).cpu().numpy().shape

    if classifier_layer is not None:
        assert (type(validator.model.module.classifier[classifier_layer]) is torch.nn.modules.linear.Linear), \
            "Must be of type Conv2D or Linear"
        print('classifier linear unit')
        model_shape = validator.model.module.classifier[classifier_layer].weight.detach(
        ).cpu().numpy().shape

    num_units = model_shape[0]
    print("num_units:", num_units)

    drop_model = copy.deepcopy(validator.model)

    mean_differences = []

    for unit in trange(num_units, desc='keep_analysis'):

        if classifier_layer is not None:
            # cache unit
            cached_classifier_weights = validator.model.module.classifier[classifier_layer].weight[unit].detach(
            ).cpu().numpy()
            # drop all units
            drop_model.module.classifier[classifier_layer].weight[:] = torch.from_numpy(
                np.zeros((model_shape))).cuda()
            # replace single unit
            drop_model.module.classifier[classifier_layer].weight[unit] = torch.from_numpy(
                cached_classifier_weights)

        if features_layer is not None:
            # cache unit
            cached_features_weights = validator.model.module.features[features_layer].weight[unit].detach(
            ).cpu().numpy()
            cached_features_bias = validator.model.module.features[features_layer].bias[unit].detach(
            ).cpu().numpy()

            # drop all units
            drop_model.module.features[features_layer].weight[:] = torch.from_numpy(
                np.zeros((model_shape))).cuda()
            drop_model.module.features[features_layer].bias[:] = torch.from_numpy(
                np.zeros((num_units))).cuda()

            # replace single unit
            drop_model.module.features[features_layer].weight[unit] = torch.from_numpy(
                cached_features_weights)
            drop_model.module.features[features_layer].bias[unit] = torch.from_numpy(
                cached_features_bias)

        # predict w/ out unit
        _, _, _, _, loss, softmax = predict(model=drop_model,
                                            data_loader=validator.data_loader,
                                            ngpus=num_gpus,
                                            task=task,
                                            topk=1,
                                            notebook=notebook,
                                            max_batches=max_batches)

        # todo: take the mean of the differences instead of the difference of the means
        # task1_mean_prob = np.max(softmax[:,:num_classes_task1],axis=1)
        # task2_mean_prob = np.max(softmax[:, num_classes_task1:],axis=1)
        # soft_differences =  task1_mean_prob - task2_mean_prob

        # if unit==0:
        #    soft_differences_all = np.expand_dims(soft_differences,axis=0)
        # else:
        #    soft_differences_all = np.concatenate((soft_differences_all, np.expand_dims(soft_differences,axis=0)), axis=0)

        if unit == 0:
            losses_all = np.expand_dims(loss, axis=0)
        else:
            losses_all = np.concatenate(
                (losses_all, np.expand_dims(loss, axis=0)), axis=0)

        if verbose:
            print('avg loss:', np.mean(loss))

        # mean_difference = np.mean(soft_differences)
        # mean_differences.append(mean_difference)
        # if verbose:
        #    print(mean_difference)

    return np.transpose(losses_all)


def gradients_and_activations(activations=False, gradients=False, max_batches=None,
                              validator=None, trainer=None, notebook=False, task=None):
    assert task is None or task == 1 or task == 2, "Task must be one of either None, 1, or 2"
    assert (trainer is None or validator is None), "Either trainer or validator must be None, but not both"

    if notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm

    all_activations, all_gradients = [], []

    if trainer is not None:
        data_loader = trainer.data_loader
    else:
        data_loader = validator.data_loader

    for step, batch in enumerate(tqdm(data_loader, desc='act/grad')):
        if max_batches is not None and step == max_batches:
            break
        x, y = batch
        if trainer is not None:
            if trainer.ngpus > 0:
                x = x.cuda(non_blocking=True)
            trainer(x=x, y=y,
                    update_weights=False,
                    activations=activations,
                    gradients=gradients)

            activation = trainer.model.module.activations if task is None else\
                trainer.model.module.get_activations(task=task)
            for i in range(len(activation)):
                print('before')
                print(activation[i])
                activation[i] = activation[i].cpu().numpy()
                print('after')
                print(activation[i])
            torch.cuda.empty_cache()
            all_activations.append(activation)

            gradient = trainer.model.module.get_activations_gradient() if task is None else\
                trainer.model.module.get_activations_gradient(task=task)
            all_gradients.append(gradient)

            torch.cuda.empty_cache()

            assert (len(all_activations[step]) == len(
                all_gradients[step])), "Something is wrong with activations or gradients, tell Julio"
        else:
            if validator.ngpus > 0:
                x = x.cuda(non_blocking=True)
            validator.model.eval()
            with torch.no_grad():
                validator(x=x, y=y, activations=activations, gradients=False)
            activation = validator.model.module.activations if task is None else\
                validator.model.module.get_activations(task=task)
            # print(len(activation))
            for i in range(len(activation)):
                # print('before')
                # print(activation[i])
                activation[i] = activation[i].cpu().numpy()
                # print('after')
                # print(activation[i])
                pass  # to collapse comments above
            torch.cuda.empty_cache()
            all_activations.append(activation)

            gradient = validator.model.module.get_activations_gradient() if task is None else\
                validator.model.module.get_activations_gradient(task=task)
            all_gradients.append(gradient)

            assert (
                len(all_gradients[step]) == 0), "Something is wrong with gradients, tell Julio"

    return all_activations, all_gradients

# being used by train_single.py


class ImageNetTrain(object):

    def __init__(self, model, name, data_dir, max_samples, batch_size,
                 workers, lr, momentum, weight_decay, step_size, ngpus):
        self.name = name
        self.model = model
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.ngpus = ngpus
        self.max_samples = max_samples
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        self.lr = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size)  # default decay by 0.1
        self.loss = torch.nn.CrossEntropyLoss()
        if self.ngpus > 0:
            self.loss = self.loss.cuda()  # for cuda GPU support

    def data(self):
        ImageFolder = folder.ImageFolder
        normalize = torchvision.transforms.Normalize(
            mean=[0.5] * 3, std=[0.5] * 3)
        dataset = ImageFolder(root=os.path.join(self.data_dir, 'train'),
                              max_samples=self.max_samples,
                              transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                                        torchvision.transforms.RandomResizedCrop(
                                                                            224),
                                                                        torchvision.transforms.RandomGrayscale(
                                                                            p=0.2),
                                                                        torchvision.transforms.ToTensor(),
                                                                        normalize,]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.workers,
                                                  pin_memory=True)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        if self.ngpus > 0:
            target = target.cuda(non_blocking=True)
        output = self.model(inp)

        record = {}
        loss = self.loss(output, target)
        loss_ = loss.item()

        top1, top5 = metrics.precision(output, target, topk=(1, 5))
        top1 /= len(output)
        top5 /= len(output)
        lr = self.lr.get_lr()[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        dur = time.time() - start
        return loss_, top1, top5, dur, lr

# being used by train_single.py


class ImageNetVal(object):

    def __init__(self, model, name, data_dir,
                 max_samples, batch_size, workers, ngpus):
        self.name = name
        self.model = model
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.ngpus = ngpus
        self.max_samples = max_samples
        self.data_loader = self.data()
        self.loss = torch.nn.CrossEntropyLoss(size_average=False)
        if self.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        ImageFolder = folder.ImageFolder
        normalize = torchvision.transforms.Normalize(
            mean=[0.5] * 3, std=[0.5] * 3)
        dataset = ImageFolder(root=os.path.join(self.data_dir, 'test'),
                              max_samples=self.max_samples,
                              transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                                        torchvision.transforms.CenterCrop(
                                                                            224),
                                                                        torchvision.transforms.ToTensor(),
                                                                        normalize]))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.workers,
                                                  pin_memory=True)
        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        loss, top1, top5 = 0., 0., 0.
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                if self.ngpus > 0:
                    target = target.cuda(non_blocking=True)
                output = self.model(inp)

                loss += self.loss(output, target).item()
                p1, p5 = metrics.precision(output, target, topk=(1, 5))
                top1 += p1
                top5 += p5

        loss /= len(self.data_loader.dataset.samples)
        top1 /= len(self.data_loader.dataset.samples)
        top5 /= len(self.data_loader.dataset.samples)

        dur = (time.time() - start) / len(self.data_loader)
        return loss, top1, top5, dur

# recursive defintion (hope it works!)


def getLayerList(child, depth=-1, node=None):
    '''
    Description:
        Recursively traverses all modules and children of a pytorch model.
        Modifies the list layerList by adding layers in order of appearance in the model tree all leaf layers
    Inputs:
        child - model.module (or any sub-module of a pytorch model)
        depth - default at -1, only set/updated during recursive call, tree depth value
        node  - default at None, only set/updated during recursive call, leaf node value
    Returns:
        None, however, layerTree = [] must exist and be empty to work properly
        layerTree is modified in place as a global variable
    '''
    if isinstance(child, list) == False:
        children_generator = list(child.children())
    else:
        child_generator = child
    num_children = len(children_generator)

    if num_children > 0:
        # children check passed -> not a leaf node
        for n, child_ in enumerate(children_generator):
            getLayerList(child=child_, depth=depth + 1, node=n)
    else:
        # children check failed -> leaf node
        try:
            # if children check failed check if it has weight
            child.weight
            try:
                child.bias
                WeightandBiasPair = True
                layerList.append(
                    (type(child), True, depth, node, WeightandBiasPair))
            except AttributeError:
                WeightandBiasPair = False
                layerList.append(
                    (type(child), True, depth, node, WeightandBiasPair))
        except AttributeError:
            # if weight check failed is nonparam layer
            WeightandBiasPair = False
            layerList.append(
                (type(child), False, depth, node, WeightandBiasPair))


def get_weight_and_bias_pairs(model):

    # weight & bias pairs
    # weight_and_bias_pairs = []
    # parameters = list(model.parameters())

    # i=0 # ParamWeightandBiasIndexPair
    # while i < len(parameters)-1:
    #    if (parameters[i].shape[0] == parameters[i+1].shape[0]) and \
    #    (len(parameters[i].shape)> 1) and (len(parameters[i+1].shape)==1):
    #        weight_and_bias_pairs.append((i,i+1))
    #        i+=2
    #    else:
    #        i+=1

    named_params = list(model.named_parameters())
    weight_and_bias_pairs = []
    i = 0
    while i < len(named_params):
        name = named_params[i][0]
        next_name = named_params[i + 1][0]
        # print(name, next_name)
        if 'weight' in name:
            weight_index = i
        bias_index = None
        if 'bias' in next_name:
            bias_index = i + 1
            i += 1
        weight_and_bias_pairs.append((weight_index, bias_index))
        i += 1

    return weight_and_bias_pairs


def getLayerMapping(model):
    '''
    Description:
        Get a layber mapping for any model
    Inputs:
        mode - pytorch model using torch.nn
    Returns:
        layerMapping - python dictionary mapping keys to arrays
            layerList - python list of layers sorted by order of appearance in model tree
            ParamWeightAndBiasGroupIndex2ParamWeightAndBiasIndexPair - i2j where i is the index for this list and j is the index in list(model.parameters())
    '''
    global layerList
    layerList = []

    getLayerList(model.module)

    # param_layers, param_types, param_pairs, nonparam_layers, nonparam_types
    param_types = []
    param_layers = []
    nonparam_types = []
    nonparam_layers = []
    for i, leaf in enumerate(layerList):
        if leaf[1] == True:
            param_types.append(leaf[0])
            param_layers.append(i)
        else:
            nonparam_types.append(leaf[0])
            nonparam_layers.append(i)

    weight_and_bias_pairs = get_weight_and_bias_pairs(model)

    layerMapping = {'ParamGroupIndex2LayerIndex': np.array(param_layers),
                    'ParamGroupIndex2LayerType': np.array(param_types),
                    'ParamWeightAndBiasGroupIndex2ParamWeightAndBiasIndexPair': np.array(weight_and_bias_pairs),
                    'NonparamGroupIndex2LayerIndex': np.array(nonparam_layers),
                    'NonparamGroupIndex2LayerType': np.array(nonparam_types),
                    'LayerList': np.array(layerList)}

    return layerMapping


def getWeightandBias(model, layerMap, paramWeightAndBiasGroupIndex):
    '''
    Description:
        returns the weight and bias for index of param layer
    Inputs:
        model    - torch.nn.module based model
        layerMap - dictionary containing, layerMap['Index2ParamIndexPair']
                   maps index to weight and bias pair, i.e. ith param pair
                   has a specific index pair in model.parameters()
        index    - the relative index, if you want the first param pair index is 0
                                       if you want the second param pair index is 1
                                       and so on
                   0 <= index <= len(layerMap['Index2ParamIndexPair'])

    '''
    key = 'ParamWeightAndBiasGroupIndex2ParamWeightAndBiasIndexPair'
    weightAndBiasIndexPair = layerMap[key][paramWeightAndBiasGroupIndex]
    weight_index, bias_index = weightAndBiasIndexPair
    tempParameters = list(model.parameters())
    if weight_index is not None:
        weight = tempParameters[weight_index]
    bias = None
    if bias_index is not None:
        bias = tempParameters[bias_index]
    return weight, bias


def printArguments(config=None, validator=None, mode=None, FLAGS=None):
    '''
    Description: Helper function for printing out common FLAG arguments and Samples
                    data from validators/trainers.

    '''
    def printFormat(name, var):
        print("{0:<40}: {1:}".format(name, var), flush=True)

    if FLAGS is not None:
        print(flush=True)
        print('---------------FLAGS------------------', flush=True)
        print('--------------------------------------', flush=True)
        for key, value in FLAGS.__dict__.items():
            printFormat(key, value)
        print('--------------------------------------', flush=True)
        print(flush=True)

    if validator is not None:
        assert (mode == 'valid' or mode ==
                'train'), 'mode should be either "valid" or "train", otherwise, update code'
        if mode == 'valid':
            max_samples = config.max_valid_samples
        elif mode == 'train':
            max_samples = config.max_train_samples

        print('----------NUMBER OF SAMPLES-----------', flush=True)
        print('--------------------------------------', flush=True)
        printFormat('validator.name', validator.name)
        print(flush=True)
        print('Samples by torch.dataset:', flush=True)
        print('-------------------------', flush=True)
        num_steps = len(validator.data_loader)
        printFormat('num_steps', num_steps)
        printFormat('~num_samples', num_steps * config.batch_size)
        print(flush=True)
        print('Samples by Configuration:', flush=True)
        print('-------------------------', flush=True)
        samplesCount = 0
        for key in validator.dataset.task_to_num_classes.keys():
            count = validator.dataset.task_to_num_classes[key] * \
                max_samples[key]
            samplesCount += count
            printFormat(key, count)
            printFormat(
                '--num_classes',
                validator.dataset.task_to_num_classes[key])
            print(flush=True)
        printFormat('total', samplesCount)
        print('--------------------------------------', flush=True)
        print(flush=True)


def displayNetworkMap(model, verbose=False):

    pd.set_option('display.max_rows', None)

    layerMapping = getLayerMapping(model)
    data = []
    WeightandBiasPairIndex = 0
    for i, layer in enumerate(layerMapping['LayerList']):
        depth = layer[2]
        structure = str(depth) + ' . ' * (2 * depth) + ' |-'
        node = layer[3]
        location = str(node) + ' -' * (node * 1)
        isWeightandBiasPair = layer[4]
        j = None
        if isWeightandBiasPair:
            j = WeightandBiasPairIndex
            WeightandBiasPairIndex += 1

        param = False
        paramIndex = None
        if i in layerMapping['ParamGroupIndex2LayerIndex']:
            param = True
            paramIndex = np.argwhere(
                layerMapping['ParamGroupIndex2LayerIndex'] == i)[0, 0]
            data.append([str(layer[0])[25:-2], structure,
                        location, param, paramIndex, i, j])

        nonParam = False
        nonParamIndex = None
        if i in layerMapping['NonparamGroupIndex2LayerIndex']:
            nonParam = True
            nonParamIndex = np.argwhere(
                layerMapping['NonparamGroupIndex2LayerIndex'] == i)[0, 0]

            data.append([str(layer[0])[25:-2], structure,
                        location, param, nonParamIndex, i, j])

    df = pd.DataFrame(
        data,
        columns=[
            'LayerType',
            'Depth',
            'Node',
            'IsParam',
            'GroupIndex',
            'LayerIndex',
            'WeightandBiasPairIndex'])

    table = tabulate(df, showindex=False, headers=df.columns)
    if verbose:
        print(table)

    return table


def drop_layer_units(model, param_group_index, units):

    # create cache
    index = param_group_index
    cache = {'W': {index: {}}, 'b': {index: {}}}

    # get mapping
    # -------------------------
    layerMap = getLayerMapping(model)
    weight, bias = getWeightandBias(
        model=model, layerMap=layerMap, paramWeightAndBiasGroupIndex=index)
    print(weight.shape)
    print(bias.shape)

    # cache units
    # ---------------
    for unit in units:
        cache['W'][index][unit] = weight[unit].detach().cpu().numpy()
        if bias is not None:
            cache['b'][index][unit] = bias[unit].detach().cpu().numpy()

    # drop units
    # ---------------
    for unit in units:
        weight[unit] = 0.0
        if bias is not None:
            bias[unit] = 0.0

    return cache


def replace_layer_units(model, param_group_index, units, cache):

    # get mapping
    # -------------------------
    index = param_group_index
    layerMap = getLayerMapping(model)
    weight, bias = getWeightandBias(
        model=model, layerMap=layerMap, paramWeightAndBiasGroupIndex=index)

    # replace units from cache
    # -------------------------
    for unit in units:
        weight[unit] = torch.from_numpy(cache['W'][index][unit])
        if bias is not None:
            bias[unit] = torch.from_numpy(cache['b'][index][unit])
    return None


def get_base_predictions(validator, ngpus):
    y_true, y_pred, _, paths, _, _ = predict(model=validator.model,
                                             data_loader=validator.data_loader,
                                             ngpus=ngpus,
                                             task=None,
                                             topk=1,
                                             notebook=True,
                                             max_batches=None,
                                             reduce_loss=False)

    y_pred = np.squeeze(y_pred)
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    return accuracy, y_true, y_pred, paths


def get_match_samples(validator, match_accuracy, ngpus, verbose=False):
    try:
        validator.dataset.matched
    except AttributeError:
        pass
    else:
        assert validator.dataset.matched == False, "samples have already been matched"

    accuracy, y_true, y_pred, paths = get_base_predictions(validator, ngpus)
    if verbose:
        print('base accuracy before matching:', accuracy)

    print('number of examples before matching: ',
          len(validator.dataset.samples))

    # sort
    sorted_indexes = np.argsort(paths)
    y_true = y_true[sorted_indexes]
    y_pred = y_pred[sorted_indexes]
    paths = paths[sorted_indexes]

    # shuffle
    num_samples = len(y_true)
    shuffle_indexes = np.arange(num_samples)
    np.random.shuffle(shuffle_indexes)
    y_true = y_true[shuffle_indexes]
    y_pred = y_pred[shuffle_indexes]
    paths = paths[shuffle_indexes]

    if match_accuracy > accuracy:  # need to increase accuracy by getting rid of errors
        # calculate number of removals
        num_removals = np.round(
            num_samples * (match_accuracy - accuracy) / match_accuracy).astype(int)
        false_predictions = np.where(y_true != y_pred)[0]
        removal_indexes = false_predictions[:num_removals]

        # remove
        new_y_true = np.delete(y_true, removal_indexes).tolist()
        new_y_pred = np.delete(y_pred, removal_indexes).tolist()
        new_paths = np.delete(paths, removal_indexes).tolist()

        new_accuracy = metrics.accuracy_score(
            y_true=new_y_true, y_pred=new_y_pred)
        # print('new accuracy:', new_accuracy)
        assert isclose(match_accuracy, new_accuracy, abs_tol=1e-1)

    elif match_accuracy < accuracy:  # need to decrease accuracy by getting rid of correct predictions
        # calculate number of removals
        num_removals = np.round(
            num_samples * (match_accuracy - accuracy) / (match_accuracy - 1.0)).astype(int)
        correct_predictions = np.where(y_true == y_pred)[0]
        removal_indexes = correct_predictions[:num_removals]

        # remove
        new_y_true = np.delete(y_true, removal_indexes).tolist()
        new_y_pred = np.delete(y_pred, removal_indexes).tolist()
        new_paths = np.delete(paths, removal_indexes).tolist()

        new_accuracy = metrics.accuracy_score(
            y_true=new_y_true, y_pred=new_y_pred)
        # print('new accuracy:', new_accuracy)
        assert isclose(match_accuracy, new_accuracy, abs_tol=1e-1)

    samples = np.array(validator.dataset.samples)
    samples, labels = samples[:, 0], samples[:, 1].astype(int)
    match_samples = []
    removed_samples = []
    for i in range(len(samples)):
        if samples[i] in new_paths:
            match_samples.append((samples[i], labels[i]))
        else:
            removed_samples.append((samples[i], labels[i]))

    validator.dataset.cached_samples = validator.dataset.samples
    validator.dataset.samples = match_samples
    validator.dataset.removed_samples = match_samples
    validator.dataset.matched = True

    print('number of examples after matching: ',
          len(validator.dataset.samples))
    return None


def restore_samples(validator):
    assert validator.dataset.matched == True, "samples have already been restored"
    validator.dataset.samples = validator.dataset.cached_samples
    del validator.dataset.cached_samples
    del validator.dataset.removed_samples
    validator.dataset.matched = False
    return None


def saving_setup(config, save_freq, keep_epochs):
    if config.keep_epochs is not None:
        keep_epochs.update(set(config.keep_epochs))

    if config.saving_function == 'step':
        assert (
            'step' in config.saving_params and config.saving_params['step'] > 0), "saving step must be positive"
        step = config.saving_params['step']
        assert ('offset' in config.saving_params and config.saving_params['offset'] >= 0 and
                config.saving_params['offset'] < step), "saving offset doesn't satisfy 0<=offset<step"
        offset = config.saving_params['offset']
        def function(e): return (e - offset) % step == 0
    else:
        assert (save_freq > 0), "save_freq must be positive"
        print(f"\nDefaulting to saving every epoch multiple of {save_freq}\n")
        def function(e): return e % save_freq == 0
    return lambda e: function(e) or e in keep_epochs
