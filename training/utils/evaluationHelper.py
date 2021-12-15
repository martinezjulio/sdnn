#!/usr/bin/env python
# coding: utf-8

# imports
from utils import helper 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
import scipy
import numpy as np

import jsonlines as jsonl


def get_pred_task_dual_net(network_name, lesion_name, version_name, task_sort_name, task_nonsort_name, layer):
    
    # 'INDEX_12/SORTEDBY_data_car/PERCENT_0.20/y_pred'
    
    filename = './lesions/' + network_name + '/' + lesion_name + '/drop_percents_records/VERSION' + version_name + '/EVALUATION_TASK_' + task_sort_name  + '/predictions/PARAM_GROUP/predictions.jsonl'
    
    perc = '0.00'

    reader = jsonl.open(filename)
    pred_data = {}
    for obj in reader:
        key = list(obj.keys())[0]
        #print(key)
        pred_data[key]=obj[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_0_true = np.zeros(shape=(3,y_true.shape[0]))
    task_0_pred = np.zeros(shape=(3,y_true.shape[0]))

    task_0_true[0,:] = y_true
    task_0_pred[0,:] = y_pred

    perc = '0.20'

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_0_true[1,:] = y_true
    task_0_pred[1,:] = y_pred

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_nonsort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_nonsort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_0_true[2,:] = y_true
    task_0_pred[2,:] = y_pred


    ### task 1 ###

    filename = './lesions/' + network_name + '/' + lesion_name + '/drop_percents_records/VERSION' + version_name + '/EVALUATION_TASK_' + task_nonsort_name  + '/predictions/PARAM_GROUP/predictions.jsonl'
    
    perc = '0.00'

    reader = jsonl.open(filename)
    pred_data = {}
    for obj in reader:
        key = list(obj.keys())[0]
        pred_data[key]=obj[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_1_true = np.zeros(shape=(3,y_true.shape[0]))
    task_1_pred = np.zeros(shape=(3,y_true.shape[0]))

    task_1_true[0,:] = y_true
    task_1_pred[0,:] = y_pred

    perc = '0.20'

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_nonsort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_nonsort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_1_true[1,:] = y_true
    task_1_pred[1,:] = y_pred

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_1_true[2,:] = y_true
    task_1_pred[2,:] = y_pred
    
    return task_0_true, task_0_pred, task_1_true, task_1_pred

def get_pred_dual_net(network_name, lesion_name, task_sort_name, task_nonsort_name, layer):
    
    version = 'evalPred'
    
    # 'INDEX_12/SORTEDBY_data_car/PERCENT_0.20/y_pred'

    filename = './lesions/' + network_name + '/' + lesion_name + '/drop_percents_records/VERSION__' + version + '/EVALUATION_TASK_' + task_sort_name  + '/predictions/PARAM_GROUP/predictions.jsonl'

    perc = '0.00'

    reader = jsonl.open(filename)
    pred_data = {}
    for obj in reader:
        key = list(obj.keys())[0]
        #print(key)
        pred_data[key]=obj[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true_0 = np.asarray(y_true)
    y_pred_0 =  np.asarray(y_pred)

    ### task 1 ###

    filename = './lesions/' + network_name + '/' + lesion_name + '/drop_percents_records/VERSION__' + version + '/EVALUATION_TASK_' + task_nonsort_name  + '/predictions/PARAM_GROUP/predictions.jsonl'
    
    perc = '0.00'

    reader = jsonl.open(filename)
    pred_data = {}
    for obj in reader:
        key = list(obj.keys())[0]
        #print(key)
        pred_data[key]=obj[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true_1 = np.asarray(y_true)
    y_pred_1 =  np.asarray(y_pred)

    return y_true_0, y_pred_0, y_true_1, y_pred_1

def get_pred_dual_net_old(network_name, lesion_name, task_sort_name, task_nonsort_name, layer):
    
    # 'INDEX_12/SORTEDBY_data_car/PERCENT_0.20/y_pred'
    version = 'evalPred'

    filename = './lesions/' + network_name + '/' + lesion_name + '/drop_percents_records/VERSION__' + version + '/EVALUATION_TASK_' + task_sort_name  + '/predictions/PARAM_GROUP/predictions.jsonl'

    perc = '0.00'

    reader = jsonl.open(filename)
    pred_data = {}
    for obj in reader:
        key = list(obj.keys())[0]
        #print(key)
        pred_data[key]=obj[key]

    key = 'INDEX_' + layer + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true_0 = np.asarray(y_true)
    y_pred_0 =  np.asarray(y_pred)

    ### task 1 ###

    filename = './lesions/' + network_name + '/' + lesion_name + '/drop_percents_records/VERSION__' + version + '/EVALUATION_TASK_' + task_nonsort_name  + '/predictions/PARAM_GROUP/predictions.jsonl'
    
    perc = '0.00'

    reader = jsonl.open(filename)
    pred_data = {}
    for obj in reader:
        key = list(obj.keys())[0]
        #print(key)
        pred_data[key]=obj[key]

    key = 'INDEX_' + layer + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true_1 = np.asarray(y_true)
    y_pred_1 =  np.asarray(y_pred)

    return y_true_0, y_pred_0, y_true_1, y_pred_1


def get_pred_single_net(network_name):
    
    # 'INDEX_12/SORTEDBY_data_car/PERCENT_0.20/y_pred'

    filename = './evaluations/' + network_name + '/predictions.jsonl'

    reader = jsonl.open(filename)
    pred_data = {}
    for obj in reader:
        key = list(obj.keys())[0]
        #print(key)
        pred_data[key]=obj[key]

    key = 'y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.squeeze(np.asarray(y_true))
    y_pred =  np.squeeze(np.asarray(y_pred))

    return y_true, y_pred


def get_bootstrap_prop_acc(task_true, task_pred):

    rand_classes = np.unique(task_true[0,:])
    n_classes = rand_classes.shape[0]

    n_boot = 1000

    prop_acc_boot_1 = np.zeros(shape=(n_boot,))
    prop_acc_boot_2 = np.zeros(shape=(n_boot,))

    for iBoot in range(n_boot):

        if np.mod(iBoot,100) == 0:
            print('iteration: ' + str(iBoot))
        y_true_boot = []
        y_pred_boot_1 = []
        y_pred_boot_2 = []
        y_pred_base_boot = []

        # bootstrap classes
        bootstrap_classes = np.random.choice(rand_classes, size=n_classes, replace=True)

        for i,iClass in enumerate (bootstrap_classes):
            class_images = np.where(task_true[0,:]==iClass)[0]

            bootstrap_images = np.random.choice(class_images, size=class_images.shape[0], replace=True)

            if i==1:
                #print(class_images)
                #print(bootstrap_images)
                y_true_boot = np.ones(shape=(class_images.shape[0],))*iClass
                y_pred_base_boot = task_pred[0,bootstrap_images]
                y_pred_boot_1 = task_pred[1,bootstrap_images]
                y_pred_boot_2 = task_pred[2,bootstrap_images]
            else:
                y_true_boot = np.hstack((y_true_boot, np.ones(shape=(class_images.shape[0],))*iClass))
                y_pred_base_boot = np.hstack((y_pred_base_boot, task_pred[0,bootstrap_images]))
                y_pred_boot_1 = np.hstack((y_pred_boot_1, task_pred[1,bootstrap_images]))
                y_pred_boot_2 = np.hstack((y_pred_boot_2, task_pred[2,bootstrap_images]))

        acc_1 = np.where(y_true_boot == y_pred_boot_1)[0].shape[0]/y_true_boot.shape[0]
        acc_2 = np.where(y_true_boot == y_pred_boot_2)[0].shape[0]/y_true_boot.shape[0]
        acc_base = np.where(y_true_boot == y_pred_base_boot)[0].shape[0]/y_true_boot.shape[0]
        # print(acc)

        prop_acc_boot_1[iBoot] = (acc_base - acc_1)/(acc_base)
        prop_acc_boot_2[iBoot] = (acc_base - acc_2)/(acc_base)

    print(np.mean(prop_acc_boot_1))
    print(np.std(prop_acc_boot_1))

    print(np.mean(prop_acc_boot_2))
    print(np.std(prop_acc_boot_2))
    
    return prop_acc_boot_1, prop_acc_boot_2

def get_bootstrap_acc(y_true, y_pred):

    rand_classes = np.unique(y_true)
    n_classes = rand_classes.shape[0]

    n_boot = 1000

    acc_boot = np.zeros(shape=(n_boot,))
    
    for iBoot in range(n_boot):

        if np.mod(iBoot,100) == 0:
            print('iteration: ' + str(iBoot))
        y_true_boot = []
        y_pred_boot = []
        
        # bootstrap classes
        bootstrap_classes = np.random.choice(rand_classes, size=n_classes, replace=True)

        for i,iClass in enumerate (bootstrap_classes):
            class_images = np.where(y_true==iClass)[0]

            bootstrap_images = np.random.choice(class_images, size=class_images.shape[0], replace=True)

            if i==1:
                y_true_boot = np.ones(shape=(class_images.shape[0],))*iClass
                y_pred_boot = y_pred[bootstrap_images]
            else:
                y_true_boot = np.hstack((y_true_boot, np.ones(shape=(class_images.shape[0],))*iClass))
                y_pred_boot = np.hstack((y_pred_boot, y_pred[bootstrap_images]))

        acc = np.where(y_true_boot == y_pred_boot)[0].shape[0]/y_true_boot.shape[0]
        
        acc_boot[iBoot] = acc
        
    print(np.mean(acc_boot))
    print(np.std(acc_boot))
    
    return acc_boot
