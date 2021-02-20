from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('project_path.ini')
sys_path = dict(cfg.items('PathSettings'))['sys_path']


import sys, os
sys.path.append(sys_path)
from shutil import copyfile

import numpy as np
import random
import time
import torch
import logging

from DatasetsLoad.sampler import SampleGenerator
from Models.engine import *
from Models.Page import *


def get_engine(Sampler, TrainSettings, ModelSettings):
    model_name = ModelSettings['model_name']
    engine_type = ModelSettings['engine_type']
    print("=========== model: " + model_name + " engine: " + engine_type + " ===========\n") 
    module = eval(model_name)
    model = getattr(module, model_name)(Sampler, ModelSettings)
    engine = eval(engine_type+'Engine')(model, TrainSettings)
    return engine

def save_model_code(source, target, code_name):
    try:
        copyfile(source, target)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)
    print(code_name+" code copy done!\n")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Run(DataSettings, ModelSettings, TrainSettings, ResultSettings,
        mode='train',
        timestamp=None):

    ## =========== setting init ===========
    setup_seed(817) # random seed
    model_name = ModelSettings['model_name']
    save_dir = sys_path + ResultSettings['save_dir'] + model_name + '/'

    ## =========== data  init ===========
    Sampler = SampleGenerator(DataSettings, TrainSettings, mode)

    ## =========== model init ===========
    Engine = get_engine(Sampler, TrainSettings, ModelSettings)


    ## ======= train || inference =======
    if timestamp == None:
        timestamp = time.time()
        localtime = str(time.asctime( time.localtime(int(timestamp)) ))  
        time_ids = str(localtime) + ' ' + str(int(timestamp))
        model_save_dir = save_dir+'files/'+time_ids+'/'
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        save_model_code(eval(model_name).__file__, model_save_dir+model_name+'.py', code_name='model')
    else:
        tbias = 8*3600
        # localtime = str(time.asctime( time.localtime(int(timestamp)) ))  
        localtime = str(time.asctime( time.localtime(int(timestamp)+tbias) ))  
        time_ids = str(localtime) + ' ' + str(int(timestamp))
        print(timestamp)
        print(localtime)
        model_save_dir = save_dir+'files/'+time_ids+'/'
        Engine.model = torch.load(f'{model_save_dir}{model_name}.pt').to(TrainSettings['device'])

    ## =========== log  init ===========
    log = logging.getLogger(model_name + str(int(timestamp)))
    log.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)

    ind_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ind_fh = logging.FileHandler(save_dir + model_name +
                                 '_%s_%s' % (localtime, str(int(timestamp))) + ".log")
    ind_fh.setFormatter(ind_formatter)

    log.addHandler(ind_fh)
    log.addHandler(sh)
    log.info('========== ' + localtime + " "+ str(int(timestamp)) + ' =========='+'\n')
    log.info(str(DataSettings)+'\n'+str(ModelSettings)+'\n'+str(TrainSettings)+'\n')


    if mode == "train": # train mode
        print("=========== Training Start ===========") 

        total_epoch = eval(TrainSettings['epoch'])
        start_epoch = eval(TrainSettings['start_epoch'])
        pre_train_epoch = eval(TrainSettings['pre_train_epoch'])
        early_stop_step = eval(TrainSettings['early_stop_step'])
        endure_count = 0

        val_auc, best_val_epoch = 0, 0
        best_val_result = ""

        # start training
        for epoch_i in range(start_epoch, total_epoch):
            ### train
            Engine.train(Sampler.Train_data, epoch_i)
            
            ### pre-validate train stage
            if epoch_i < pre_train_epoch:
                continue

            ### test 
            result, tmp_auc = Engine.evaluate(Sampler.Val_data, epoch_i)
            ### early stop
            if tmp_auc > val_auc:
                val_auc = tmp_auc
                endure_count = 0
                best_val_result = result
                best_val_epoch = epoch_i

                t_result, t_auc = Engine.evaluate(Sampler.Test_data, epoch_i)

                # save log
                log.info(str(int(timestamp))+' epoch: ' + str(epoch_i))
                log.info('better val_result:\n'+best_val_result+'\n')
                log.info('current test_result:\n'+t_result+'\n')
                # save model file
                torch.save(Engine.model, f'{model_save_dir}{model_name}.pt')
            else:
                endure_count += 1
            if endure_count >= early_stop_step:
                break
        log.info('best val results(epoch: '+str(best_val_epoch)+' timestamp: '+str(int(timestamp))+'):\n'+best_val_result+'\n')
        
        # start test
        print("Inference on the test dataset")
        Engine.model = torch.load(f'{model_save_dir}{model_name}.pt').to(TrainSettings['device'])
        result, _ = Engine.evaluate(Sampler.Test_data, epoch_id=0)
        test_result = result
        log.info('test results( timestamp: '+str(int(timestamp))+'):\n'+test_result+'\n')
        # mark this timestamp directory to denote a success training
        with open(model_save_dir+'Finish', 'w') as f:
            f.write('FINISH!!!')
        

    elif mode == 'test': # test mode
        print("=========== Inference Start ===========") 
        Engine.model = torch.load(f'{model_save_dir}{model_name}.pt').to(TrainSettings['device'])
        result, _ = Engine.evaluate(Sampler.Test_data, epoch_id=0)
        test_result = result
        log.info('test results( timestamp: '+str(int(timestamp))+'):\n'+test_result+'\n')

    else:
        raise NotImplementedError('unkonw mode: ', mode)