from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('project_path.ini')
sys_path = dict(cfg.items('PathSettings'))['sys_path']

import sys, os, argparse
sys.path.append(sys_path)

from Runs.ctr_task import Run
cfg = ConfigParser()


import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?', help='data name')
    parser.add_argument('--model_name', nargs='?', help='model name')
    parser.add_argument('--timestamp', default=None, nargs='?', help='timestamp')
    parser.add_argument('--mode', default='train', help='train or test')

    args = parser.parse_args()
    data = args.data_name
    model = args.model_name
    timestamp = args.timestamp
    mode = args.mode


    # ======= get the running setting ========
    config_path = sys_path+'Runs/configurations/'
    config_file = config_path + model+'.ini'
    if os.path.exists(config_file):
        cfg.read(config_file) # must set as absolute path
    else:
        raise FileNotFoundError(config_file +' Not Found !!!')

    print(model)
    # ======= run the main file ============
    
    Run(DataSettings   = dict(cfg.items("DataSettings")),
        ModelSettings  = dict(cfg.items("ModelSettings")),
        TrainSettings  = dict(cfg.items("TrainSettings")),
        ResultSettings = dict(cfg.items("ResultSettings")),
        mode=mode,
        timestamp=timestamp
        )

