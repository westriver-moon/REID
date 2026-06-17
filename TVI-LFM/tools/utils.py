import os
import time
from easydict import EasyDict as edict
import yaml

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

def time_now():
    return time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))

def save_train_configs(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/configs.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

def load_train_configs(path):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isfile(path):
        repo_candidate = os.path.join(repo_root, path)
        if os.path.isfile(repo_candidate):
            path = repo_candidate
    default_path = os.path.join(repo_root, 'config', 'default.yaml')
    with open(default_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    with open(path, 'r') as f:
        selected_args = yaml.load(f, Loader=yaml.FullLoader)
    if selected_args:
        args.update(selected_args)
    pmt_pretrained = args.get('pmt_pretrained')
    if pmt_pretrained and not os.path.isabs(pmt_pretrained):
        candidates = [
            os.path.abspath(pmt_pretrained),
            os.path.abspath(os.path.join(os.path.dirname(path), pmt_pretrained)),
            os.path.abspath(os.path.join(repo_root, pmt_pretrained)),
            os.path.abspath(os.path.join(repo_root, '..', pmt_pretrained)),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                args['pmt_pretrained'] = candidate
                break
    args['config_select'] = path
    return edict(args)

