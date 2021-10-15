from easydict import EasyDict
import torch

config = EasyDict()

# ----------basic cfg----------
config.num_workers = 4
config.model_savedir='/home/pei_group/jupyter/Wujingjing/Text_Detection/save_models/'
config.write_dir='/home/pei_group/jupyter/Wujingjing/Text_Detection/exp/'
# model
config.model='Hourglass'#'Hourglass''ResNet'#'TextNet'#'Hourglass'


config.heads = {'hm_t': 1, 'hm_l': 1,
                'hm_b': 1, 'hm_r': 1,
                'hm': 1,'dense_wh':4,
                'offsets':2}
# batch_size
config.batch_size = 8
# weight
config.hm_weight=1
config.wh_weight=1
config.geo_weight=10
config.off_weight=1


# training epoch number
config.max_epoch = 200
config.start_epoch = 0

# learning rate
config.lr = 3e-4

# ----------about data----------
config.n_disk = 15

config.output_dir = 'output'

config.input_size = 512

config.downsample=4

# max polygon per image
config.max_annotation = 128

# max point per polygon
config.max_points = 20

# use hard examples (annotated as '#')
config.use_hard = True

# demo tr threshold
config.tr_thresh = 0.6

config.post_process_merge = False
config.means=(0.485, 0.456, 0.406)
config.stds=(0.229, 0.224, 0.225)
# ----------post processing----------
config.max_diffusion=10
config.K=9
def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
