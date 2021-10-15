from code.dataset import TotalText
from code.tools.augmentation import BaseTransform,Augmentation
from config.Totaltext import config as cfg
from code.models.Hourglass import get_large_hourglass_net
from code.models.msra_resnet import get_pose_net
from code.U_resNet import U_resNet
from code.textnet import TextNet
from code.tools.Progbar import Progbar
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset,DataLoader
from code.loss import TextLoss,CtdetLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil
import os
import copy
import numpy as np
#from pyecharts.charts import HeatMap
import cv2
from code.tools.utils import draw_umich_gaussian,gaussian_radius
torch.autograd.set_detect_anomaly(True)
def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y


def display(id,image,gt_map):
    plt.imshow(image.cpu().detach().numpy()[0][0])
    plt.title("{}".format(id))
    plt.show()


    plt.imshow((gt_map['dense_wh']).cpu().detach().numpy()[0][0]*gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()



    plt.imshow((gt_map['tr']).cpu().detach().numpy()[0][0]*gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((gt_map['hm']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()






if __name__=="__main__":
    data_train = TotalText(
        data_root='/home/pei_group/jupyter/Wujingjing/data/totaltext/',
        ignore_list=None,
        is_training=True,
        transform=Augmentation(size=512, mean=cfg.means, std=cfg.stds),
        map_transform=BaseTransform(size=cfg.input_size // cfg.downsample, mean=cfg.means,
                                    std=cfg.stds) if cfg.downsample > 0 else None,
        map_size=cfg.input_size if cfg.downsample == 0 else cfg.input_size // cfg.downsample
    )

    train_loader = DataLoader(
        data_train,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True)
    data_valid = TotalText(
        data_root='/home/pei_group/jupyter/Wujingjing/data/totaltext/',
        ignore_list=None,
        is_training=False,
        transform=Augmentation(size=512, mean=cfg.means, std=cfg.stds),
        map_transform=BaseTransform(size=cfg.input_size // cfg.downsample, mean=cfg.means,
                                    std=cfg.stds) if cfg.downsample > 0 else None,
        map_size=cfg.input_size if cfg.downsample == 0 else cfg.input_size // cfg.downsample
    )

    valid_loader = DataLoader(
        data_valid,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=True)

    if cfg.model == 'Hourglass':
        #modelfile="/home/pei_group/jupyter/Wujingjing/Text_Detection/save_models/Hourglass_epoch71_168.2355.pt"
        #start_epoch=modelfile.split('/')[7].split('_')[1][5:]

        #print("start training at epoch {}".format(start_epoch))
        #checkpoint=torch.load(modelfile)
        model = get_large_hourglass_net(18, cfg.heads, 64,None).cuda()
        criterion = CtdetLoss(cfg)
    elif cfg.model == 'ResNet':
        model = get_pose_net(101, cfg.heads, 64).cuda()
        criterion = CtdetLoss(cfg)
    else:
        model = TextNet(cfg.heads, output_channel=4).cuda()
        model.init_weight()
        criterion = TextLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)
    writer_dir = os.path.join(cfg.write_dir, cfg.model)
    if os.path.exists(writer_dir):
        shutil.rmtree(writer_dir, ignore_errors=True)
    writer = SummaryWriter(logdir=writer_dir)

    best_loss=10000
    for e in range(cfg.max_epoch):
        model.train()
        progbar_train = Progbar(
            len(train_loader), stateful_metrics=["epoch", "config", "lr"]
        )
        running_loss = 0.0

        k = 0
        train_loss = []
        i=0
        for i, ret in enumerate(train_loader):
            image = ret['input'].cuda( )
            gt_map = {'hm': ret['hm'].cuda( ),  # 'tr_m':ret['tr_m'].cuda(),
                      'hm_t': ret['hm_t'].cuda( ), 'hm_b': ret['hm_b'].cuda( ),
                      'hm_l': ret['hm_l'].cuda( ), 'hm_r': ret['hm_r'].cuda( ),
                      'trm': ret['trm'].cuda( ),
                      'dense_wh': ret['dense_wh'].cuda( ), 'dense_wh_mask': ret['dense_wh_mask'].cuda( )
                , 'center_points': ret['center_points'].cuda( ),
                      'off_mask': ret['off_mask'].cuda( ), 'offsets': ret['offsets'].cuda( )}

            display('100', image, gt_map)








