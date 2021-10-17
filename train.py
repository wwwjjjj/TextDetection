from code.dataset import TotalText
from code.tools.augmentation import BaseTransform,Augmentation
from config.Totaltext import config as cfg
from code.models.Hourglass import get_large_hourglass_net
from code.models.msra_resnet import get_pose_net

#from code.textnet import TextNet
from code.tools.Progbar import Progbar
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset,DataLoader
from code.loss import CtdetLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil

import os
import copy
import numpy as np
import argparse
#from pyecharts.charts import HeatMap
import cv2
from code.tools.utils import draw_umich_gaussian,gaussian_radius
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.autograd.set_detect_anomaly(True)
def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y


def display(id,image,gt_map,pred_map):
    plt.imshow(image.cpu().detach().numpy()[0][0])
    plt.title("{}".format(id))
    plt.show()

    plt.imshow((pred_map['hm']).cpu().detach().numpy()[0][0])
    plt.show()

#    plt.imshow((gt_map['dense_wh']).cpu().detach().numpy()[0][0])
  #  plt.show()

    plt.imshow((pred_map['hm_b']).cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((pred_map['hm_l']).cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((pred_map['hm_r']).cpu().detach().numpy()[0][0])
    plt.show()
    '''
    plt.imshow(gt_map['hm'].cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((pred_map['hm_b']).cpu().detach().numpy()[0][0])
    plt.show()'''



    plt.imshow((gt_map['hm']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()

    '''plt.imshow((gt_map['hm_t']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((gt_map['hm_b']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((gt_map['hm_l']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()

    plt.imshow((gt_map['hm_r']).cpu().detach().numpy()[0][0] * gt_map['trm'].cpu().detach().numpy()[0][0])
    plt.show()
'''




if __name__=="__main__":
    # ---------------- ARGS AND CONFIGS ----------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelfile', type=str, default="/home/pei_group/jupyter/Wujingjing/Text_Detection/save_models/Hourglass_epoch55_71.7055.pt")
    parser.add_argument('--data_root', type=str, default="/home/pei_group/jupyter/Wujingjing/data/totaltext/")
    opt = parser.parse_args()
    print("--- TRAINING ARGS ---")
    print(opt)
    # 其他的配置信息都在config/Totaltext.py中

    data_train = TotalText(
        data_root=opt.data_root,#'D:/python_Project/data/totaltext',#'/home/pei_group/jupyter/Wujingjing/data/totaltext/',
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
        data_root=opt.data_root,#'/home/pei_group/jupyter/Wujingjing/data/totaltext/',#'D:/python_Project/data/totaltext',#'/home/pei_group/jupyter/Wujingjing/data/totaltext/',
        ignore_list=None,
        is_training=False,
        transform=BaseTransform(size=512, mean=cfg.means,
                                    std=cfg.stds),
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
        modelfile=opt.modelfile#"/home/pei_group/jupyter/Wujingjing/Text_Detection/save_models/Hourglass_epoch1_106.6700.pt"
        start_epoch=cfg.start_epoch
        checkpoint=None
        if modelfile!=None:
            start_epoch=modelfile.split('/')[7].split('_')[1][5:]#7
            checkpoint = torch.load(modelfile)

        print("start training at epoch {}".format(start_epoch))

        model = get_large_hourglass_net(18, cfg.heads, 64,checkpoint).cuda()
        criterion = CtdetLoss(cfg)
        print(model)
    elif cfg.model == 'ResNet':
        model = get_pose_net(101, cfg.heads, 64).cuda()
        criterion = CtdetLoss(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)
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

            gt_map = {'hm': ret['hm'].cuda( ),#'tr_m':ret['tr_m'].cuda(),
                      'hm_t': ret['hm_t'].cuda( ), 'hm_b': ret['hm_b'].cuda( ),
                      'hm_l': ret['hm_l'].cuda( ), 'hm_r': ret['hm_r'].cuda( ),
                      'trm':ret['trm'].cuda( ),
                      'dense_wh':ret['dense_wh'].cuda( ),'dense_wh_mask':ret['dense_wh_mask'].cuda( )
                      ,'center_points':ret['center_points'].cuda( ),
                      'off_mask':ret['off_mask'].cuda( ),'offsets':ret['offsets'].cuda( )}
            # print(gt_map['hm_t'].dtype,gt_map['hm_t'].shape)

            output = model(image)

            loss, loss_stas, pred_map = criterion(output, gt_map)
            if i==0:
                display('epoch {}'.format(e),image,gt_map,pred_map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss.append(loss.item())

            if i % 5 == 0 :
                writer.add_scalars(cfg.model,
                                   {"loss": loss.item(),
                                    "hm_loss": loss_stas['hm_loss'],
                                    "geo_loss": loss_stas['geo_loss'],

                                    "wh_loss": loss_stas['wh_loss'],
                                    "off_loss": loss_stas['off_loss']
                                    }, i)


                progbar_train.add(min(5,156-i), values=[("epoch", e+int(start_epoch)),
                                             ("loss", loss.item()),
                                             ("hm_loss", loss_stas['hm_loss'].item()),
                                             ("geo_loss", loss_stas['geo_loss'].item()),
                                             ("wh_loss",loss_stas['wh_loss'].item()),
                                             ("off_loss", loss_stas['off_loss'].item())
                                             ])

        print('EPOCH<', e+int(start_epoch), '>: train loss:', running_loss / i if i>0 else running_loss
              )
        # ---------------- VALIDATION ----------------

        model.eval()
        progbar_val = Progbar(
            len(valid_loader), stateful_metrics=["epoch", "config", "lr"]
        )
        valid_loss = 0.0


        nums = 0
        i=0
        for i, ret in enumerate(valid_loader):

            image = ret['input'].cuda( )
            gt_map = {'hm': ret['hm'].cuda( ),  # 'tr_m':ret['tr_m'].cuda(),
                      'hm_t': ret['hm_t'].cuda( ), 'hm_b': ret['hm_b'].cuda( ),
                      'hm_l': ret['hm_l'].cuda( ), 'hm_r': ret['hm_r'].cuda( ),
                      'trm': ret['trm'].cuda( ),
                      'dense_wh': ret['dense_wh'].cuda( ), 'dense_wh_mask': ret['dense_wh_mask'].cuda( )
                , 'center_points': ret['center_points'].cuda( ),
                      'off_mask': ret['off_mask'].cuda( ), 'offsets': ret['offsets'].cuda( )}
            # print(gt_map['hm_t'].dtype,gt_map['hm_t'].shape)

            output = model(image)

            loss, loss_stas, pred_map = criterion(output, gt_map)


            valid_loss += loss.item()

            if i % 5 == 0:
                writer.add_scalars(cfg.model,
                                   {"val_loss": loss.item(),
                                    "val_hm_loss": loss_stas['hm_loss'],
                                    "val_geo_loss": loss_stas['geo_loss'],
                                    "val_wh_loss": loss_stas['wh_loss'],
                                    "val_off_loss": loss_stas['off_loss']
                                    }, i)

            progbar_val.add(1, values=[("epoch", e + int(start_epoch)),
                                                           ("val_loss", loss.item()),
                                                           ("val_hm_loss", loss_stas['hm_loss'].item()),
                                                           ("val_wh_loss", loss_stas['wh_loss'].item()),
                                                           ("val_geo_loss", loss_stas['geo_loss'].item()),
                                                           ("val_off_loss", loss_stas['off_loss'].item())
                                                           ])
        scheduler.step()
        print('EPOCH<', e+int(start_epoch), '>: valid loss:', valid_loss / i)
        torch.save(model.state_dict(), os.path.join(cfg.model_savedir, cfg.model+"_epoch%d_%.4f.pt" % (e+int(start_epoch)+1, valid_loss)))
        print("epoch {} save done!".format(e+int(start_epoch)))
        # best models
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(cfg.model_savedir,"{}_best.pt".format( cfg.model)))








