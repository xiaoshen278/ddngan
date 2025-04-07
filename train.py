# -*- coding:utf-8 -*-
#@Project: NestFuse for image fusion
#@Author: Li Hui, Jiangnan University
#@Email: hui_li_jnu@163.com
#@File : train_autoencoder.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt 
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import Generator_Nest, Discriminator_i, Discriminator_v
from args_fusion import args
import pytorch_msssim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from torch.utils.tensorboard import SummaryWriter
# from logger import Logger
from loss import Fro_LOSS, L1_LOSS, discriminator_loss, generator_loss, fro_loss
import torch.cuda.amp as amp


seed = args.seed
random.seed(seed)
torch.manual_seed(seed)  # 1，11
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    original_ir_path = utils.list_images(args.dataset_ir)
    # print(len(original_ir_path))
    original_vis_path = utils.list_images(args.dataset_vis)
    # print(len(original_vis_path))
    
    zipped_data = list(zip(original_ir_path, original_vis_path))
    random.shuffle(zipped_data)
    train_num =100
    train_data = zipped_data[:train_num]
    # for i in range(2,3):
        # i = 3
    GAN_train(train_data)


def GAN_train(train_data):
    batch_size = args.batch_size
    idx = args.ssim_num
    # load network model
    # nest_model = FusionNet_gra()
    input_nc = args.input_channels
    if input_nc == 2:
        mode = 'L'
    else:
        mode = 'RGB'
        
    output_nc = args.output_channels
    # true for deeply supervision
    # In our paper, deeply supervision strategy was not used.
    deepsupervision = False
    nb_filter = args.nb_filter
    D_i = Discriminator_i()  # 可见光判别器
    D_v = Discriminator_v()  # 红外判别器
    G = Generator_Nest(nb_filter, input_nc, output_nc, deepsupervision)  # 生成器
    print('# discriminator_v parameters:', sum(param.numel() for param in D_v.parameters()))
    print('# discriminator_i parameters:', sum(param.numel() for param in D_i.parameters()))
    print('# generator parameters:', sum(param.numel() for param in G.parameters()))

    # 创建 SummaryWriter 对象，并指定日志文件保存路径
#     writer = SummaryWriter('./logs')
    pixel_loss_weight = args.pixel_loss_weight[0]
    grad_loss_weight = args.grad_loss_weight[0]
    if args.resume_G and args.resume_D_v and args.resume_D_i is not None:  # 从现有的查找点加载权重，默认None
        print('Resuming, initializing using weight from model.')
        D_v.load_state_dict(torch.load(args.resume_D_v))
        D_i.load_state_dict(torch.load(args.resume_D_i)) 
        G.load_state_dict(torch.load(args.resume_G))
    # print(G)
    optimizer_G = Adam(G.parameters(), args.lr_G)
    optimizer_D_v = Adam(D_v.parameters(), args.lr_D)
    optimizer_D_i = Adam(D_i.parameters(), args.lr_D)
    
    scheduler_G = CosineAnnealingWarmRestarts(optimizer_G, T_0=3, T_mult=2, last_epoch=-1)
    scheduler_D_v = CosineAnnealingWarmRestarts(optimizer_D_v, T_0=3, T_mult=2, last_epoch=-1)
    scheduler_D_i = CosineAnnealingWarmRestarts(optimizer_D_i, T_0=3, T_mult=2, last_epoch=-1)
    scaler_G = amp.GradScaler()
    scaler_D_v = amp.GradScaler()
    scaler_D_i = amp.GradScaler()
    
    criterionGAN = utils.GANLoss('vanilla').cuda()

    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        G.cuda()
        D_v.cuda()
        D_i.cuda()
        
    tbar = trange(1, args.epochs)
    print('Start training.....')
    Loss_dv = []
    Loss_di = []
    Loss_adv = []
    Loss_pixel = []
    Loss_ssim = []
    Loss_grad = []
    Loss_g_all = []
    D_v_losses = []
    D_i_losses = []
    G_v_losses = []
    G_i_losses = []
    G_losses = []

    # 消融研究指数
    con_lam = [1, 10, 50, 200, 400, 1000]  # 1000红外虚
    count_loss = 0
    all_ssim_loss = 0.
    all_pixel_loss = 0.
    all_grad_loss = 0.
    all_adv_loss = 0.
    all_D_v_loss = 0.
    all_D_i_loss = 0.
    # k = 3  # 判别器训练次数/batch
    # Set the logger
#     logger = Logger('./logs')
#     logger.writer.flush()
    
    for e in tbar:
        print('Epoch %d.....' % e)
        original_imgs_path_ir = [x[0] for x in train_data]
        original_imgs_path_vis = [x[1] for x in train_data]
        # load training database
        image_set_ir, batches = utils.load_dataset(original_imgs_path_ir, batch_size)
        image_set_vis, _ = utils.load_dataset(original_imgs_path_vis, batch_size)
        
        G.train()
        D_v.train()  
        D_i.train()
        
        count = 0
        for batch in range(batches):
            # train Discriminator
            for p in D_v.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netDv update
            for q in D_i.parameters():  # reset requires_grad
                q.requires_grad = True  # they are set to False below in netDi update
            for l in G.parameters():  # reset requires_grad
                l.requires_grad = False  # they are set to False below in netG update
            ir_paths = image_set_ir[batch * batch_size: (batch * batch_size + batch_size)]
            #print(mode)
            img_ir = utils.get_train_images_auto(ir_paths, height=args.HEIGHT, width=args.WIDTH, mode='L')# , flag=False,
            # print('len_img_ir:', img_ir.size())  # torch.Size([16, 1, 128, 128])
            vis_paths = image_set_vis[batch * batch_size: (batch * batch_size + batch_size)]
            img_vis = utils.get_train_images_auto(vis_paths, height=args.HEIGHT, width=args.WIDTH, mode='L') # , flag=False,
            count += 1
            
            real_ir = Variable(img_ir.data.clone(), requires_grad=True)
#             print(real_ir.shape)
            real_vis = Variable(img_vis.data.clone(), requires_grad=True)
#             print(real_ir.shape)
            if args.cuda:
                real_ir = real_ir.cuda()
                real_vis = real_vis.cuda()
            en_s = torch.cat([real_vis, real_ir], 1)
#             print(en_s.shape)
            fakes = G(en_s)
#             print(fakes.shape)
            D_v_loss_value = 0.
            D_i_loss_value = 0.
            
            # for j in range(k):
            # 判别损失
            optimizer_D_v.zero_grad()
            optimizer_D_i.zero_grad()
            # for fake in fakes:
            for i in range(batch_size):
                # 判别器判别
                D_v_real = D_v(real_vis[i])
                D_v_fake = D_v(fakes[i])
                D_i_real = D_i(real_ir[i])
                D_i_fake = D_i(fakes[i])

                D_v_loss_real = criterionGAN(D_v_real, True)
                D_v_loss_fake = criterionGAN(D_v_fake.detach(), False)
                D_v_loss = (D_v_loss_real + D_v_loss_fake)*2
 
                D_i_loss_real = criterionGAN(D_i_real, True)
                D_i_loss_fake = criterionGAN(D_i_fake.detach(), False)
                D_i_loss = (D_i_loss_real + D_i_loss_fake)*2

                D_v_loss_value += D_v_loss
                D_i_loss_value += D_i_loss

            D_v_loss_value /= batch_size
            D_i_loss_value /= batch_size

            scaler_D_v.scale(D_v_loss_value).backward(retain_graph=True)
            scaler_D_v.step(optimizer_D_v)
            scaler_D_v.update()
            scheduler_D_v.step()

            scaler_D_i.scale(D_i_loss_value).backward(retain_graph=True)
            scaler_D_i.step(optimizer_D_i)
            scaler_D_i.update()
            scheduler_D_i.step()


            all_D_v_loss += D_v_loss_value.item()
            all_D_i_loss += D_i_loss_value.item()
            
            # train Generator
            for p in D_v.parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netDv update
            for q in D_i.parameters():  # reset requires_grad
                q.requires_grad = False  # they are set to False below in netDi update
            for l in G.parameters():  # reset requires_grad
                l.requires_grad = True  # they are set to False below in netG update
                
            optimizer_G.zero_grad()
#             real_ir = Variable(img_ir.data.clone(), requires_grad=False).cuda()
#             real_vis = Variable(img_vis.data.clone(), requires_grad=False).cuda()
#             en_s_g = torch.cat([real_vis, real_ir], 1).cuda()     
            
            adv_loss_value = 0.
            # 计算生成损失-adv
            g_fakes = G(en_s)
            
            # real_vis = Variable(img_vis.data.clone(), requires_grad=True).cuda()
            for i in range(batch_size):
                G_v_loss = criterionGAN(D_v(g_fakes[i]), True)
                G_i_loss = criterionGAN(D_i(g_fakes[i]), True)

                G_loss = G_v_loss + G_i_loss
                adv_loss_value += G_loss
                
            adv_loss_value /= batch_size
            pixel_loss_value = 0.
            grad_loss_value = 0.
            # 计算内容损失-con
            for i in range(batch_size):
            # for fake in g_fakes:
                fake = g_fakes[i].cuda() 
                # 计算像素损失
                pixel_loss_temp_ir = Fro_LOSS(fake - real_ir[i])
                pixel_loss_temp_vis = Fro_LOSS(fake - real_vis[i])
                pixel_loss_value = pixel_loss_weight * pixel_loss_temp_ir + (1 - pixel_loss_weight) * pixel_loss_temp_vis
                
                # 计算梯度损失
                grad_loss_temp_ir = L1_LOSS(utils.grad(fake) - utils.grad(real_ir[i]))
                grad_loss_temp_vis = L1_LOSS(utils.grad(fake) - utils.grad(real_vis[i]))
                grad_loss_value = (1 - grad_loss_weight) * grad_loss_temp_ir + grad_loss_weight * grad_loss_temp_vis
                
            # ssim_loss_value /= batch_size
            pixel_loss_value /= batch_size
            grad_loss_value /= batch_size
            

            # con loss
            con_loss = pixel_loss_value + con_lam[3] * grad_loss_value #  + args.ssim_weight[idx] * ssim_loss_value   
            G_total_loss = adv_loss_value + con_loss

            scaler_G.scale(G_total_loss).mean().backward(retain_graph=True)
            scaler_G.step(optimizer_G)
            scaler_G.update()
            scheduler_G.step()
            
            # all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.mean().item()
            all_grad_loss += grad_loss_value.mean().item()
            all_adv_loss += adv_loss_value.item()
            
            if batch % (20*args.log_interval) == 0:
                mesg = "Epoch{}:[{}]Dv_loss:{:.6f}Di_loss:{:.6f}Adv loss:{:.6f}Pixel loss:{:.6f}Grad loss:{:.6f}G_total:{:.6f}".format(
                                e, 
                                count, 
                                all_D_v_loss / args.log_interval,
                                all_D_i_loss / args.log_interval,
                                all_adv_loss / args.log_interval,
                                all_pixel_loss / args.log_interval,
                                all_grad_loss / args.log_interval,
                                # (args.ssim_weight[idx] * all_ssim_loss) / args.log_interval,
                                (all_grad_loss + all_pixel_loss + all_adv_loss) / args.log_interval
                )  # {}\t SSIM weight {}\t  time.ctime(), i,Grad loss:{:.6f}, + all_grad_loss
                tbar.set_description(mesg)
                Loss_dv.append(all_D_v_loss / args.log_interval)
                Loss_di.append(all_D_i_loss / args.log_interval)
                Loss_adv.append(all_adv_loss / args.log_interval)
                Loss_pixel.append(all_pixel_loss / args.log_interval)
                # Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_grad.append(all_grad_loss / args.log_interval)
                
                Loss_g_all.append((all_pixel_loss + all_adv_loss+ all_grad_loss) / args.log_interval)
                # args.ssim_weight[idx] * all_ssim_loss +
                count_loss = count_loss + 1
                # all_ssim_loss = 0.
                all_pixel_loss = 0.
                all_grad_loss = 0.

            
            if (batch+1) % (1000 * args.log_interval) == 0:  # 
                # save model
                G.eval()
                G.cpu()
                save_model_filename_G = "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + "G"+ ".model"
                save_model_path_G = os.path.join(args.save_model_dir_autoencoder, save_model_filename_G)
                torch.save(G.state_dict(), save_model_path_G)
                save_model_filename_v = "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + "_"+ "D_v" + ".model"
                save_model_path_v = os.path.join(args.save_model_dir_autoencoder, save_model_filename_v)
                torch.save(D_v.state_dict(), save_model_path_v)
                save_model_filename_i = "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + "_"+ "D_i"  + ".model"
                save_model_path_i = os.path.join(args.save_model_dir_autoencoder, save_model_filename_i)
                torch.save(D_i.state_dict(), save_model_path_i)
                # save loss data
                # D v loss
                loss_filename_path = args.save_loss_dir + '/' + "loss_dv_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_"  + ".mat"
                scio.savemat(loss_filename_path, {'loss_dv': Loss_dv})
                # D i loss
                loss_filename_path = args.save_loss_dir  + '/' + "loss_di_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_"  + ".mat"
                scio.savemat(loss_filename_path, {'Loss_di': Loss_di})
                # Adv loss
                loss_filename_path = args.save_loss_dir + '/' + "loss_adv_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_"  + ".mat"
                scio.savemat(loss_filename_path, {'Loss_adv': Loss_adv})
                # pixel loss
                loss_data_pixel = Loss_pixel
                loss_filename_path = args.save_loss_dir + '/' + "loss_pixel_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_"  + ".mat"
                scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
                # SSIM loss
                loss_data_grad = Loss_grad # Loss_ssim
                loss_filename_path = args.save_loss_dir  + '/' + "loss_grad_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_"  + ".mat" # + args.ssim_path[idx]
                scio.savemat(loss_filename_path, {'loss_grad': loss_data_grad})
                # g all loss
                loss_data = Loss_g_all
                loss_filename_path = args.save_loss_dir + '/' + "Loss_g_all_epoch_" + str(e) + "_iters_" + \
                                     str(count) + "-" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".mat"
                scio.savemat(loss_filename_path, {'Loss_g_all': Loss_g_all})
                
                G.train()
                G.cuda()
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path_G)
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path_v)
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path_i)
                
            if batch % (20 * args.log_interval) == 0:  # 40 *             
                # for j in range(batch_size):
                j = 0
                # for fake in g_fakes:
                for j in range(batch_size):
                    # print(g_fakes[j].size())
                    # print(g_fakes[j])
                    utils.tensor_save_grayimage(g_fakes[j], './bev_fuse/generated-attn-res-images-{:03d}-{:04d}_{:02d}.png'.format(e, batch, j))
                    if e == 1:
                        utils.tensor_save_grayimage(real_ir[j], './bev_fuse/ir-images-{:04d}_{:02d}.png'.format(batch, j))
                        utils.tensor_save_grayimage(real_vis[j], './bev_fuse/vis-images-{:04d}_{:02d}.png'.format(batch, j))
                        j += 1
        G.eval()
        G.cpu()
        save_model_filename_D_v = "epoch_D_v" + str(args.epochs) + "_" + \
                                  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".model"
        save_model_filename_D_i = "epoch_D_i" + str(args.epochs) + "_" + \
                                  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".model"
        save_model_filename_G = "epoch_G" + str(args.epochs) + "_" + \
                                str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".model"

        torch.save(G.state_dict(), save_model_filename_G)
        torch.save(D_v.state_dict(), save_model_filename_D_v)
        torch.save(D_i.state_dict(), save_model_filename_D_i)
        G.train()  # 恢复为训练模式
        G.cuda()  # 如果使用 GPU，恢复到 GPU
    # D v loss
    loss_filename_path = args.save_loss_dir + '/' + "Final_loss_dv_epoch_" + str(
        args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".mat"
    scio.savemat(loss_filename_path, {'final_loss_dv': Loss_dv})
    # D i loss
    loss_filename_path = args.save_loss_dir + '/' + "Final_loss_di_epoch_" + str(
        args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".mat"
    scio.savemat(loss_filename_path, {'final_loss_di': Loss_di})
    # Adv loss
    loss_filename_path = args.save_loss_dir + '/' + "Final_loss_adv_epoch_" + str(
        args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".mat"
    scio.savemat(loss_filename_path, {'final_loss_adv': Loss_adv})
    # pixel loss
    loss_data_pixel = Loss_pixel
    loss_filename_path = args.save_loss_dir + '/' + "Final_loss_pixel_epoch_" + str(
        args.epochs) + "_" + str(
        time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".mat"
    scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
    # grad loss
    loss_data_grad = Loss_grad
    loss_filename_path = args.save_loss_dir + '/' + "Final_loss_grad_epoch_" + str(
        args.epochs) + "_" + str(
        time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".mat"
    scio.savemat(loss_filename_path, {'final_loss_grad': loss_data_grad})
    # all loss
    loss_data = Loss_g_all
    loss_filename_path = args.save_loss_dir + '/' + "Final_loss_all_epoch_" + str(
        args.epochs) + "_" + str(
        time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".mat"
    scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
    # save model
    # G.eval()
    # G.cpu()
    # save_model_filename_D_v = "Final_epoch_D_v" + str(args.epochs) + "_" + \
    #                       str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".model"
    # save_model_filename_D_i = "Final_epoch_D_i" + str(args.epochs) + "_" + \
    #                       str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".model"
    # save_model_filename_G = "Final_epoch_G" + str(args.epochs) + "_" + \
    #                       str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + ".model"
    #
    # torch.save(G.state_dict(), save_model_filename_G)
    # torch.save(D_v.state_dict(), save_model_filename_D_v)
    # torch.save(D_i.state_dict(), save_model_filename_D_i)
    

def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
