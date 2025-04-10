import torch
import torch.nn as nn
from args_fusion import args
from torch import autograd
import numpy as np
# import tflib as lib
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
"""
from torch.autograd import Variable 
# import torch_npu

# device = args.device
# torch_npu.npu.set_device(device)

class HingeLoss(nn.Module):
    def __init__(self, margin=1.0, size_average=True, sign=1.0):
        super(HingeLoss, self).__init__()
        self.sign = sign
        self.margin = margin
        self.size_average = size_average
 
    def forward(self, input, target):
        #
        input = input.view(-1)

        #
        assert input.dim() == target.dim()
        for i in range(input.dim()): 
            assert input.size(i) == target.size(i)

        #
        output = self.margin - torch.mul(target, input)

        # 
        mask = torch.FloatTensor(input.size()).zero_().to('cuda')
        """
        if 'cuda' in input.data.type():
            # mask = torch.cuda.FloatTensor(input.size()).zero_()
        else:
            mask = torch.FloatTensor(input.size()).zero_()
        """
        mask = Variable(mask)
        mask[torch.gt(output, 0.0)] = 1.0

        #
        output = torch.mul(output, mask)

        # size average
        if self.size_average:
            output = torch.mul(output, 1.0 / input.nelement())

        # sum
        output = output.sum()

        # apply sign
        output = torch.mul(output, self.sign)
        return output

def L1_LOSS(batchimg):
    # batchimg = batchimg.unsqueeze(0)
    # print(batchimg.size())
    batch_size, num_channels, height, width = batchimg.size()
    L1_norm = torch.sum(torch.abs(batchimg), dim=[2, 3])
    E = torch.mean(L1_norm) / (height * width)
    return E

"""
def Fro_LOSS(batchimg):
	fro_norm = torch.norm(batchimg, p='fro', dim=[1,2])**2
	E = torch.mean(fro_norm)
	return E
"""
def Fro_LOSS(batchimg):
    batchimg = batchimg.unsqueeze(0)
    # print(batchimg.size())
    batch_size, num_channels, height, width = batchimg.size()
    fro_norm = torch.norm(batchimg.view(batch_size, num_channels, height*width), p='fro', dim=2)**2
    E = torch.mean(fro_norm, dim=0) / (height * width)
    return E

def fro_loss(batchimg):
    # batchimg = batchimg.unsqueeze(0)
    # print(batchimg.size())
    batch_size, num_channels, height, width = batchimg.size()
    fro_norm = torch.norm(batchimg.view(batch_size, num_channels, height*width), p='fro', dim=2)**2
    E = torch.mean(fro_norm, dim=0) / (height * width)
    return E
"""
def ssim_loss(x, y):
    # Compute SSIM loss
    ssim_loss = 1 - torch.mean(pytorch_msssim.ssim(x, y, max_val=x.max()))

    return ssim_loss
"""
# 定义判别器的Wasserstein Loss
def discriminator_loss(real_output, fake_output):
    return torch.mean(fake_output)-torch.mean(real_output)

# 定义生成器的Wasserstein Loss
def generator_loss(fake_output):
    return -torch.mean(fake_output)

def loss_fused_ir(visible, infrared, fused, xi):
    h, w = visible.shape[-2], visible.shape[-1]
    print(h, w)
    # Compute gradient of fused and visible images
    grad_fused = torch.autograd.grad(fused, (fused,), grad_outputs=torch.ones_like(fused), create_graph=True)[0]
    grad_visible = torch.autograd.grad(visible, (visible,), grad_outputs=torch.ones_like(visible), create_graph=True)[0]
    
    # Compute pixel-wise L2 loss and regularization term
    pixel_loss = nn.MSELoss(reduction='mean')(fused, infrared)
    print(pixel_loss)
    reg_loss = xi * nn.MSELoss(reduction='mean')(grad_fused, grad_visible)
    print(reg_loss)
    # Combine losses with weighting factor
    loss = 1 / (h * w) * (pixel_loss + reg_loss)
    
    return loss


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(args.batch_size, real_data.nelement()/args.batch_size).contiguous().view(args.batch_size, 1, 120, 120)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if args.cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.LAMBDA
    return gradient_penalty
"""
# For calculating inception score
def get_inception_score(G, ):
    all_samples = []
    for i in xrange(10):
        samples_100 = torch.randn(100, 128)
        if use_cuda:
            samples_100 = samples_100.cuda(gpu)
        samples_100 = autograd.Variable(samples_100, volatile=True)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))
"""


def get_inception_score(G, dataset_ir, dataset_vis, batch_size=args.batch_size):
    dataloader_ir = torch.utils.data.DataLoader(dataset_ir, batch_size=batch_size, shuffle=False)
    dataloader_vis = torch.utils.data.DataLoader(dataset_vis, batch_size=batch_size, shuffle=False)
    all_samples = []
    for batch1, batch2 in zip(dataloader_ir, dataloader_vis):
        if args.cuda:
            batch1 = batch1.cuda()
            batch2 = batch2.cuda()
        real_ir = autograd.Variable(batch1, volatile=True)
        real_vis = autograd.Variable(batch2, volatile=True)
        en_r = G.encoder(real_ir)
        en_v = G.encoder(real_vis)
        # print(en_r)
        # 融合两种图像
        f_type = 'attention_max'
        f = G.fusion(en_r, en_v, f_type)
        # print(f)
        # fake_images = G.decoder_eval(f)
        fake_images = G.decoder_train(f)
        generated_images = Variable(fake_images, requires_grad=True)
        # generated_images = G(real_ir)
        generated_images = (generated_images + 1.0) / 2.0 # Normalize generated images to [0, 1]
        # generated_images = torch.nn.functional.interpolate(generated_images, size=(120, 120), mode='bilinear', align_corners=False) # Upsample to 120x120
        all_samples.append(generated_images.cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(all_samples, 255).astype('int32')
    all_samples = all_samples.transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))
