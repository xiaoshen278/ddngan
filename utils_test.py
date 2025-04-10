import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
# from torch.utils.serialization import load_lua
from args_fusion import args
# from scipy.misc import imread, imsave, imresize
from imageio import imread, imsave
from skimage.transform import resize as imresize
import matplotlib as mpl
import cv2
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def gray_to_rgb(image):
    image = image.squeeze(0)
    # Assuming image is a PyTorch tensor with size [1, H, W]
    assert len(image.shape) == 3 and image.shape[0] == 1, "The input image should be a grayscale image with size [1, H, W]."

    # Convert the image tensor to a numpy array and squeeze out the channel dimension
    image_np = image.squeeze(0).numpy()

    # Apply the colormap
    image_rgb_np = plt.get_cmap('jet')(image_np)

    # Convert the image back to a tensor
    # Also, discard the alpha channel if it exists
    image_rgb = torch.from_numpy(image_rgb_np[..., :3].copy()).permute(2, 0, 1)
    
    return image_rgb

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches

def get_image(path, height=None, width=None, flag=False):
    if flag is True:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return image
"""
def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, pilmode=mode)
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')

    if height is not None and width is not None:
        image = imresize(image, [height, width], order=0)
    return image
"""

def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


# load images - test phase
def get_test_images(paths, height=None, width=None, flag=False):
    print(paths)
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        # image = imread(path, pilmode='L')
        if flag == False:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
        if height is not None and width is not None:
            image = imresize(image, [height, width], order=0)
        print(image.shape)
        h = image.shape[0]
        w = image.shape[1]
        image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).float()

    # images = np.stack(images, axis=0)
    # images = torch.from_numpy(images).float()
    return images, h, w  # , c

def get_img_parts(image, h, w):
    images = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    img1 = image[0:h_cen + 3, 0: w_cen + 3]
    img1 = np.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:h_cen + 3, w_cen - 2: w]
    img2 = np.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[h_cen - 2:h, 0: w_cen + 3]
    img3 = np.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[h_cen - 2:h, w_cen - 2: w]
    img4 = np.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images

# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)

def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    ones_temp = torch.ones(1, 1, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        # save_image_test(img1, './outputs/test/block1.png')
        # save_image_test(img2, './outputs/test/block2.png')
        # save_image_test(img3, './outputs/test/block3.png')
        # save_image_test(img4, './outputs/test/block4.png')

        img_f = torch.zeros(1, 1, h, w).cuda()
        count = torch.zeros(1, 1, h, w).cuda()

        img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list

def save_images(img_fusion, output_path):
    img_fusion = img_fusion.float()
    print(img_fusion.size())
    
    if args.cuda:
        # print(img_fusion.data.size())
        # print('img_fusion.data.numpy():', img_fusion.data[0].size())
        
        img_fusion = img_fusion.cpu().data.numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.squeeze(0).transpose(1, 2, 0).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    # 	img_fusion = imresize(img_fusion, [h, w])
    imsave(output_path, img_fusion)
"""
def save_images(img_fusion, output_path):
    img_fusion = img_fusion.float()
    print(img_fusion.size())
    
    if args.cuda:
        img_fusion = img_fusion.cpu().data.numpy()
    else:
        img_fusion = img_fusion.data.numpy()

    # normalize to [0,1]
    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    
    # Convert the image back to a PyTorch tensor
    img_fusion = torch.from_numpy(img_fusion)

    # convert to RGB image
    img_fusion_rgb = gray_to_rgb(img_fusion)

    # convert tensor to numpy array
    img_fusion_rgb = img_fusion_rgb.permute(1, 2, 0).numpy()  # permute dimensions for color image

    # Scale to [0,255] for uint8 format
    img_fusion_rgb = (img_fusion_rgb * 255).astype('uint8')

    # save image
    imsave(output_path, img_fusion_rgb)
"""
def save_image(img_fusion, output_path):
    img_fusion = img_fusion.float()
    print(img_fusion.size())
    
    if args.cuda:
        # print(img_fusion.data.size())
        # print('img_fusion.data.numpy():', img_fusion.data[0].size())
        
        img_fusion = img_fusion.cpu().data.numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    # 	img_fusion = imresize(img_fusion, [h, w])
    imsave(output_path, img_fusion)

