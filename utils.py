import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
# from torch.utils.serialization import load_lua
import torchfile
from args_fusion import args
from scipy.misc import imread, imsave, imresize
import matplotlib as mpl
import cv2
from torchvision import datasets, transforms


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


def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, mode=mode)
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image


def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    image_masks = []
    image_ens = []
    image_bs = []
    image_ds = []
    
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        # image_mask = cv2.Canny(image,100,120)
        # image_mask = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        # image_en = image.astype(np.float) + image_mask.astype(np.float)
        # image_en = ((image_en-image_en.min())/(image_en.max()-image_en.min())) * 255
        # image_en = image_en.astype(np.uint8)

        image_mask = get_image('E:/train2014_mask/mask' + path[13:], height, width, mode=mode)
        image_en = get_image('E:/train2014_en/en' + path[13:], height, width, mode=mode)
        image_b = get_image('E:/train2014_b/b' + path[13:], height, width, mode=mode)
        image_d = get_image('E:/train2014_d/d' + path[13:], height, width, mode=mode)

        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
            image_mask = np.reshape(image_mask, [1, 256, 256])
            image_en = np.reshape(image_en, [1, 256, 256])
            image_b = np.reshape(image_b, [1, 256, 256])
            image_d = np.reshape(image_d, [1, 256, 256])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])

        images.append(image)
        image_masks.append(image_mask)
        image_ens.append(image_en)
        image_bs.append(image_b)
        image_ds.append(image_d)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    image_masks = np.stack(image_masks, axis=0)
    image_masks = torch.from_numpy(image_masks).float()
    image_ens = np.stack(image_ens, axis=0)
    image_ens = torch.from_numpy(image_ens).float()
    image_bs = np.stack(image_bs, axis=0)
    image_bs = torch.from_numpy(image_bs).float()
    image_ds = np.stack(image_ds, axis=0)
    image_ds = torch.from_numpy(image_ds).float()
    return [images, image_masks, image_ens, image_bs, image_ds]


def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    image_masks = []
    image_bs = []
    image_ds = []
    
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        image_mask = get_image('E:/xubenfuse/images/IV_images_mask/mask' + path[17:], height, width, mode=mode)
        image_b = get_image('E:/xubenfuse/images/IV_images_b/b' + path[17:], height, width, mode=mode)
        image_d = get_image('E:/xubenfuse/images/IV_images_d/d' + path[17:], height, width, mode=mode)


        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
            image_mask = np.reshape(image_mask, [1, image_mask.shape[0], image_mask.shape[1]])
            image_b = np.reshape(image_b, [1, image_b.shape[0], image_b.shape[1]])
            image_d = np.reshape(image_d, [1, image_d.shape[0], image_d.shape[1]])
        else:
            # test = ImageToTensor(image).numpy()
            # shape = ImageToTensor(image).size()
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    image_masks.append(image_mask)
    image_bs.append(image_b)
    image_ds.append(image_d)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    image_masks = np.stack(image_masks, axis=0)
    image_masks = torch.from_numpy(image_masks).float()
    image_bs = np.stack(image_bs, axis=0)
    image_bs = torch.from_numpy(image_bs).float()
    image_ds = np.stack(image_ds, axis=0)
    image_ds = torch.from_numpy(image_ds).float()
    return [images, image_masks, image_bs, image_ds]


# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)


def save_images(path, data):
    # if isinstance(paths, str):
    #     paths = [paths]
    #
    # t1 = len(paths)
    # t2 = len(datas)
    # assert (len(paths) == len(datas))

    # if prefix is None:
    #     prefix = ''
    # if suffix is None:
    #     suffix = ''

    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    imsave(path, data)

    # for i, path in enumerate(paths):
    #     data = datas[i]
    #     # print('data ==>>\n', data)
    #     if data.shape[2] == 1:
    #         data = data.reshape([data.shape[0], data.shape[1]])
    #     # print('data reshape==>>\n', data)
    #
    #     name, ext = splitext(path)
    #     name = name.split(sep)[-1]
    #
    #     path = join(save_path, prefix + suffix + ext)
    #     print('data path==>>', path)
    #
    #     # new_im = Image.fromarray(data)
    #     # new_im.show()
    #
    #     imsave(path, data)


