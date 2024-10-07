import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision.models import vgg16_bn

import os


#################################################################################################
def compute_measure(y_gt, y_pred, data_range):
    pred_psnr = compute_PSNR(y_pred, y_gt, data_range)
    pred_ssim = compute_SSIM(y_pred, y_gt, data_range)
    pred_rmse = compute_RMSE(y_pred, y_gt)
    return (pred_psnr, pred_ssim, pred_rmse)


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # img1 = torch.unsqueeze(img1, dim=0)
    # img2 = torch.unsqueeze(img2, dim=0)
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(img1.size()) == 2:
        shape_ = img1.size()
        img1 = img1.view(1, 1, shape_[0], shape_[1])
        img2 = img2.view(1, 1, shape_[0], shape_[1])
    window = create_window(window_size, channel)
    window = window.type_as(img1).to(device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, padding=window_size // 2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2

    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    # C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


#################################################################################################
# loss
def tv_loss_beta1(img):  # β=1,开根号的TV loss
    loss = torch.mean(torch.sqrt(torch.pow(img[:, :, :-1, :-1] - img[:, :, :-1, 1:], 2) +
                                 torch.pow(img[:, :, :-1, :-1] - img[:, :, 1:, :-1], 2)))
    # print('loss_tmp%0.6f' % loss_tmp)
    # print('%0.6f' % torch.sum(torch.sqrt(torch.pow(img[:, :, :-1, :-1] - img[:, :, :-1, 1:], 2) +
    #                                      torch.pow(img[:, :, :-1, 1:] - img[:, :, 1:, 1:], 2))))
    return loss


#################################################################################################
def tv_loss_beta2(img):  # β=2
    w_variance_mean = torch.mean(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    # print('torch_mean=%.8f'%count_mean)
    # print('w_variance=',w_variance)
    h_variance_mean = torch.mean(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = h_variance_mean + w_variance_mean
    return loss


#########################################################################################
def tv_loss_anisotropic(img):  # 各向异性
    w_variance_mean = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    h_variance_mean = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    loss = h_variance_mean + w_variance_mean
    return loss


#########################################################################################
def tv4_loss(img):  # TV4,加上斜对角TV loss
    w_variance_mean = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    h_variance_mean = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    diagonal_var_mean = torch.mean(torch.abs(img[:, :, :-1, :-1] - img[:, :, 1:, 1:]))
    back_diagonal_mean = torch.mean(torch.abs(img[:, :, :-1, 1:] - img[:, :, 1:, :-1]))
    loss = h_variance_mean + w_variance_mean + diagonal_var_mean + back_diagonal_mean
    return loss


#########################################################################################
def wc_roi_RMSE(phantom_gt, prediction_phantoms):
    # get the number of prediction_phantoms and number of pixels in x and y
    nim, ch, nx, ny = prediction_phantoms.shape
    prediction_phantoms = prediction_phantoms.reshape(nx, ny)
    phantom_gt = phantom_gt.reshape(nx, ny)
    prediction_phantoms = prediction_phantoms.data.cpu().numpy()
    phantom_gt = phantom_gt.data.cpu().numpy()

    # mean RMSE computation
    diffsquared = (phantom_gt - prediction_phantoms) ** 2
    num_pix = float(nx * ny)

    meanrmse = np.sqrt(((diffsquared / num_pix).sum(axis=0)).sum(axis=0)).mean()
    # print("The mean RSME over %3i images is %8.6f " % (nim, meanrmse))
    # worst-case ROI RMSE computation
    roisize = 25  # width and height of test ROI in pixels
    x0 = 0  # tracks x-coordinate for the worst-case ROI
    y0 = 0  # tracks x-coordinate for the worst-case ROI
    im0 = 0  # tracks image index for the worst-case ROI

    maxerr = -1.
    for i in range(nim):  # For each image
        # print("Searching image %3i" % (i))
        phantom = phantom_gt.copy()  # GT
        prediction = prediction_phantoms.copy()  # Pred
        # These for loops cross every pixel in image (from region of interest)
        for ix in range(nx - roisize):
            for iy in range(ny - roisize):
                roiGT = phantom[ix:(ix + roisize), iy:(iy + roisize)].copy()  # GT
                roiPred = prediction[ix:(ix + roisize), iy:(iy + roisize)].copy()  # Pred
                if roiGT.max() > 0.01:  # Don't search ROIs in regions where the truth image is zero
                    roirmse = np.sqrt((((roiGT - roiPred) ** 2) / float(roisize ** 2)).sum())
                    if roirmse > maxerr:
                        maxerr = roirmse
                        x0 = ix
                        y0 = iy
                        im0 = i

    # print("Worst-case ROI RMSE is %8.6f" % (maxerr))
    # print("Worst-case ROI location is (%3i,%3i) in image number %3i " % (x0, y0, im0 + 1))
    return torch.from_numpy(np.array(meanrmse)), torch.from_numpy(np.array(maxerr))


#################################################################################################
# Perceptual Loss 图像风格迁移
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


class PerceptualLoss:
    def __init__(self, args):
        self.content_layer = args.content_layer
        device = get_device(args)
        self.vgg = nn.DataParallel(Vgg16())
        self.vgg.eval()
        self.mse = nn.DataParallel(nn.MSELoss())
        self.mse_sum = nn.DataParallel(nn.MSELoss(reduction='sum'))
        style_image = Image.open(args.style_image).convert('RGB')
        _, transform = get_transform(args)
        style_image = transform(style_image).repeat(args.batch_size, 1, 1, 1).to(device)

        with torch.no_grad():
            self.style_features = self.vgg(style_image)
            self.style_gram = [gram(fmap) for fmap in self.style_features]
        pass

    def __call__(self, x, y_hat):
        b, c, h, w = x.shape
        y_content_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        recon = y_content_features[self.content_layer]
        recon_hat = y_hat_features[self.content_layer]
        L_content = self.mse(recon_hat, recon)

        y_hat_gram = [gram(fmap) for fmap in y_hat_features]
        L_style = 0
        for j in range(len(y_content_features)):
            _, c_l, h_l, w_l = y_hat_features[j].shape
            L_style += self.mse_sum(y_hat_gram[j], self.style_gram[j]) / float(c_l * h_l * w_l)

        L_pixel = self.mse(y_hat, x)

        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        L_tv = (diff_i + diff_j) / float(c * h * w)

        return L_content, L_style, L_pixel, L_tv


#################################################################################################
from torchvision.models import vgg16_bn


class FeatureLoss(nn.Module):
    def __init__(self, blocks, device):
        super().__init__()

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.to(device)

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # inputs = F.normalize(inputs, mean, std)
        # targets = F.normalize(targets, mean, std)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]
        input_gram = [gram(fmap) for fmap in input_features]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]
        target_gram = [gram(fmap) for fmap in target_features]

        loss_content = 0.0
        # compare their weighted loss
        for lhs, rhs in zip(input_features, target_features):
            # loss_content += F.mse_loss(lhs, rhs)
            loss_content += torch.mean(torch.abs(lhs - rhs))
            # print('loss content',loss_content)

        loss_style = 0
        for j in range(len(input_features)):
            # loss_style += F.mse_loss(input_gram[j], target_gram[j])
            loss_style += torch.mean(torch.abs(input_gram[j] - target_gram[j]))
            # print('loss_style',loss_style)

        return loss_content, loss_style


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()
