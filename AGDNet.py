# from DeepRFT_MIMO import *
import torch
import torch.nn as nn
import numpy as np
from basicsr.models.archs.FNAFNet_arch import *


class Gradient_Net(nn.Module):
    def __init__(self, ratio):
        super(Gradient_Net, self).__init__()
        self.ratio = ratio
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        x_0 = F.pad(input=x[:, 0, :, :].unsqueeze(1), pad=(
            self.weight_x.size()[2] // 2, self.weight_x.size()[2] // 2, self.weight_x.size()[2] // 2,
            self.weight_x.size()[2] // 2),
                    mode='reflect')
        x_1 = F.pad(input=x[:, 1, :, :].unsqueeze(1), pad=(
            self.weight_x.size()[2] // 2, self.weight_x.size()[2] // 2, self.weight_x.size()[2] // 2,
            self.weight_x.size()[2] // 2),
                    mode='reflect')
        x_2 = F.pad(input=x[:, 2, :, :].unsqueeze(1), pad=(
            self.weight_x.size()[2] // 2, self.weight_x.size()[2] // 2, self.weight_x.size()[2] // 2,
            self.weight_x.size()[2] // 2),
                    mode='reflect')
        # 水平梯度
        grad_x_r = F.conv2d(x_0, self.weight_x / self.ratio, stride=1, padding=0)
        grad_x_g = F.conv2d(x_1, self.weight_x / self.ratio, stride=1, padding=0)
        grad_x_b = F.conv2d(x_2, self.weight_x / self.ratio, stride=1, padding=0)
        grad_x = torch.cat([grad_x_r, grad_x_g, grad_x_b], 1)
        # 垂直梯度
        grad_y_r = F.conv2d(x_0, self.weight_y / self.ratio, stride=1, padding=0)
        grad_y_g = F.conv2d(x_1, self.weight_y / self.ratio, stride=1, padding=0)
        grad_y_b = F.conv2d(x_2, self.weight_y / self.ratio, stride=1, padding=0)
        grad_y = torch.cat([grad_y_r, grad_y_g, grad_y_b], 1)
        return grad_x, grad_y


class Gradient_Net_transpose(nn.Module):
    # 梯度转置算子
    def __init__(self, ratio):
        super(Gradient_Net_transpose, self).__init__()
        # Sobel算子
        self.ratio = ratio
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x, xORy):

        if xORy == 'x':
            # 水平梯度
            grad_x_r_T = F.conv_transpose2d(x[:, 0, :, :].unsqueeze(1), self.weight_x / self.ratio, stride=1, padding=1)
            grad_x_g_T = F.conv_transpose2d(x[:, 1, :, :].unsqueeze(1), self.weight_x / self.ratio, stride=1, padding=1)
            grad_x_b_T = F.conv_transpose2d(x[:, 2, :, :].unsqueeze(1), self.weight_x / self.ratio, stride=1, padding=1)
            grad_T = torch.cat([grad_x_r_T, grad_x_g_T, grad_x_b_T], 1)
        else:
            # 垂直梯度
            grad_y_r_T = F.conv_transpose2d(x[:, 0, :, :].unsqueeze(1), self.weight_y / self.ratio, stride=1, padding=1)
            grad_y_g_T = F.conv_transpose2d(x[:, 1, :, :].unsqueeze(1), self.weight_y / self.ratio, stride=1, padding=1)
            grad_y_b_T = F.conv_transpose2d(x[:, 2, :, :].unsqueeze(1), self.weight_y / self.ratio, stride=1, padding=1)
            grad_T = torch.cat([grad_y_r_T, grad_y_g_T, grad_y_b_T], 1)

        return grad_T


class GPMNet(nn.Module):
    # 梯度细化网络
    def __init__(self, out_channel, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m):
        super(GPMNet, self).__init__()
        self.module = nn.Sequential(
            WNAFBlock_ffc3_gelu_sin_2block(out_channel, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m),
            WNAFBlock_ffc3_gelu_sin_2block(out_channel, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m),
        )

    def forward(self, x):
        return self.module(x)


class ImageFeatureGuideNet(nn.Module):
    # 图像特征核映射
    def __init__(self):
        super(ImageFeatureGuideNet, self).__init__()
        self.up = nn.PixelShuffle(2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(128, 128 // 16, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(128 // 16, 128, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

        self.module = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.up(x)
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        s = x * y.expand_as(x)
        return self.module(s), s


def saveimg(img, name, cmap):
    from torchvision import utils
    # 使用utils.save_image保存灰度图像
    utils.save_image(img, name, nrow=1, normalize=True, range=(0, 1), cmap=cmap)


class BasicLayer(torch.nn.Module):
    # 完成一次迭代更新
    def __init__(self, n_feats=32, num_heads_m=16, window_size_m=8, window_size_m_fft=-1, window_sizex_m=8):
        super(BasicLayer, self).__init__()

        self.orb1 = nn.Sequential(
            WNAFBlock_ffc3_gelu_sin_2block(n_feats, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m),
            WNAFBlock_ffc3_gelu_sin_2block(n_feats, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m),
        )
        self.orb2 = nn.Sequential(
            WNAFBlock_ffc3_gelu_sin_2block(n_feats, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m),
            WNAFBlock_ffc3_gelu_sin_2block(n_feats, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m),
        )
        ################################################################################################
        self.lambda_step_u = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.lambda_step_u_x = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.lambda_step_u_y = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.lambda_step_h = nn.Parameter(torch.Tensor([0.00001]), requires_grad=True)
        self.soft_thr_h = nn.Parameter(torch.Tensor([0.0002]), requires_grad=True)
        ################################################################################################
        self.conv1_forward_h = nn.Conv2d(in_channels=1, out_channels=n_feats, kernel_size=3, padding=3 // 2)
        self.conv4_backward_h = nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=3, padding=3 // 2)
        self.conv11_kernel_balance = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=1 // 2)
        ################################################################################################
        self.IPMNet = FNAFNet(img_channel=3,
                              width=32,
                              middle_blk_num=1,
                              enc_blk_nums=[1, 1, 1, 16],
                              dec_blk_nums=[1, 1, 1, 1],
                              window_size_e=[64, 32, 16, 8],
                              window_size_m=[8],
                              window_size_e_fft=[64, 32, 16, -1],
                              window_size_m_fft=[-1],
                              window_sizex_e=[8, 8, 8, 8],
                              window_sizex_m=[8],
                              num_heads_e=[1, 2, 4, 8],
                              num_heads_m=[16])

        self.gxi_conv = nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=3, stride=1, padding=1)
        self.gyi_conv = nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=3, stride=1, padding=1)
        self.gxo_conv = nn.Conv2d(in_channels=n_feats, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.gyo_conv = nn.Conv2d(in_channels=n_feats, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.GPMNet_x = GPMNet(n_feats, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m)
        self.GPMNet_y = GPMNet(n_feats, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m)
        self.gra = Gradient_Net(1)
        self.gra_T = Gradient_Net_transpose(1)

        self.IBFG = ImageFeatureGuideNet()

    def forward(self, u_k_1, u0_init, h_k_1):
        B = u_k_1.size()[0]  # 图片数
        # 前一次的预测u_k-1
        u_pred_k_1 = u_k_1
        # 对每张图和每个模糊核分别执行IGFM IPGD IPMM KPGD KPMM模块
        h_pred_k = []
        u_input_k = []
        # 先图像后模糊核
        # IGFM + IPGD
        for i in range(B):
            u0 = torch.unsqueeze(u0_init[i, :, :, :], dim=0)
            h = torch.unsqueeze(h_k_1[i, :, :, :], dim=0)
            #  u_rotate = torch.unsqueeze(u_rotate_k_1[i, :, :, :], dim=0)
            h_rotate = torch.rot90(torch.rot90(h, dims=(2, 3)), dims=(2, 3))
            u_pred1_nopad = torch.unsqueeze(u_pred_k_1[i, :, :, :], dim=0)
            # IPGD
            u1 = F.pad(input=u_pred1_nopad,
                       pad=(h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2),
                       mode='reflect')
            u2_0 = F.conv2d(u1[0, 0, :, :].expand(1, 1, u1.size()[2], u1.size()[3]), h, padding=0) - \
                   u0[0, 0, :, :].expand(1, 1, u0.size()[2], u0.size()[3])
            u2_1 = F.conv2d(u1[0, 1, :, :].expand(1, 1, u1.size()[2], u1.size()[3]), h, padding=0) - \
                   u0[0, 1, :, :].expand(1, 1, u0.size()[2], u0.size()[3])
            u2_2 = F.conv2d(u1[0, 2, :, :].expand(1, 1, u1.size()[2], u1.size()[3]), h, padding=0) - \
                   u0[0, 2, :, :].expand(1, 1, u0.size()[2], u0.size()[3])
            u2_0 = F.pad(input=u2_0, pad=(h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2),
                         mode='reflect')
            u2_1 = F.pad(input=u2_1, pad=(h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2),
                         mode='reflect')
            u2_2 = F.pad(input=u2_2, pad=(h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2),
                         mode='reflect')
            u3_0 = F.conv2d(u2_0, h_rotate, padding=0)
            u3_1 = F.conv2d(u2_1, h_rotate, padding=0)
            u3_2 = F.conv2d(u2_2, h_rotate, padding=0)
            u3 = torch.cat([u3_0, u3_1, u3_2], dim=1)
            # IGFM
            gra_u_x, gra_u_y = self.gra(u_pred1_nopad)
            # saveimg(gra_u_x, 'test_images/x_grad.png', 'gray')
            # saveimg(gra_u_y, 'test_images/y_grad.png', 'gray')
            gra_u = self.gxo_conv(self.GPMNet_x(self.gxi_conv(gra_u_x)))
            gra_v = self.gyo_conv(self.GPMNet_y(self.gyi_conv(gra_u_y)))
            # saveimg(gra_u, 'test_images/u_grad.png', 'gray')
            # saveimg(gra_v, 'test_images/v_grad.png', 'gray')
            u4 = self.gra_T(gra_u_x - gra_u, 'x')
            u5 = self.gra_T(gra_u_y - gra_v, 'y')
            u_input = u_pred1_nopad - self.lambda_step_u * u3 - self.lambda_step_u_x * u4 - self.lambda_step_u_y * u5
            # saveimg(u_input, 'test_images/u_update.png', 'rgb')
            if i == 0:
                u_input_k = u_input
            else:
                u_input_k = torch.cat([u_input_k, u_input], dim=0)

        # IPMM
        u_output_k, mid_features = self.IPMNet(u_input_k)  # 1，512，16，16
        u_pred = u_output_k + u_input_k

        # KPGD + KPMM
        # u_k 旋转180°得到 u_k(-x,-y)
        u_rotate_k_1 = torch.rot90(torch.rot90(u_pred, dims=(2, 3)), dims=(2, 3))
        for i in range(B):
            I_mid_features = torch.unsqueeze(mid_features[i, :, :, :], dim=0)
            u0 = torch.unsqueeze(u0_init[i, :, :, :], dim=0)
            h = torch.unsqueeze(h_k_1[i, :, :, :], dim=0)
            u_rotate = torch.unsqueeze(u_rotate_k_1[i, :, :, :], dim=0)
            u_pred1_nopad = torch.unsqueeze(u_pred[i, :, :, :], dim=0)
            #################################################################################
            # KGDN 模糊核梯度下降模块
            u_pred1 = F.pad(input=u_pred1_nopad,
                            pad=(h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2),
                            mode='reflect')
            # 0 1 2 对应 R G B通道分开卷积
            h2_0 = F.conv2d(u_pred1[0, 0, :, :].expand(1, 1, u_pred1.size()[2], u_pred1.size()[3]), h, padding=0) - \
                   u0[0, 0, :, :].expand(1, 1, u0.size()[2], u0.size()[3])
            h2_1 = F.conv2d(u_pred1[0, 1, :, :].expand(1, 1, u_pred1.size()[2], u_pred1.size()[3]), h, padding=0) - \
                   u0[0, 1, :, :].expand(1, 1, u0.size()[2], u0.size()[3])
            h2_2 = F.conv2d(u_pred1[0, 2, :, :].expand(1, 1, u_pred1.size()[2], u_pred1.size()[3]), h, padding=0) - \
                   u0[0, 2, :, :].expand(1, 1, u0.size()[2], u0.size()[3])
            h2_0 = F.pad(input=h2_0, pad=(h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2),
                         mode='reflect')
            h2_1 = F.pad(input=h2_1, pad=(h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2),
                         mode='reflect')
            h2_2 = F.pad(input=h2_2, pad=(h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2, h.size()[2] // 2),
                         mode='reflect')
            h3_0 = F.conv2d(h2_0, u_rotate[0, 0, :, :].expand(1, 1, u_rotate.size()[2], u_rotate.size()[3]), padding=0)
            h3_1 = F.conv2d(h2_1, u_rotate[0, 1, :, :].expand(1, 1, u_rotate.size()[2], u_rotate.size()[3]), padding=0)
            h3_2 = F.conv2d(h2_2, u_rotate[0, 2, :, :].expand(1, 1, u_rotate.size()[2], u_rotate.size()[3]), padding=0)
            # 将三个方向的梯度相加，然后下降， self.lambda_step_h即原文中的参数 u_k
            # KPGD
            h_input = h - self.lambda_step_h * (h3_0 + h3_1 + h3_2)
            #################################################################################
            # KPMM 模糊核近端隐射模块
            h_if, s = self.IBFG(I_mid_features)
            h_forward = self.orb1(self.conv1_forward_h(h_input))
            h_backward = self.conv4_backward_h(
                self.orb2(torch.mul(torch.sign(h_forward), F.relu(torch.abs(h_forward) - self.soft_thr_h))))
            h_pred = self.conv11_kernel_balance(torch.cat([h_backward, h_input, h_if], dim=1))
            # 负值置0并且归一化
            h_pred = F.relu(h_pred) + 1e-8
            h_pred = h_pred / torch.sum(h_pred)
            if i == 0:
                h_pred_k = h_pred
            else:
                h_pred_k = torch.cat([h_pred_k, h_pred], dim=0)

        return [u_pred, h_pred_k], s, h_if


class AGDNet(torch.nn.Module):
    def __init__(self, LayerNo, n_feats, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m, overlap_size,
                 train_size):
        super(AGDNet, self).__init__()
        self.overlap_size = (overlap_size, overlap_size)
        self.train_size = train_size
        self.kernel_size = [train_size, train_size]
        onelayer = []
        self.LayerNo = LayerNo
        for i in range(LayerNo):
            onelayer.append(BasicLayer(n_feats, num_heads_m, window_size_m, window_size_m_fft, window_sizex_m))
        self.fcs = nn.ModuleList(onelayer)

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        # assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = self.overlap_size  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])
        self.stride = stride
        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.overlap_size[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size[0], :] *= self.fuse_matrix_h2.to(outs.device)
            if i + k1 * 2 - self.ek1 < h:
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
                outs[cnt, :, -self.overlap_size[0]:, :] *= self.fuse_matrix_h1.to(outs.device)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] *= self.fuse_matrix_eh2.to(outs.device)
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] *= self.fuse_matrix_eh1.to(outs.device)

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size[1]] *= self.fuse_matrix_w2.to(outs.device)
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size[1]:] *= self.fuse_matrix_w1.to(outs.device)
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] *= self.fuse_matrix_ew2.to(outs.device)
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] *= self.fuse_matrix_ew1.to(outs.device)
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt

    def forward(self, u0, u, h):
        _, _, H, W = u.shape
        u_all = u0
        h_all = h
        input_h = h
        if H != self.train_size:
            u = self.grids(u)
            u0 = self.grids(u0)
            b, _, _, _ = u.shape
            h_block = []
            u_block_stage0 = []
            u_block_stage1 = []
            u_block_stage2 = []
            for i in range(b):
                input_u0 = torch.unsqueeze(u0[i, :, :, :], dim=0)
                input_u = torch.unsqueeze(u[i, :, :, :], dim=0)
                for j in range(self.LayerNo):
                    [u_new, h_new], ift, kft = self.fcs[j](input_u, input_u0, input_h)
                    input_u = u_new
                    input_h = h_new
                    if j == 0:
                        if i == 0:
                            u_block_stage0 = u_new
                        else:
                            u_block_stage0 = torch.cat([u_block_stage0, u_new])
                    if j == 1:
                        if i == 0:
                            u_block_stage1 = u_new
                        else:
                            u_block_stage1 = torch.cat([u_block_stage1, u_new])
                    if j == 2:
                        if i == 0:
                            u_block_stage2 = u_new
                        else:
                            u_block_stage2 = torch.cat([u_block_stage2, u_new])
                if i == 0:
                    h_block = input_h
                else:
                    h_block = torch.cat([h_block, input_h])

            u_block_stage0 = self.grids_inverse(u_block_stage0)
            u_block_stage1 = self.grids_inverse(u_block_stage1)
            u_block_stage2 = self.grids_inverse(u_block_stage2)

            u_all = torch.cat([u_all, u_block_stage0, u_block_stage1, u_block_stage2], dim=0)
            h_all = None  # torch.cat([h_all, h_new], dim=0)
            u_final = u_block_stage2
            h_final = h_block
        else:
            input_u0 = u0
            input_u = u
            input_h = h
            for i in range(self.LayerNo):
                [u_new, h_new], ift, kft = self.fcs[i](input_u, input_u0, input_h)
                input_u = u_new
                input_h = h_new
                u_all = torch.cat([u_all, u_new], dim=0)
                h_all = torch.cat([h_all, h_new], dim=0)
            u_final = input_u
            h_final = input_h

        return [u_final, h_final, u_all, h_all], ift, kft


if __name__ == '__main__':
    kernel_size = 31
    bs = 1
    input = torch.rand(1, 3, 361, 481).cuda()
    h_input = np.ones((kernel_size, kernel_size), dtype=np.float64) / float(kernel_size) / float(kernel_size)
    h_input = torch.FloatTensor(h_input).expand(1, 1, kernel_size, kernel_size).cuda()
    h_ = h_input
    for i in range(bs - 1):
        h_input = torch.cat([h_input, h_], dim=0)
    net = AGDNet(3, 32, 16, 8, -1, 8, 32, 256).cuda()
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("总参数数量和：" + str(k))
    net.eval()
    with torch.no_grad():
        output = net(input, input, h_input)
