import torch
import torchvision
import tqdm
from tensorboardX import SummaryWriter
import losses
from torch.nn import init
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from argparse import ArgumentParser
from metric import *
from main_code_.adjust_lr import *
from dataset import *
from AGDNet import AGDNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


###########################################################################################
# parameter
parser = ArgumentParser(description='train')
parser.add_argument('--net_name', type=str, default='AGDNet', help='name of net')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=1500, help='epoch number of end training')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')
parser.add_argument('--batch_size', type=float, default=2, help='batch size')
parser.add_argument('--num_work', type=float, default=2, help='num work,linux=4,win=1')  # train
parser.add_argument('--model_dir', type=str, default='model_sparse', help='trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='results_demo', help='result directory')
parser.add_argument('--run_mode', type=str, default='test', help='train/test/train&test')
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--num-features', type=int, default=32)
parser.add_argument('--layer_num', type=int, default=3, help='phase number of Net')
parser.add_argument('--kernel_size', type=int, default=31)
parser.add_argument('--dataset', default='RealSR', help='GoPro, BSD500, RealSR')
args = parser.parse_args(args=[])
num_work = args.num_work
batch_size = args.batch_size
run_mode = args.run_mode
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
setup_seed(args.seed)

# 设置初始化模糊核为全1张量
h_input = np.ones((args.kernel_size, args.kernel_size), dtype=np.float64) / float(args.kernel_size) / float(
    args.kernel_size)
h_input = torch.FloatTensor(h_input).expand(1, 1, args.kernel_size, args.kernel_size).cuda()
h_ = h_input
for i in range(args.batch_size - 1):
    h_input = torch.cat([h_input, h_], dim=0)

# 数据集位置
# RealSR_dataset
# file_name1_train = 'E:/guozheng_deblur/MGSTNet/Dataset/RealSR_dataset/train/blur/'
# file_name2_train = 'E:/guozheng_deblur/MGSTNet/Dataset/RealSR_dataset/train/sharp/'
# file_name1_val = 'E:/guozheng_deblur/MGSTNet/Dataset/RealSR_dataset/test/blur/'
# file_name2_val = 'E:/guozheng_deblur/MGSTNet/Dataset/RealSR_dataset/test/sharp/'
# GoPro
file_name1_train = 'D:/GDUNet/Dataset/GoPro_dataset/train/blur/'
file_name2_train = 'D:/GDUNet/Dataset/GoPro_dataset/train/sharp/'
# file_name1_val = 'D:/GDUNet/Dataset/GoPro_dataset/test/blur/'
# file_name2_val = 'D:/GDUNet/Dataset/GoPro_dataset/test/sharp/'
# BSD500
# file_name1_train = 'E:/guozheng_deblur/MGSTNet/Dataset/BSD500_dataset/train/blur/'
# file_name2_train = 'E:/guozheng_deblur/MGSTNet/Dataset/BSD500_dataset/train/sharp/'
# file_name1_val = 'E:/guozheng_deblur/MGSTNet/Dataset/BSD500_dataset/test/blur/'
# file_name2_val = 'E:/guozheng_deblur/MGSTNet/Dataset/BSD500_dataset/test/sharp/'

file_name1_val = 'C:/Users/guozheng/Desktop/test/blur/'
file_name2_val = 'C:/Users/guozheng/Desktop/test/sharp/'

if run_mode == 'train':
    train_dataset = UnAlignedDataset(file_name1_train, file_name2_train, 'train')
    val_dataset = UnAlignedDataset(file_name1_val, file_name2_val, 'val')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_work,
        drop_last=True,
        pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
elif run_mode == 'test':
    # Load GoPro_dataset
    test_dataset = UnAlignedDataset(file_name1_val, file_name2_val, 'test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

# define model save dir
model_dir = "../%s/%s_layer_%d_lr_%f_%s" % (args.model_dir, args.net_name, layer_num, learning_rate, args.dataset)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# Define initialize parametes
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


# model = MGSTNet(layer_num)
model = AGDNet(3, 32, 16, 8, -1, 8, 32, 256)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
my_lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=exp_lr_scheduler)
writer = SummaryWriter(
    "../%s/%s_layer_%d_lr_%f_%s" % (args.model_dir, args.net_name, layer_num, learning_rate, args.dataset))
# 继续训练
if start_epoch > 0:  # train stop and restart
    for i in range(start_epoch):
        optimizer.zero_grad()
        optimizer.step()  # 更新参数
        my_lr_scheduler.step()
    pre_model_dir = model_dir
    checkpoint = torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch % 100 - 1))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def saveimg(img, name, cmap):
    # 使用utils.save_image保存灰度图像 gray/rgb
    torchvision.utils.save_image(img, name)  # range=(0, 1),


################################################################################################
def val(model, best_RMSE, val_dataloader, h_input, epoch):
    model = model.eval()
    with torch.no_grad():
        RMSE_total = []
        PSNR_total = []
        SSIM_total = []
        for i, data_fbp in enumerate(tqdm.tqdm(val_dataloader)):
            x = data_fbp['A']
            y = data_fbp['B']
            batch_x = x
            batch_x = batch_x.to(device)
            y = y.to(device)
            [x_output, h_output, u_all, h_all] = model(batch_x, batch_x, h_input)

            p_reg = compute_PSNR(x_output, y, 1)
            s_reg1 = compute_SSIM(x_output[0, 0, :, :], y[0, 0, :, :], 1)
            s_reg2 = compute_SSIM(x_output[0, 1, :, :], y[0, 1, :, :], 1)
            s_reg3 = compute_SSIM(x_output[0, 2, :, :], y[0, 2, :, :], 1)
            s_reg = 1.0 / 3.0 * (s_reg1 + s_reg2 + s_reg3)
            m_reg = compute_RMSE(x_output, y)

            RMSE_total.append(m_reg)
            PSNR_total.append(p_reg)
            SSIM_total.append(s_reg)

            # RealSR_dataset
            if i == 0:
                name1 = 'val_sample/test_%d_%.2f.png' % (epoch, p_reg)
                # name2 = 'val_sample/test__k_%d.png' % epoch
                saveimg(x_output, name1, 'rgb')
                # saveimg(h_output, name2, 'gray')
            # if i == 629:
            #     break

        aver_RMSE = np.array(RMSE_total).mean()
        aver_PSNR = np.array(PSNR_total).mean()
        aver_SSIM = np.array(SSIM_total).mean()

        # print('aver_RMSE=%f,aver_PSNR=%f,aver_SSIM=%f' % (aver_RMSE, aver_PSNR, aver_SSIM))
        print('aver_RMSE=%f' % np.array(RMSE_total).mean(), '| std=%f' % np.array(RMSE_total).std(),
              '| median=%f' % np.median(np.array(RMSE_total)),
              '| max=%f' % np.array(RMSE_total).max(), '| min=%f' % np.array(RMSE_total).min())
        print('aver_PSNR =%f' % np.array(PSNR_total).mean(), '| std=%f' % np.array(PSNR_total).std(),
              '| median=%f' % np.median(np.array(PSNR_total)),
              '| max=%f' % np.array(PSNR_total).max(), '| min=%f' % np.array(PSNR_total).min())
        print('aver_SSIM =%f' % np.array(SSIM_total).mean(), '| std=%f' % np.array(SSIM_total).std(),
              '| median=%f' % np.median(np.array(SSIM_total)),
              '| max=%f' % np.array(SSIM_total).max(), '| min=%f' % np.array(SSIM_total).min())

        if aver_RMSE < best_RMSE:
            best_RMSE = aver_RMSE
            print('===========save best model!===========')
            torch.save(model.state_dict(), r'./AGDNet_' + str(args.dataset) + '_7_2_best.pth')
        return best_RMSE, aver_RMSE, aver_PSNR, aver_SSIM


# torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    if run_mode == 'train':
        best_RMSE, aver_RMSE, aver_PSNR, aver_SSIM = 1, 1, 0, 0
        loss_list = []
        RMSE_list = []
        PSNR_list = []
        SSIM_list = []
        L1LOSS = losses.CharbonnierLoss()
        L2LOSS = losses.MSELoss()
        criterion_tv = losses.TVLoss()
        for epoch_i in range(start_epoch + 1, end_epoch + 1):
            print('--------------------------------------------')
            print('     epoch_i =', epoch_i, ', learn_rate =', optimizer.state_dict()['param_groups'][0]['lr'], )
            writer.add_scalar('learn rate/epoch', optimizer.state_dict()['param_groups'][0]['lr'], epoch_i)
            model = model.train()
            step = 0
            epoch_loss = 0
            if epoch_i <= 100:
                criterion = L2LOSS
            else:
                criterion = L1LOSS
            optimizer.zero_grad()
            for i, data_fbp in enumerate(tqdm.tqdm(train_dataloader)):
                step = step + 1
                batch_x = data_fbp['A']
                y = data_fbp['B']
                batch_x = batch_x.to(device)
                y = y.to(device)
                h_input = h_input.to(device)
                [x_output, h_output, u_all, h_all] = model(batch_x, batch_x, h_input)
                # 3个阶段，cat前两个阶段所需的标签
                batch_y = y
                for n in range(1):
                    batch_y = torch.cat([batch_y, y], dim=0)
                loss_u_mid = criterion(batch_y, u_all[batch_size:batch_size * 3, :, :, :])
                loss_h_mid = 0
                loss_h = 0
                for n in range(batch_size):
                    blur = torch.unsqueeze(batch_x[n, :, :, :], dim=0)
                    sharp = torch.unsqueeze(y[n, :, :, :], dim=0)

                    h_pred = torch.unsqueeze(h_all[batch_size * 3 + n, :, :, :], dim=0)
                    u_pred1 = F.pad(input=sharp, pad=(
                        h_pred.size()[2] // 2, h_pred.size()[2] // 2, h_pred.size()[2] // 2, h_pred.size()[2] // 2),
                                    mode='reflect')
                    ublur_0 = F.conv2d(u_pred1[:, 0, :, :], h_pred, padding=0)
                    ublur_1 = F.conv2d(u_pred1[:, 1, :, :], h_pred, padding=0)
                    ublur_2 = F.conv2d(u_pred1[:, 2, :, :], h_pred, padding=0)
                    ublur = torch.unsqueeze(torch.cat([ublur_0, ublur_1, ublur_2], dim=0), dim=0)
                    loss_h = loss_h + criterion(ublur, blur) + criterion_tv(h_pred)

                    for m in range(2):
                        h_pred = torch.unsqueeze(h_all[batch_size * (m + 1) + n, :, :, :], dim=0)
                        reblur_0 = F.conv2d(u_pred1[:, 0, :, :], h_pred, padding=0)
                        reblur_1 = F.conv2d(u_pred1[:, 1, :, :], h_pred, padding=0)
                        reblur_2 = F.conv2d(u_pred1[:, 2, :, :], h_pred, padding=0)
                        reblur = torch.unsqueeze(torch.cat([reblur_0, reblur_1, reblur_2], dim=0), dim=0)
                        loss_h_mid = loss_h_mid + criterion(reblur, blur) + criterion_tv(h_pred)

                loss_x = criterion(x_output, y)
                loss_all = loss_x + loss_h + 0.5 * (loss_u_mid + loss_h_mid) + criterion_tv(h_output)
                epoch_loss += loss_all.item()
                loss_all.backward()
                if (i + 1) % 8 == 0:
                    optimizer.step()  # 更新参数
                    optimizer.zero_grad()  # 梯度清零
                # optimizer.step()
            # 最后一个不足16的batch的梯度更新
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 梯度清零

            print('Epoch = ', epoch_i, '| Epoch loss =', epoch_loss)
            writer.add_scalar('loss/epoch', epoch_loss, epoch_i)
            if epoch_i % 5 == 0:
                best_RMSE, aver_RMSE, aver_PSNR, aver_SSIM = val(model, best_RMSE, val_dataloader, h_input, epoch_i)
                writer.add_scalar('val_PSNR/epoch', aver_PSNR, epoch_i)
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            if epoch_i % 1 == 0:
                torch.save(checkpoint, "./%s/net_params_%d.pkl" % (model_dir, epoch_i % 100))
            my_lr_scheduler.step()

    elif run_mode == 'test':
        if args.dataset == 'GoPro':
            checkpoint = torch.load('AGDNet_GoPro_7_2_best.pth')
            model.load_state_dict(checkpoint)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            checkpoint = torch.load('AGDNet_RealSR_6_13_best.pth')
            model.load_state_dict(checkpoint)
        #checkpoint = torch.load(r'D:\MGSTNet\net_params_56.pkl')
        #model.load_state_dict(checkpoint['model'])
        # model.load_state_dict(checkpoint)
        # model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, end_epoch)))
        model = model.eval()
        with torch.no_grad():
            PSNR_total = []
            SSIM_total = []
            x_val_all = []
            x_in = []
            y_out = []
            u_all_sum = []
            h_all_sum = []
            for i, data_fbp in enumerate(tqdm.tqdm(test_dataloader)):
                x = data_fbp['A']
                y = data_fbp['B']
                batch_x = x
                batch_x = batch_x.to(device)
                h_input = h_input.to(device)
                y = y.to(device)
                [x_output, h_output, u_all, h_all], ift, kft = model(batch_x, batch_x, h_input)
                kftname = 'D:/GDUNet/kft.png'
                saveimg(kft[0, 0, :, :], kftname, 'rgb')
                for t in range(127):
                    iftname = 'D:/GDUNet/ift/%d.png' % t
                    saveimg(ift[0, t, :, :], iftname, 'rgb')
                p_reg = compute_PSNR(x_output, y, 1)
                s_reg1 = compute_SSIM(x_output[0, 0, :, :], y[0, 0, :, :], 1)
                s_reg2 = compute_SSIM(x_output[0, 1, :, :], y[0, 1, :, :], 1)
                s_reg3 = compute_SSIM(x_output[0, 2, :, :], y[0, 2, :, :], 1)
                s_reg = 1.0 / 3.0 * (s_reg1 + s_reg2 + s_reg3)
                PSNR_total.append(p_reg)
                SSIM_total.append(s_reg)
                imgname = ('D:/GDUNet/' + args.result_dir + '/img/%d.png') % i
                kernelname = ('D:/GDUNet/' + args.result_dir + '/kernel/%d_k.png') % i
                saveimg(x_output[0, :, :, :], imgname, 'rgb')
                saveimg(h_output[0, 0, :, :] / torch.max(h_output[0, 0, :, :]), kernelname, 'gray')

                # imgname1 = ('D:/GDUNet/' + args.result_dir + '/img/%d_1.png') % i
                # imgname2 = ('D:/GDUNet/' + args.result_dir + '/img/%d_2.png') % i
                # imgname3 = ('D:/GDUNet/' + args.result_dir + '/img/%d_3.png') % i
                # saveimg(u_all[1, :, :, :], imgname1, 'rgb')
                # saveimg(u_all[2, :, :, :], imgname2, 'rgb')
                # saveimg(u_all[3, :, :, :], imgname3, 'rgb')
                # p_reg = compute_PSNR(u_all[1, :, :, :], y, 1)
                # s_reg1 = compute_SSIM(u_all[1, 0, :, :], y[0, 0, :, :], 1)
                # s_reg2 = compute_SSIM(u_all[1, 1, :, :], y[0, 1, :, :], 1)
                # s_reg3 = compute_SSIM(u_all[1, 2, :, :], y[0, 2, :, :], 1)
                # s_reg = 1.0 / 3.0 * (s_reg1 + s_reg2 + s_reg3)
                # print('PSNR=%f, SSIM=%f' % (p_reg, s_reg))
                # p_reg = compute_PSNR(u_all[2, :, :, :], y, 1)
                # s_reg1 = compute_SSIM(u_all[2, 0, :, :], y[0, 0, :, :], 1)
                # s_reg2 = compute_SSIM(u_all[2, 1, :, :], y[0, 1, :, :], 1)
                # s_reg3 = compute_SSIM(u_all[2, 2, :, :], y[0, 2, :, :], 1)
                # s_reg = 1.0 / 3.0 * (s_reg1 + s_reg2 + s_reg3)
                # print('PSNR=%f, SSIM=%f' % (p_reg, s_reg))
                # p_reg = compute_PSNR(u_all[3, :, :, :], y, 1)
                # s_reg1 = compute_SSIM(u_all[3, 0, :, :], y[0, 0, :, :], 1)
                # s_reg2 = compute_SSIM(u_all[3, 1, :, :], y[0, 1, :, :], 1)
                # s_reg3 = compute_SSIM(u_all[3, 2, :, :], y[0, 2, :, :], 1)
                # s_reg = 1.0 / 3.0 * (s_reg1 + s_reg2 + s_reg3)
                # print('PSNR=%f, SSIM=%f' % (p_reg, s_reg))

            aver_PSNR = np.array(PSNR_total).mean()
            aver_SSIM = np.array(SSIM_total).mean()
            print('aver_PSNR=%f, aver_SSIM=%f' % (aver_PSNR, aver_SSIM))

    #########################################################################################
