import random
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if "map" in dir:
        for fname in sorted(os.listdir(dir)):
            path = os.path.join(dir, fname)
            images.append(path)
    else:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSizeX, opt.loadSizeY]
        transform_list.append(transforms.Resize(osize, InterpolationMode.BICUBIC))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSizeX)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


class UnAlignedDataset(BaseDataset):
    def __init__(self, path_blur, path_sharp, run_mode):
        super().__init__()
        self.run_mode = run_mode
        self.dir_A = path_blur
        self.dir_B = path_sharp
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.patchsize = 256

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        A = self.transform(A)
        B = self.transform(B)

        if self.run_mode == 'train':
            width = A.size(2)
            if width > 128:
                w_total = A.size(2)
                w = int(w_total / 2)
                h = A.size(1)
                # 随机裁剪出256*256的patch
                w_offset = random.randint(0, max(0, w - self.patchsize - 1))
                h_offset = random.randint(0, max(0, h - self.patchsize - 1))
                inp_img = A[:, h_offset:h_offset + self.patchsize, w_offset:w_offset + self.patchsize]
                tar_img = B[:, h_offset:h_offset + self.patchsize, w_offset:w_offset + self.patchsize]

            aug = random.randint(0, 8)

            # Data Augmentations
            if aug == 1:
                inp_img = inp_img.flip(1)
                tar_img = tar_img.flip(1)
            elif aug == 2:
                inp_img = inp_img.flip(2)
                tar_img = tar_img.flip(2)
            elif aug == 3:
                inp_img = torch.rot90(inp_img, dims=(1, 2))
                tar_img = torch.rot90(tar_img, dims=(1, 2))
            elif aug == 4:
                inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
                tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            elif aug == 5:
                inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
                tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            elif aug == 6:
                inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
                tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
            elif aug == 7:
                inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
                tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

            return {'A': inp_img, 'B': tar_img, 'A_paths': A_path, 'B_paths': B_path}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'UnAlignedDataset'
