import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import read_list, read_data, read_list_2d, read_data_2d
import torch
from scipy import ndimage


class DatasetAllTasks(Dataset):
    def __init__(self, split='train', num_cls=1, task="", repeat=None, transform=None, unlabeled=False,
                 is_val=False, is_2d=False, data_dir=None):
        if data_dir is not None:
            split = data_dir

        if split is None:
            raise ValueError("Either split or data_dir must be provided for DatasetAllTasks")

        self.task = task
        self.is_2d = is_2d
        self.unlabeled = unlabeled
        self.num_cls = num_cls
        self.repeat = repeat
        self.transform = transform
        self.is_val = is_val
        self._path_mode = False

        if self.is_2d:
            if os.path.isdir(split):
                self.ids_list = self._scan_directory(split, is_2d=True)
                self._path_mode = True
            else:
                self.ids_list = read_list_2d(split, task=task)
        else:
            if os.path.isdir(split):
                self.ids_list = self._scan_directory(split, is_2d=False)
                self._path_mode = True
            else:
                self.ids_list = read_list(split, task=task)

        if not self.ids_list:
            raise ValueError(f"No data found for split '{split}' and task '{task}'")

        if self.repeat is None:
            self.repeat = len(self.ids_list)

        print(f'total {len(self.ids_list)} datas')

        if self.is_val:
            self.data_list = {}
            for data_id in tqdm(self.ids_list):
                if self.is_2d:
                    image, label = self._read_data(data_id, use_2d=True)
                else:
                    image, label = self._read_data(data_id)
                self.data_list[data_id] = (image, label)

    def __len__(self):
        return self.repeat

    def _scan_directory(self, directory, is_2d=False):
        if is_2d:
            valid_suffixes = ('.png', '.jpg', '.jpeg', '.npy', '.h5')
        else:
            valid_suffixes = ('.npy',)

        image_paths = []
        if os.path.isdir(os.path.join(directory, 'image')):
            image_dir = os.path.join(directory, 'image')
            for fname in sorted(os.listdir(image_dir)):
                if fname.endswith(valid_suffixes) and ('_label' not in fname):
                    image_paths.append(os.path.join(image_dir, fname))
        else:
            for fname in sorted(os.listdir(directory)):
                if fname.endswith(valid_suffixes) and ('_label' in fname or '_seg' in fname):
                    continue
                if fname.endswith('_image.npy') or not is_2d:
                    image_paths.append(os.path.join(directory, fname))

        return image_paths

    def _read_data(self, data_id, use_2d=False):
        if use_2d:
            return read_data_2d(data_id, task=self.task)
        else:
            return read_data(data_id, task=self.task)

    def _get_data(self, data_id):
        if self.is_val:
            image, label = self.data_list[data_id]
        else:
            if self.is_2d:
                image, label = self._read_data(data_id, use_2d=True)
            else:
                image, label = self._read_data(data_id)
        return data_id, image, label


    def __getitem__(self, index):
        index = index % len(self.ids_list)
        data_id = self.ids_list[index]
        _, image, label = self._get_data(data_id)
        # print("dataset", image.shape, label.shape)
        if self.unlabeled: # <-- for safety
            label[:] = 0
        if "synapse" in self.task:
            image = image.clip(min=-75, max=275)

        elif "mnms" in self.task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        elif "mms2d" in self.task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        elif "udamms" in self.task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        denom = image.max() - image.min()
        if denom > 1e-6:
            image = (image - image.min()) / denom
        else:
            image = image - image.min()

        image = image.astype(np.float32)
        label = label.astype(np.int8)
        # image, label = self.crop_depth(image, label)
        # image, label = self.resize2(image, label)
        # image = torch.from_numpy(image.copy()).unsqueeze(0).float()
        # label = torch.from_numpy(label.copy()).float()
        # print("image", image.shape, label.shape)
        sample = {'image': image, 'label': label}
        # print('shape_before_transform', sample['image'].shape, sample['label'].shape)

        if self.transform:
            sample = self.transform(sample)
        # print("shape", sample['image'].shape, sample['label'].shape, torch.unique(sample['label']))

        return sample

    def crop_depth(self, img, lab, phase='train'):
        # print("dep", img.shape)
        D, H, W = img.shape
        if D > 10:
            if phase == 'train':
                target_ssh = np.random.randint(0, int(D - 10), 1)[0]
                zero_img = img[target_ssh:target_ssh + 10, :, :]
                zero_lab = lab[target_ssh:target_ssh + 10, :, :]
            elif phase == 'valid':
                zero_img, zero_lab = img, lab
            elif phase == 'feta':
                sample_indices = np.random.choice(D, size=10, replace=False)
                zero_img = np.zeros((10, H, W))
                zero_lab = np.zeros((10, H, W))
                for i, index in enumerate(sample_indices):
                    zero_img[i] = img[index]
                    zero_lab[i] = lab[index]
        else:
            zero_img = np.zeros((10, H, W))
            zero_lab = np.zeros((10, H, W))
            zero_img[0:D, :, :] = img
            zero_lab[0:D, :, :] = lab
        return zero_img, zero_lab

    def winadj_mri(array):
        v0 = np.percentile(array, 1)
        v1 = np.percentile(array, 99)
        array[array < v0] = v0
        array[array > v1] = v1
        v0 = array.min()
        v1 = array.max()
        array = (array - v0) / (v1 - v0) * 2.0 - 1.0
        return array

    def resize2(self, img, lab):
        D, H, W = img.shape
        zoom = [1, 256 / H, 256 / W]
        img = ndimage.zoom(img, zoom, order=2)
        lab = ndimage.zoom(lab, zoom, order=0)
        return img, lab
