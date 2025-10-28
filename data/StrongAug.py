import random
import torch
import copy
from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
import torch.nn.functional as F
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform, BrightnessTransform
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, rotate_coords_2d, rotate_coords_3d, scale_coords

import numpy as np
import albumentations as A
import cv2


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':
                ret_dict[key] = torch.from_numpy(item).unsqueeze(0).float()
            elif key == 'label':
                ret_dict[key] = torch.from_numpy(item).long()
            else:
                raise ValueError(key)
        return ret_dict


def augment_rotation(data, seg, patch_size, patch_center_dist_from_border=30,
                     angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                     border_mode_data='constant', border_cval_data=0, order_data=3,
                     border_mode_seg='constant', border_cval_seg=0, order_seg=1, random_crop=False):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        a_x = np.random.uniform(angle_x[0], angle_x[1])
        if dim == 3:
            a_y = 0
            a_z = 0
            coords = rotate_coords_3d(coords, a_x, a_y, a_z)
        else:
            coords = rotate_coords_2d(coords, a_x)

        # now find a nice center location
        for d in range(dim):
            if random_crop:
                ctr = np.random.uniform(patch_center_dist_from_border[d],
                                        data.shape[d + 2] - patch_center_dist_from_border[d])
            else:
                ctr = data.shape[d + 2] / 2. - 0.5
            coords[d] += ctr
        for channel_id in range(data.shape[1]):
            data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                 border_mode_data, cval=border_cval_data)
        if seg is not None:
            for channel_id in range(seg.shape[1]):
                seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg,
                                                                    is_seg=True)
    return data_result, seg_result


def augment_scale(data, seg, patch_size, patch_center_dist_from_border=30,
                  scale=(0.6, 1.0), border_mode_data='constant', border_cval_data=0, order_data=3,
                  border_mode_seg='constant', border_cval_seg=0, order_seg=1, random_crop=False,
                  independent_scale_for_each_axis=False, p_independent_scale_per_axis: int = 1):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)

        if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
            sc = []
            for _ in range(dim):
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc.append(np.random.uniform(scale[0], 1))
                else:
                    sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
        else:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])
        coords = scale_coords(coords, sc)

        for d in range(dim):
            ctr = data.shape[d + 2] / 2. - 0.5
            coords[d] += ctr


        for channel_id in range(data.shape[1]):
            data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                 border_mode_data, cval=border_cval_data)
        if seg is not None:
            for channel_id in range(seg.shape[1]):
                seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg,
                                                                    is_seg=True)

    return data_result, seg_result




class RandomSelect(Compose):
    def __init__(self, transforms, sample_num=1):
        super(RandomSelect, self).__init__(transforms)
        self.transforms = transforms
        self.sample_num = sample_num

    def update_list(self, list):
        self.list = list

    def __call__(self, data_dict):
        # print("random", self.transforms, self.sample_num)
        self.sample_num = min(self.sample_num, len(self.transforms))
        tr_transforms = random.sample(self.transforms, k=self.sample_num)
        list = copy.deepcopy(self.list)
        if tr_transforms is not None:
            for i in range(len(tr_transforms)):
                list.insert(3, tr_transforms[i])

        for t in list:
            data_dict = t(**data_dict)
        del tr_transforms
        del list

        for key in data_dict.keys():
            if key == "image":
                data_dict[key] = data_dict[key].squeeze(0)
            elif key == "label":
                data_dict[key] = data_dict[key].squeeze(0).squeeze(0)
        return data_dict


class ScaleTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1.0, scale=(0.6, 1.0)):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample
        self.scale = scale
    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        for b in range(data.shape[0]):
            if random.random() < self.p_per_sample:
                data, label = augment_scale(data, label, patch_size=data.shape[2:5], scale=self.scale)
        data_dict[self.data_key] = data
        data_dict[self.label_key] = label
        return data_dict



class RotationTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1.0):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        for b in range(data.shape[0]):
            if random.random() < self.p_per_sample:
                data, label = augment_rotation(data, label, patch_size=data.shape[2:5])
        data_dict[self.data_key] = data
        data_dict[self.label_key] = label
        return data_dict


class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ret_dict = {}
        resize_shape=(self.output_size[0],
                      self.output_size[1],
                      self.output_size[2])
        for key in sample.keys():
            item = sample[key]
            item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
            if key == 'image':
                item = F.interpolate(item, size=resize_shape,mode='trilinear', align_corners=False)
            else:
                item = F.interpolate(item, size=resize_shape, mode="nearest")
            item = item.squeeze().numpy()
            ret_dict[key] = item

        return ret_dict


class RandomCrop(AbstractTransform):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, **data_dict):
        image, label = data_dict['image'], data_dict['label']
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label = label[np.newaxis, np.newaxis, ...]
        image = image[np.newaxis, np.newaxis, ...]
        return {'image': image, 'label': label}



class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}



class ElasticTransform(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg", p_per_sample=1.0, z_last=False):
        self.opt = A.augmentations.geometric.transforms.ElasticTransform(value=0, mask_value=0)
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.z_last = z_last

    def __call__(self, **data_dict):
        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        params = self.opt.get_params()

        if self.z_last:
            feature = feature.transpose((0, 1, 3, 4, 2))
            target = target.transpose((0, 1, 3, 4, 2))

        for batch_index in range(feature.shape[0]):
            if random.random() < self.p_per_sample:
                for channel_index in range(feature.shape[1]):
                    for z_index in range(feature.shape[2]):
                        feature[batch_index, channel_index, z_index] = self.opt.apply(
                            feature[batch_index, channel_index, z_index], **params
                        )
                        if channel_index == 0:
                            target[batch_index, channel_index, z_index] = self.opt.apply_to_mask(
                                target[batch_index, channel_index, z_index], **params
                            )

        if self.z_last:
            feature = feature.transpose((0, 1, 4, 2, 3))
            target = target.transpose((0, 1, 4, 2, 3))

        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict


class OpticalDistortion(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg", p_per_sample=1.0):
        self.opt = A.augmentations.geometric.transforms.OpticalDistortion(value=0, mask_value=0)
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample


    def __call__(self, **data_dict):

        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        params = self.opt.get_params()

        for batch_index in range(feature.shape[0]):
            if random.random() < self.p_per_sample:

                for channel_index in range(feature.shape[1]):
                    for z_index in range(feature.shape[2]):
                        feature[batch_index, channel_index, z_index] = self.opt.apply(
                            feature[batch_index, channel_index, z_index], **params
                        )
                        if channel_index == 0:
                            target[batch_index, channel_index, z_index] = self.opt.apply_to_mask(
                                target[batch_index, channel_index, z_index], **params
                            )

        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict




class GridDistortion(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg", p_per_sample=1.0, z_last=1):
        self.opt = A.GridDistortion(value=0, mask_value=0)
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.z_last = z_last


    def __call__(self, **data_dict):

        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        transform = A.GridDistortion(distort_limit=(0.2, 0.4),border_mode=cv2.BORDER_CONSTANT, p=1)
        # print(feature.shape)

        if self.z_last:
            feature = feature.transpose((0, 1, 3, 4, 2))
            target = target.transpose((0, 1, 3, 4, 2))


        for batch_index in range(feature.shape[0]):
            if random.random() < self.p_per_sample:
                for channel_index in range(feature.shape[1]):
                    for z_index in range(feature.shape[2]):
                        feature[batch_index, channel_index, z_index] = transform(image=feature[batch_index, channel_index, z_index]
                        )['image']
                        if channel_index == 0:
                            target[batch_index, channel_index, z_index] = transform(image=target[batch_index, channel_index, z_index]
                            )['image']

        if self.z_last:
            feature = feature.transpose((0, 1, 4, 2, 3))
            target = target.transpose((0, 1, 4, 2, 3))

        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict


class SharpenTransform(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg", p_per_sample=1.0, z_last=False):
        self.opt = A.augmentations.transforms.Sharpen(alpha=(0.2, 0.8), p=1)
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.z_last = z_last

    def __call__(self, **data_dict):
        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        params = self.opt.get_params()
        if self.z_last:
            feature = feature.transpose((0, 1, 3, 4, 2))
            target = target.transpose((0, 1, 3, 4, 2))

        for batch_index in range(feature.shape[0]):
            if random.random() < self.p_per_sample:
                for channel_index in range(feature.shape[1]):
                    for z_index in range(feature.shape[2]):
                        feature[batch_index, channel_index, z_index] = self.opt.apply(
                            feature[batch_index, channel_index, z_index], **params
                        )

        if self.z_last:
            feature = feature.transpose((0, 1, 4, 2, 3))
            target = target.transpose((0, 1, 4, 2, 3))
        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict



class MixUpTransform(AbstractTransform):
    """
    MixUp augmentation for segmentation: blends two samples with a random lambda coefficient.
    Applies to both image and label (one-hot encoded labels).
    """
    def __init__(self, data_key="data", seg_key="seg", alpha=1.0, p_per_sample=0.5):
        self.data_key = data_key
        self.seg_key = seg_key
        self.alpha = alpha
        self.p_per_sample = p_per_sample
        self._cache = []

    def __call__(self, **data_dict):
        """
        Apply MixUp by blending current sample with a cached sample.
        Cache is built incrementally during the first pass through the batch.
        """
        data = data_dict[self.data_key]
        seg = data_dict[self.seg_key]

        # Decide whether to apply MixUp
        if random.random() >= self.p_per_sample or len(self._cache) == 0:
            # Store current sample in cache for future mixing
            self._cache.append((data.copy(), seg.copy()))
            if len(self._cache) > 100:  # Limit cache size
                self._cache.pop(0)
            return data_dict

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # Pick a random sample from cache
        data2, seg2 = random.choice(self._cache)

        # Ensure shapes match (handle potential size mismatches)
        if data.shape != data2.shape or seg.shape != seg2.shape:
            self._cache.append((data.copy(), seg.copy()))
            if len(self._cache) > 100:
                self._cache.pop(0)
            return data_dict

        # Blend images and labels
        mixed_data = lam * data + (1 - lam) * data2
        mixed_seg = lam * seg + (1 - lam) * seg2

        # Update cache with current sample
        self._cache.append((data.copy(), seg.copy()))
        if len(self._cache) > 100:
            self._cache.pop(0)

        data_dict[self.data_key] = mixed_data
        data_dict[self.seg_key] = mixed_seg
        return data_dict


class CutMixTransform(AbstractTransform):
    """
    CutMix augmentation for 3D segmentation: cuts and pastes a rectangular region from another sample.
    Preserves spatial structure better than MixUp by avoiding global blending.
    """
    def __init__(self, data_key="data", seg_key="seg", alpha=1.0, p_per_sample=0.5):
        self.data_key = data_key
        self.seg_key = seg_key
        self.alpha = alpha
        self.p_per_sample = p_per_sample
        self._cache = []

    def _rand_bbox(self, shape, lam):
        """Generate random bounding box coordinates based on lambda."""
        # shape: (batch, channel, D, H, W) or (channel, D, H, W)
        spatial_dims = shape[-3:]  # (D, H, W)
        D, H, W = spatial_dims
        
        # Compute cut ratio from lambda
        cut_ratio = np.sqrt(1.0 - lam)
        cut_d = int(D * cut_ratio)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        # Random center point
        cd = np.random.randint(D) if D > 1 else 0
        ch = np.random.randint(H) if H > 1 else 0
        cw = np.random.randint(W) if W > 1 else 0
        
        # Bounding box coordinates
        d1 = np.clip(cd - cut_d // 2, 0, D)
        d2 = np.clip(cd + cut_d // 2, 0, D)
        h1 = np.clip(ch - cut_h // 2, 0, H)
        h2 = np.clip(ch + cut_h // 2, 0, H)
        w1 = np.clip(cw - cut_w // 2, 0, W)
        w2 = np.clip(cw + cut_w // 2, 0, W)
        
        return d1, d2, h1, h2, w1, w2

    def __call__(self, **data_dict):
        """
        Apply CutMix by cutting and pasting a region from a cached sample.
        """
        data = data_dict[self.data_key]
        seg = data_dict[self.seg_key]

        # Decide whether to apply CutMix
        if random.random() >= self.p_per_sample or len(self._cache) == 0:
            # Store current sample in cache for future mixing
            self._cache.append((data.copy(), seg.copy()))
            if len(self._cache) > 100:  # Limit cache size
                self._cache.pop(0)
            return data_dict

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # Pick a random sample from cache
        data2, seg2 = random.choice(self._cache)

        # Ensure shapes match
        if data.shape != data2.shape or seg.shape != seg2.shape:
            self._cache.append((data.copy(), seg.copy()))
            if len(self._cache) > 100:
                self._cache.pop(0)
            return data_dict

        # Generate bounding box
        d1, d2, h1, h2, w1, w2 = self._rand_bbox(data.shape, lam)

        # Apply CutMix: replace the region with data2
        mixed_data = data.copy()
        mixed_seg = seg.copy()
        
        if len(data.shape) == 5:  # (batch, channel, D, H, W)
            mixed_data[:, :, d1:d2, h1:h2, w1:w2] = data2[:, :, d1:d2, h1:h2, w1:w2]
            mixed_seg[:, :, d1:d2, h1:h2, w1:w2] = seg2[:, :, d1:d2, h1:h2, w1:w2]
        elif len(data.shape) == 4:  # (channel, D, H, W)
            mixed_data[:, d1:d2, h1:h2, w1:w2] = data2[:, d1:d2, h1:h2, w1:w2]
            mixed_seg[:, d1:d2, h1:h2, w1:w2] = seg2[:, d1:d2, h1:h2, w1:w2]

        # Update cache with current sample
        self._cache.append((data.copy(), seg.copy()))
        if len(self._cache) > 100:
            self._cache.pop(0)

        data_dict[self.data_key] = mixed_data
        data_dict[self.seg_key] = mixed_seg
        return data_dict


def get_StrongAug(patch_size, sample_num, p_per_sample=0.3, mixup_alpha=0.0, mixup_prob=0.0, cutmix_alpha=0.0, cutmix_prob=0.0):
    tr_transforms = []
    tr_transforms_select = []
    tr_transforms.append(RandomCrop(patch_size))
    tr_transforms.append(RenameTransform('image', 'data', True))
    tr_transforms.append(RenameTransform('label', 'seg', True))

    # ========== Spatial-level Transforms =================
    tr_transforms_select.append(RotationTransform(p_per_sample=p_per_sample))
    tr_transforms_select.append(ScaleTransform(scale=(0.7, 1.0) , p_per_sample=p_per_sample))

    # ========== Pixel-level Transforms =================
    # tr_transforms_select.append(GaussianBlurTransform((0.7, 1.3), p_per_sample=p_per_sample))
    # tr_transforms_select.append(BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3), p_per_sample=p_per_sample))
    # tr_transforms_select.append(ContrastAugmentationTransform(contrast_range=(0.7, 1.3), p_per_sample=p_per_sample))
    # tr_transforms_select.append(GammaTransform(gamma_range=(0.7, 1.3), invert_image=False, per_channel=True, retain_stats=True, p_per_sample=p_per_sample))  # inverted gamma

    # ========== MixUp Augmentation =================
    if mixup_alpha > 0 and mixup_prob > 0:
        tr_transforms_select.append(MixUpTransform(alpha=mixup_alpha, p_per_sample=mixup_prob))

    # ========== CutMix Augmentation =================
    if cutmix_alpha > 0 and cutmix_prob > 0:
        tr_transforms_select.append(CutMixTransform(alpha=cutmix_alpha, p_per_sample=cutmix_prob))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(RenameTransform('seg', 'label', True))
    tr_transforms.append(RenameTransform('data', 'image', True))
    tr_transforms.append(NumpyToTensor(['image', 'label'], 'float'))
    trivialAug = RandomSelect(tr_transforms_select, sample_num)
    trivialAug.update_list(tr_transforms)
    return trivialAug



