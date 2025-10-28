import os
import math
from tqdm import tqdm
import numpy as np
import random
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.config import Config
from utils.loss import SoftDiceLoss
import numbers
import cv2
import h5py

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor.
    """
    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        return self.conv(input, weight=self.weight.cuda(), groups=self.groups, padding="same")


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def print_func(item):
    if type(item) == torch.Tensor:
        return [round(x,2) for x in item.data.cpu().numpy().tolist()]
    elif type(item) == np.ndarray:
        return [round(x,2) for x in item.tolist()]
    else:
        raise TypeError


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / (x_sum)
    return s

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr

def read_list(split, task=None):
    """Return a sorted list of data identifiers or file paths for the given split/task."""
    if split is None:
        raise ValueError("split must not be None")

    if os.path.isdir(split):
        image_dir = os.path.join(split, 'image') if os.path.isdir(os.path.join(split, 'image')) else split
        image_files = []
        for fname in os.listdir(image_dir):
            if fname.endswith('.npy') and ('_label' not in fname and '_seg' not in fname):
                image_files.append(os.path.join(image_dir, fname))
        if not image_files:
            print(f"Warning: No '.npy' files found in {image_dir}")
            return []
        return sorted(image_files)

    if task is None:
        raise ValueError("task must be provided when split is not a directory")

    config = Config(task)
    split_path = os.path.join(config.save_dir, 'split_txts', f'{split}.txt')
    if not os.path.exists(split_path):
        raise FileNotFoundError(split_path)
    ids_list = np.loadtxt(split_path, dtype=str).tolist()
    return sorted(ids_list)


def _infer_label_path_from_image(im_path):
    candidates = []
    if '/image/' in im_path:
        candidates.append(im_path.replace('/image/', '/label/'))
    base, ext = os.path.splitext(im_path)
    if base.endswith('_image'):
        candidates.append(base.replace('_image', '_label') + ext)
    candidates.append(base + '_label' + ext)

    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Unable to infer label path for image: {im_path}")


def read_data(data_id, task=None, normalize=False):
    """Read image/label pair either by identifier or direct file paths."""
    if data_id is None:
        raise ValueError("data_id must not be None")

    if os.path.isfile(data_id):
        im_path = data_id
        lb_path = _infer_label_path_from_image(im_path)
    else:
        if task is None:
            raise ValueError("task must be provided when data_id is not a file path")
        config = Config(task)
        im_path = os.path.join(config.save_dir, 'npy', f'{data_id}_image.npy')
        lb_path = os.path.join(config.save_dir, 'npy', f'{data_id}_label.npy')
        if not os.path.exists(im_path) or not os.path.exists(lb_path):
            print(im_path)
            print(lb_path)
            raise ValueError(data_id)

    if not os.path.exists(im_path):
        raise FileNotFoundError(im_path)
    if not os.path.exists(lb_path):
        raise FileNotFoundError(lb_path)

    image = np.load(im_path)
    label = np.load(lb_path)

    task_name = task or ""

    if normalize:
        if "synapse" in task_name:
            image = image.clip(min=-75, max=275)
        elif "mnms" in task_name or "mms2d" in task_name or "udamms" in task_name:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        image = image.astype(np.float32)
        denom = image.max() - image.min()
        if denom > 1e-6:
            image = (image - image.min()) / denom
        else:
            image = image - image.min()

    return image, label


def read_list_2d(split, task):
    if split is None:
        raise ValueError("split must not be None")

    if os.path.isdir(split):
        image_dir = os.path.join(split, 'image') if os.path.isdir(os.path.join(split, 'image')) else split
        image_files = []
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.npy', '.h5')) and '_label' not in fname:
                image_files.append(os.path.join(image_dir, fname))
        return sorted(image_files)

    config = Config(task+"_2d")
    split_path = os.path.join(config.save_dir, 'split_txts', f'{split}.txt')
    if not os.path.exists(split_path):
        raise FileNotFoundError(split_path)
    ids_list = np.loadtxt(split_path, dtype=str).tolist()
    return sorted(ids_list)


def read_data_2d(data_id, task, normalize=False):
    if os.path.isfile(data_id):
        im_path = data_id
        if '/image/' in im_path:
            lb_path = im_path.replace('/image/', '/label/')
        else:
            base, ext = os.path.splitext(im_path)
            lb_path = base + '_label' + ext
    else:
        config = Config(task+"_2d")
        if "acdc" in task:
            h5File = h5py.File(os.path.join(config.save_dir, 'h5', f'{data_id}.h5'), 'r')
            image = h5File["image"][:]
            label = h5File["label"][:]
            return image, label
        else:
            im_path = os.path.join(config.save_dir, 'png', f'{data_id}_image.png')
            lb_path = os.path.join(config.save_dir, 'png', f'{data_id}_label.png')

    if not os.path.exists(im_path):
        raise FileNotFoundError(im_path)
    if not os.path.exists(lb_path):
        raise FileNotFoundError(lb_path)

    if im_path.endswith('.npy'):
        image = np.load(im_path)
    else:
        image = cv2.imread(im_path, 0)

    if lb_path.endswith('.npy'):
        label = np.load(lb_path)
    else:
        label = cv2.imread(lb_path, 0)

    if normalize:
        if "synapse" in task:
            image = image.clip(min=-75, max=275)
        elif "mnms" in task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        image = image.astype(np.float32)
        denom = image.max() - image.min()
        if denom > 1e-6:
            image = (image - image.min()) / denom
        else:
            image = image - image.min()

    return image, label


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fetch_data(batch, labeled=True):
    image = batch['image'].cuda()
    if labeled:
        label = batch['label'].cuda().unsqueeze(1)
        return image, label
    else:
        return image


def test_all_case(task, net, ids_list, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    dice_func = SoftDiceLoss(smooth=1e-8, do_bg=False)
    all_sample_dice = []

    for im_path in tqdm(ids_list):
        data_id = os.path.basename(str(im_path)).replace('.npy', '')
        image, label_gt = read_data(im_path, task=task, normalize=True)

        pred_map, score_map, pseudo_label, label_gt_processed = test_single_case(
            net,
            image,
            label_gt,
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )

        pred_tensor = torch.from_numpy(pred_map.astype(np.int64)).unsqueeze(0).unsqueeze(0).cuda()
        label_gt_tensor = torch.from_numpy(label_gt_processed.astype(np.int64)).unsqueeze(0).unsqueeze(0).cuda()

        pred_one_hot = torch.zeros((1, num_classes) + pred_map.shape).cuda()
        pred_one_hot.scatter_(1, pred_tensor.long(), 1)

        label_gt_one_hot = torch.zeros((1, num_classes) + label_gt_processed.shape).cuda()
        label_gt_one_hot.scatter_(1, label_gt_tensor.long(), 1)

        sample_dice_scores = dice_func(pred_one_hot, label_gt_one_hot, is_training=False)
        avg_dice = sample_dice_scores.mean().item()
        all_sample_dice.append(avg_dice)

        print(f"Sample: {data_id}, Avg Dice: {avg_dice:.4f}, Class Dice: {[f'{d:.4f}' for d in sample_dice_scores.cpu().numpy()]}")

        if test_save_path is not None:
            out = sitk.GetImageFromArray(pred_map.astype(np.uint8))
            sitk.WriteImage(out, os.path.join(test_save_path, f'{data_id}.nii.gz'))

    if all_sample_dice:
        overall_avg_dice = np.mean(all_sample_dice)
        print(f"\nOverall Average Dice Score: {overall_avg_dice:.4f}")


def test_all_case_ensemble(task, nets, ids_list, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    if not nets:
        raise ValueError("'nets' must contain at least one model for ensembling")

    for net in nets:
        net.eval()

    dice_func = SoftDiceLoss(smooth=1e-8, do_bg=False)
    all_sample_dice = []

    for data_ref in tqdm(ids_list):
        data_id = os.path.basename(str(data_ref)).replace('.npy', '')
        image, label_gt = read_data(data_ref, task=task, normalize=True)

        score_sum = None
        label_processed = None

        for net in nets:
            _, score_map, _, label_gt_processed = test_single_case(
                net,
                np.copy(image),
                np.copy(label_gt),
                stride_xy,
                stride_z,
                patch_size,
                num_classes=num_classes
            )
            if score_sum is None:
                score_sum = score_map
                label_processed = label_gt_processed
            else:
                score_sum += score_map

        score_avg = score_sum / len(nets)
        pred_map = np.argmax(score_avg, axis=0).astype(np.int32)

        pred_tensor = torch.from_numpy(pred_map).unsqueeze(0).unsqueeze(0).cuda()
        label_tensor = torch.from_numpy(label_processed.astype(np.int64)).unsqueeze(0).unsqueeze(0).cuda()

        pred_one_hot = torch.zeros((1, num_classes) + pred_map.shape, device=pred_tensor.device)
        pred_one_hot.scatter_(1, pred_tensor.long(), 1)

        label_one_hot = torch.zeros((1, num_classes) + label_processed.shape, device=pred_tensor.device)
        label_one_hot.scatter_(1, label_tensor.long(), 1)

        sample_dice_scores = dice_func(pred_one_hot, label_one_hot, is_training=False)
        avg_dice = sample_dice_scores.mean().item()
        all_sample_dice.append(avg_dice)

        print(f"[Ensemble] Sample: {data_id}, Avg Dice: {avg_dice:.4f}, Class Dice: {[f'{d:.4f}' for d in sample_dice_scores.cpu().numpy()]}")

        if test_save_path is not None:
            maybe_mkdir(test_save_path)
            out = sitk.GetImageFromArray(pred_map.astype(np.uint8))
            sitk.WriteImage(out, os.path.join(test_save_path, f'{data_id}.nii.gz'))

    if all_sample_dice:
        overall_avg_dice = np.mean(all_sample_dice)
        print(f"\n[Ensemble] Overall Average Dice Score: {overall_avg_dice:.4f}")
    return all_sample_dice


def test_single_case(net, image, label_gt, stride_xy, stride_z, patch_size, num_classes):
    original_shape = image.shape

    padding_flag = image.shape[0] < patch_size[0] or image.shape[1] < patch_size[1] or image.shape[2] < patch_size[2]
    if padding_flag:
        pw = max((patch_size[0] - image.shape[0]) // 2 + 1, 0)
        ph = max((patch_size[1] - image.shape[1]) // 2 + 1, 0)
        pd = max((patch_size[2] - image.shape[2]) // 2 + 1, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label_gt = np.pad(label_gt, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    image = image[np.newaxis]
    _, dd, ww, hh = image.shape

    image = image.transpose(0, 3, 2, 1)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    pseudo_label_map = np.zeros(image.shape[1:4]).astype(np.float32)

    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                test_patch = test_patch.transpose(2, 4)
                
                y1 = net(test_patch, pred_type="D_theta_u")
                p_u_xi = net(test_patch, pred_type="ddim_sample")
                p_u_psi = net(test_patch, pred_type="D_psi_l")
                
                smoothing = GaussianSmoothing(num_classes, 3, 1)
                p_u_xi = smoothing(F.gumbel_softmax(p_u_xi, dim=1))
                p_u_psi = F.softmax(p_u_psi, dim=1)
                p_u_fake = net(test_patch, pred_type="fake")
                p_u_fake = F.softmax(p_u_fake, dim=1)
                pseudo_label = torch.argmax(p_u_xi + p_u_psi + p_u_fake, dim=1, keepdim=True)

                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)

                pseudo_label = pseudo_label.cpu().data.numpy()[0, 0, ...]
                pseudo_label = pseudo_label.transpose(2, 1, 0)
                pseudo_label_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += pseudo_label

                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    score_map = score_map.transpose(0, 3, 2, 1)
    pred_map = np.argmax(score_map, axis=0)

    pseudo_label_map = pseudo_label_map / cnt
    pseudo_label_map = pseudo_label_map.transpose(2, 1, 0)

    if padding_flag:
         pred_map = pred_map[pw:-pw, ph:-ph, pd:-pd]
         label_gt = label_gt[pw:-pw, ph:-ph, pd:-pd]
         score_map = score_map[:, pw:-pw, ph:-ph, pd:-pd]
         pseudo_label_map = pseudo_label_map[pw:-pw, ph:-ph, pd:-pd]

    return pred_map, score_map, pseudo_label_map, label_gt


def test_all_case_2d(task, net, ids_list, num_classes, patch_size, stride_xy, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, label = read_data(data_id, task=task+"_2d", normalize=True)
        label = label.astype(np.uint8)
        print(np.unique(label))
        pred, label = test_single_case_2d(
            net,
            image, label,
            stride_xy,
            patch_size,
            num_classes=num_classes
        )
        cv2.imwrite(f'{test_save_path}/{data_id}.png', pred/3*255)
        cv2.imwrite(f'{test_save_path}/{data_id}_label.png', label/3*255)


def test_single_case_2d(net, image, label, stride_xy, patch_size, num_classes):
    padding_flag = image.shape[0] <= patch_size[0] or image.shape[1] <= patch_size[1]
    if padding_flag:
        pw = max((patch_size[0] - image.shape[0]) // 2 + 1, 0)
        ph = max((patch_size[1] - image.shape[1]) // 2 + 1, 0)
        image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

    image = image[np.newaxis]

    _, hh, ww = image.shape

    sx = math.ceil((hh - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((ww - patch_size[1]) / stride_xy) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:3]).astype(np.float32)
    cnt = np.zeros(image.shape[1:3]).astype(np.float32)
    for x in range(sx):
        xs = min(stride_xy*x, hh-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, ww-patch_size[1])
            test_patch = image[:,  xs:xs+patch_size[0], ys:ys+patch_size[1]]
            test_patch = torch.from_numpy(test_patch).cuda().float()
            y1 = net(test_patch.unsqueeze(0), pred_type="student")
            y = F.softmax(y1, dim=1)
            y = y.cpu().data.numpy()
            y = y[0, ...]
            score_map[:,xs:xs+patch_size[0], ys:ys+patch_size[1]] += y
            cnt[xs:xs+patch_size[0], ys:ys+patch_size[1]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    return label_map, label
