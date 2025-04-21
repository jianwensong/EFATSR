import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
from basicsr.models.archs.EFATSR_arch import EFATSR
from einops import rearrange
from scripts.utils import *

def define_model(args):
    model = EFATSR(args.scale)
    pretrained_model = torch.load(args.model_path,map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_model['params'],strict=False)

    return model
def setup(args,dataset):
    save_dir = args.save_dir+'/x'+str(args.scale)+'/'+dataset
    folder = args.folder_gt+'/'+dataset+'/HR'
    lq_folder = args.folder_lq+'/'+dataset+'/LR_bicubic/X'+str(args.scale)
    border = args.scale
    return folder, save_dir, border,lq_folder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--model_path', type=str,default='./experiments/pretrained_models/EFATSR_x4.pth')
    parser.add_argument('--folder_lq', type=str, default='./datasets/SISR/test', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='./datasets/SISR/test', help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--self_ensemble', type=bool, default=False, help='self-ensemble')
    parser.add_argument('--save_dir', type=str, default='./Results/EFATSR')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)

    dataset_list = ['Set5','Set14','B100','Urban100','manga109'] #'Set5','Set14','B100','Urban100','manga109'

    for i in range(len(dataset_list)):
        folder, save_dir, border, lq_folder = setup(args,dataset_list[i])
        os.makedirs(save_dir, exist_ok=True)
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []
        test_results['psnrb'] = []
        test_results['psnrb_y'] = []
        psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0

        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            # read image
            imgname, img_lq, img_gt = get_image_pair(args, path, lq_folder)  # image to HWC-BGR, float32
            img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
            img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

            # inference
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.shape
                #output,mask = test(img_lq, model, args)
                output = test(img_lq, model, args)
                output = output[:,:,:int(h_old*args.scale),:int(w_old*args.scale)]

            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
            cv2.imwrite(f'{save_dir}/{imgname}.png', output) #save images

            # evaluate psnr/ssim/psnr_b
            if img_gt is not None:
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
                img_gt = np.squeeze(img_gt)

                psnr = calculate_psnr(output, img_gt, crop_border=border)
                ssim = calculate_ssim(output, img_gt, crop_border=border)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                if img_gt.ndim == 3:  # RGB image
                    psnr_y = calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                    ssim_y = calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNRB: {:.2f} dB;'
                    'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; PSNRB_Y: {:.2f} dB.'.
                    format(idx, imgname, psnr, ssim, psnrb, psnr_y, ssim_y, psnrb_y))
            else:
                print('Testing {:d} {:20s}'.format(idx, imgname))

        # summarize psnr/ssim
        if img_gt is not None:
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
            if img_gt.ndim == 3:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))

def get_image_pair(args, path,folder_lq):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq, img_gt

def get_stereo_image_pair(args, path,folder_lq):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq, img_gt

def test(img_lq, model, args):
    if args.tile is None:

        if args.self_ensemble:
            img_lqs = ensemble(img_lq)
            output = [model(x) for x in img_lqs]
            output = reverse(output)
            output = torch.stack(output,dim=0).mean(dim=0)

        else:
            output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

def ensemble(x):
    y = []
    for i in range(8):
        xs = x
        if i > 3:
            xs = _transform(xs, 't')
        if i % 4 > 1:
            xs = _transform(xs, 'h')
        if (i % 4) % 2 == 1:
            xs = _transform(xs, 'v')
        y.append(xs)
    return y

def reverse(y):
    out=[]
    for i in range(len(y)):
        xs = y[i]
        if (i % 4) % 2 == 1:
            xs = _transform(xs, 'v')
        if i % 4 > 1:
            xs = _transform(xs, 'h')
        if i > 3:
            xs = _transform(xs, 't')
        out.append(xs)
    return out
def _transform(v, op):

    v2np = v.data.cpu().numpy()
    if op == 'v':
        tfnp = v2np[:, :, :, ::-1].copy()
    elif op == 'h':
        tfnp = v2np[:, :, ::-1, :].copy()
    elif op == 't':
        tfnp = v2np.transpose((0, 1, 3, 2)).copy()

    ret = torch.Tensor(tfnp).to(v.device)
    return ret
if __name__ == '__main__':
    main()