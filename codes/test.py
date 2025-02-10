import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import torch

import options.options as option
import utils.util as util
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

#metrics
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import hausdorff_distance as hd
import lpips
import piq

from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    max_psnr=0
    min_psnr=100000
    sum_psnr=0

    max_ssim=0
    min_ssim=100000
    sum_ssim=0

    max_nmi=0
    min_nmi=100000
    sum_nmi=0

    max_hd=0
    min_hd=100000
    sum_hd=0

    max_nrmse=0
    min_nrmse=100000
    sum_nrmse=0

    max_lpips=0
    min_lpips=100000
    sum_lpips=0

    max_brisque=0
    min_brisque=100000
    sum_brisque=0

    max_fsim=0
    min_fsim=100000
    sum_fsim=0

    max_hpsi=0
    min_hpsi=100000
    sum_hpsi=0
    j=0

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        model.feed_data(data, need_GT=need_GT)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals(need_GT=need_GT)

        sr_img = util.tensor2img(visuals['rlt'])  # uint8

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)

        # calculate PSNR and SSIM
        if need_GT:
            gt_img = util.tensor2img(visuals['GT'])
            sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
            target_height = min(sr_img.shape[0], gt_img.shape[0])  # Smaller height
            target_width = min(sr_img.shape[1], gt_img.shape[1])   # Smaller width
            sr_img_pil = Image.fromarray(sr_img)
            gt_img_pil = Image.fromarray(gt_img)
            sr_img = sr_img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            gt_img = gt_img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            sr_img = np.array(sr_img)
            gt_img = np.array(gt_img)
            output_file = "output_{}.pdf".format(test_loader.dataset.opt['name'])
            combined_img = np.concatenate((sr_img, gt_img), axis=1)
            if not os.path.exists(output_file):
                # Create a new PDF file if it doesn't exist
                plt.imsave("temp_image.png", combined_img)  # Save image temporarily
                img_pil = Image.open("temp_image.png")
                img_pil.save(output_file, save_all=True)
            else:
                # Append to the existing PDF
                from PyPDF2 import PdfMerger
                plt.imsave("temp_image.png", combined_img)  # Save image temporarily
                img_pil = Image.open("temp_image.png")
                img_pil.save("temp_image.pdf")

                # Merge with existing file
                merger = PdfMerger()
                merger.append(output_file)
                merger.append("temp_image.pdf")
                merger.write(output_file)
                merger.close()

            # Clean up the temporary files
            os.remove("temp_image.png")
            if os.path.exists("temp_image.pdf"):
                os.remove("temp_image.pdf")  
            psnr = util.calculate_psnr(sr_img, gt_img)
            max_psnr= max(max_psnr,psnr)
            min_psnr= min(min_psnr,psnr)
            sum_psnr+=psnr
            avg_psnr=sum_psnr/(j+1)
            ssim = util.calculate_ssim(sr_img, gt_img)
            max_ssim= max(max_ssim,ssim)
            min_ssim= min(min_ssim,ssim)
            sum_ssim+=ssim
            avg_ssim=sum_ssim/(j+1)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)

            # other metrics
            i1=sr_img
            i2=gt_img

            nmi_score_value = nmi(i2, i1)   #nmi
            nrmse_score_value = nrmse(i2, i1)  #nrmse
            hd_score_value = hd(i2, i1)

            max_nmi= max(max_nmi,nmi_score_value)
            min_nmi= min(min_nmi,nmi_score_value)
            sum_nmi+=nmi_score_value
            avg_nmi=sum_nmi/(j+1)

            max_nrmse= max(max_nrmse,nrmse_score_value)
            min_nrmse= min(min_nrmse,nrmse_score_value)
            sum_nrmse+=nrmse_score_value
            avg_nrmse=sum_nrmse/(j+1)

            max_hd= max(max_hd,hd_score_value)
            min_hd= min(min_hd,hd_score_value)
            sum_hd+=hd_score_value
            avg_hd=sum_hd/(j+1)

            i1= torch.tensor(i1).permute(2, 0, 1).unsqueeze(0).float()
            i2= torch.tensor(i2).permute(2, 0, 1).unsqueeze(0).float()

            i1 = i1 / 255
            i2 = i2 / 355  

            loss_fn_alex = lpips.LPIPS(net='alex')
            score = loss_fn_alex(i1, i2).item()
            max_lpips= max(max_lpips,score)
            min_lpips= min(min_lpips,score)
            sum_lpips+=score
            avg_lpips=sum_lpips/(j+1)

            score_b=piq.brisque(i1)
            score_h=piq.haarpsi(i1,i2)
            score_f=piq.fsim(i1,i2)

            max_hpsi= max(max_hpsi,score_h)
            min_hpsi= min(min_hpsi,score_h)
            sum_hpsi+=score_h
            avg_hpsi=sum_hpsi/(j+1)

            max_brisque= max(max_brisque,score_b)
            min_brisque= min(min_brisque,score_b)
            sum_brisque+=score_b
            avg_brisque=sum_brisque/(j+1)

            max_fsim= max(max_fsim,score_f)
            min_fsim= min(min_fsim,score_f)
            sum_fsim+=score_f
            avg_fsim=sum_fsim/(j+1)

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img / 255., only_y=True)
                gt_img_y = bgr2ycbcr(gt_img / 255., only_y=True)

                psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                logger.info(
                    '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                    format(img_name, psnr, ssim, psnr_y, ssim_y))
            else:
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
        else:
            logger.info(img_name)
        print('Till Iteration {}\nPSNR: Max: {:.3f} Min: {:.3f} Avg: {:.3f}\nSSIM: Max: {:.3f} Min: {:.3f} Avg: {:.3f}\n'.format(j+1,max_psnr,min_psnr,avg_psnr,max_ssim,min_ssim,avg_ssim))
        print('NMI: Max: {:.3f} Min: {:.3f} Avg: {:.3f}\nNRMSE: Max: {:.3f} Min: {:.3f} Avg: {:.3f}\nHausdorff Distance: Max: {:.3f} Min: {:.3f} Avg: {:.3f}\n'.format(max_nmi,min_nmi,avg_nmi,max_nrmse,min_nrmse,avg_nrmse,max_hd,min_hd,avg_hd))
        print('LPIPS: Max: {:.3f} Min: {:.3f} Avg: {:.3f}\nBRISQUE: Max: {:.3f} Min: {:.3f} Avg: {:.3f}\nFSIM: Max: {:.3f} Min: {:.3f} Avg: {:.3f}\nHPSI: Max: {:.3f} Min: {:.3f} Avg: {:.3f}\n'.format(max_lpips,min_lpips,avg_lpips,max_brisque,min_brisque,avg_brisque,max_fsim,min_fsim,avg_fsim,max_hpsi,min_hpsi,avg_hpsi))
        j += 1

    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info(
            '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
                test_set_name, ave_psnr, ave_ssim))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info(
                '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
                format(ave_psnr_y, ave_ssim_y))
