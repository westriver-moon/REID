import os

import numpy as np
import torch
from torch.autograd import Variable

from tools import eval_llcm, eval_regdb, eval_sysu

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _need_text(config):
    if 'Text' in config.test_modality:
        return True
    if 'Fusion' in config.test_modality and 'Text' in getattr(config, 'training_mode', ''):
        return True
    return False


def _eval_max_batches(config):
    return max(int(getattr(config, 'eval_max_batches', 0) or 0), 0)


def _eval_num_trials(config, default_trials):
    max_trials = max(int(getattr(config, 'eval_max_trials', 0) or 0), 0)
    return min(default_trials, max_trials) if max_trials else default_trials


def _iter_eval_batches(dataloader, config):
    limit = _eval_max_batches(config)
    for batch_idx, batch_dict in enumerate(dataloader):
        yield batch_idx, batch_dict
        if limit and batch_idx + 1 >= limit:
            break


def _extract_eval_feat(base, feat_map, config, mode='RGB', use_backup=False):
    if use_backup:
        feat = base.backup_pool(feat_map).squeeze()
        return base.backup_classifier(feat, mode)
    feat = base.base_model.visual.__getattr__(config.pooling)(feat_map).squeeze()
    return base.classifier(feat, mode)


def _log_eval_limits(config):
    batch_limit = _eval_max_batches(config)
    trial_limit = int(getattr(config, 'eval_max_trials', 0) or 0)
    if batch_limit or trial_limit:
        print(f"Eval smoke enabled: max_batches={batch_limit or 'full'}, max_trials={trial_limit or 'full'}")


def _extract_query_features(base, query_loader, config, device, embed_dim, image_mode='ir'):
    query_size = len(query_loader.dataset)
    ptr = 0
    outputs = {}

    if 'IR' in config.test_modality:
        outputs['IR'] = np.zeros((query_size, embed_dim))
    if 'Fusion' in config.test_modality:
        outputs['Fusion'] = np.zeros((query_size, embed_dim))
        if config.CAT_EVAL:
            outputs['IR_CAT'] = np.zeros((query_size, embed_dim))
            outputs['TEXT_CAT'] = np.zeros((query_size, embed_dim))
    if 'Text' in config.test_modality:
        outputs['Text'] = np.zeros((query_size, embed_dim))

    with torch.no_grad():
        for _, batch_dict in _iter_eval_batches(query_loader, config):
            input_tensor = Variable(batch_dict['img'].to(device))
            batch_num = input_tensor.size(0)
            text = None

            if _need_text(config):
                text = batch_dict['text'].to(device).long()

            if 'IR' in config.test_modality:
                feat_ir_map = base.encode_image_featmap(input_tensor, image_mode)
                feat_ir = _extract_eval_feat(base, feat_ir_map, config, mode='IR', use_backup=config.Fix_Visual)
                outputs['IR'][ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()

            if 'Fusion' in config.test_modality:
                if _need_text(config):
                    if config.Feat_Filter:
                        text_filter = batch_dict['text_filter'].to(device).long()
                        feat_fusion = base.encode_filtered_fusion(text, text_filter, input_tensor)
                    else:
                        feat_fusion = base.encode_fusion(text, input_tensor, image_mode)
                else:
                    # RGB_IR setting: use image-only fusion branch to avoid text annotation dependency during eval.
                    feat_fusion = base.encode_image_feat(input_tensor, image_mode)

                feat_fusion = base.classifier(feat_fusion, 'Fusion')
                outputs['Fusion'][ptr:ptr + batch_num, :] = feat_fusion.detach().cpu().numpy()

                if config.CAT_EVAL and _need_text(config):
                    feat_text = base.encode_text_feat(text)
                    feat_text = base.classifier(feat_text, 'Text')
                    outputs['TEXT_CAT'][ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()

                    feat_ir = base.encode_image_feat(input_tensor, image_mode)
                    feat_ir = base.classifier(feat_ir, 'IR')
                    outputs['IR_CAT'][ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()

            if 'Text' in config.test_modality:
                feat_text = base.encode_text_feat(text)
                feat_text = base.classifier(feat_text, 'Text')
                outputs['Text'][ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()

            ptr += batch_num

    outputs = {key: value[:ptr] for key, value in outputs.items()}
    return outputs, ptr


def _extract_gallery_features(base, gall_loader, config, device, embed_dim):
    gallery_size = len(gall_loader.dataset)
    ptr = 0
    outputs = {}

    if 'IR' in config.test_modality and config.Fix_Visual:
        outputs['IR_FIX'] = np.zeros((gallery_size, embed_dim))
    if 'IR' in config.test_modality or 'Fusion' in config.test_modality or 'Text' in config.test_modality:
        outputs['Main'] = np.zeros((gallery_size, embed_dim))

    with torch.no_grad():
        for _, batch_dict in _iter_eval_batches(gall_loader, config):
            input_tensor = Variable(batch_dict['img'].to(device))
            batch_num = input_tensor.size(0)
            feat_map = base.encode_image_featmap(input_tensor, 'rgb')

            if 'IR' in config.test_modality and config.Fix_Visual:
                feat_for_ir = _extract_eval_feat(base, feat_map, config, mode='RGB', use_backup=True)
                outputs['IR_FIX'][ptr:ptr + batch_num, :] = feat_for_ir.detach().cpu().numpy()

            if 'IR' in config.test_modality or 'Text' in config.test_modality or 'Fusion' in config.test_modality:
                feat = _extract_eval_feat(base, feat_map, config, mode='RGB', use_backup=False)
                outputs['Main'][ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            else:
                raise ValueError('Error: test_modality not found!')

            ptr += batch_num

    outputs = {key: value[:ptr] for key, value in outputs.items()}
    return outputs, ptr


def test(base, loader, config, device):
    embed_dim = base.embed_dim
    base.set_eval()
    print('Extracting Query Feature...')
    print('Test Mode: ', config.test_modality)
    _log_eval_limits(config)
    result_dict = dict()

    assert loader.dataset in ['sysu', 'regdb', 'llcm'], 'Invalid dataset!'
    assert 'IR' in config.test_modality or 'Fusion' in config.test_modality or 'Text' in config.test_modality, 'Invalid test modality!'

    if loader.dataset == 'regdb':
        query_feat_ir_list = []
        query_feat_fusion_list = []
        query_feat_text_list = []
        query_feat_ir_cat_list = []
        query_feat_text_cat_list = []
        query_count_list = []
        num_trials = _eval_num_trials(config, len(loader.query_loaders))
        for i in range(num_trials):
            query_loader = loader.query_loaders[i]
            query_outputs, query_count = _extract_query_features(base, query_loader, config, device, embed_dim, image_mode='ir')
            query_count_list.append(query_count)
            print(f"[eval] regdb query trial {i + 1}: {query_count}/{len(query_loader.dataset)} samples")
            if 'IR' in config.test_modality:
                query_feat_ir_list.append(query_outputs['IR'])
            if 'Fusion' in config.test_modality:
                query_feat_fusion_list.append(query_outputs['Fusion'])
                if config.CAT_EVAL:
                    query_feat_ir_cat_list.append(query_outputs['IR_CAT'])
                    query_feat_text_cat_list.append(query_outputs['TEXT_CAT'])
            if 'Text' in config.test_modality:
                query_feat_text_list.append(query_outputs['Text'])
    else:
        query_outputs, query_count = _extract_query_features(base, loader.query_loader, config, device, embed_dim, image_mode='ir')
        print(f"[eval] {loader.dataset} query: {query_count}/{len(loader.query_loader.dataset)} samples")
        if 'IR' in config.test_modality:
            query_feat_ir = query_outputs['IR']
        if 'Fusion' in config.test_modality:
            query_feat_fusion = query_outputs['Fusion']
            if config.CAT_EVAL:
                query_feat_ir_f = query_outputs['IR_CAT']
                query_feat_text_f = query_outputs['TEXT_CAT']
        if 'Text' in config.test_modality:
            query_feat_text = query_outputs['Text']

    print('Extracting Gallery Feature...')

    if loader.dataset == 'sysu':
        all_cmc_ir = 0
        all_mAP_ir = 0
        all_mINP_ir = 0
        all_cmc_fusion = 0
        all_mAP_fusion = 0
        all_mINP_fusion = 0
        all_cmc_text = 0
        all_mAP_text = 0
        all_mINP_text = 0
        num_trials = _eval_num_trials(config, len(loader.gallery_loaders))
        query_label = loader.query_loader.dataset.test_label[:query_count]
        query_cam = loader.query_cam[:query_count]
        for i in range(num_trials):
            gall_loader = loader.gallery_loaders[i]
            gall_outputs, gall_count = _extract_gallery_features(base, gall_loader, config, device, embed_dim)
            gall_label = gall_loader.dataset.test_label[:gall_count]
            gall_cam = loader.gall_cam_list[i][:gall_count]
            print(f"[eval] sysu gallery trial {i + 1}: {gall_count}/{len(gall_loader.dataset)} samples")

            if 'IR' in config.test_modality:
                if config.Fix_Visual:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_outputs['IR_FIX']))
                else:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_outputs['Main']))
                cmc_ir, mAP_ir, mINP_ir = eval_sysu(-distmat_ir, query_label, gall_label, query_cam, gall_cam)
                all_cmc_ir += cmc_ir
                all_mAP_ir += mAP_ir
                all_mINP_ir += mINP_ir
            if 'Fusion' in config.test_modality:
                if config.CAT_EVAL:
                    distmat_ir_f = np.matmul(query_feat_ir_f, np.transpose(gall_outputs['Main']))
                    distmat_text_f = np.matmul(query_feat_text_f, np.transpose(gall_outputs['Main']))
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_outputs['Main'])) + distmat_ir_f + distmat_text_f
                else:
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_outputs['Main']))
                cmc_fusion, mAP_fusion, mINP_fusion = eval_sysu(-distmat_fusion, query_label, gall_label, query_cam, gall_cam)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion
            if 'Text' in config.test_modality:
                distmat_text = np.matmul(query_feat_text, np.transpose(gall_outputs['Main']))
                cmc_text, mAP_text, mINP_text = eval_sysu(-distmat_text, query_label, gall_label, query_cam, gall_cam)
                all_cmc_text += cmc_text
                all_mAP_text += mAP_text
                all_mINP_text += mINP_text
        if 'IR' in config.test_modality:
            all_cmc_ir /= num_trials
            all_mAP_ir /= num_trials
            all_mINP_ir /= num_trials
            result_dict['IR'] = (all_mINP_ir, all_mAP_ir, all_cmc_ir)
        if 'Fusion' in config.test_modality:
            all_cmc_fusion /= num_trials
            all_mAP_fusion /= num_trials
            all_mINP_fusion /= num_trials
            result_dict['Fusion'] = (all_mINP_fusion, all_mAP_fusion, all_cmc_fusion)
        if 'Text' in config.test_modality:
            all_cmc_text /= num_trials
            all_mAP_text /= num_trials
            all_mINP_text /= num_trials
            result_dict['Text'] = (all_mINP_text, all_mAP_text, all_cmc_text)

    elif loader.dataset == 'regdb':
        all_cmc_ir = 0
        all_mAP_ir = 0
        all_mINP_ir = 0
        all_cmc_fusion = 0
        all_mAP_fusion = 0
        all_mINP_fusion = 0
        all_cmc_text = 0
        all_mAP_text = 0
        all_mINP_text = 0
        num_trials = _eval_num_trials(config, len(loader.gallery_loaders))
        for i in range(num_trials):
            gall_loader = loader.gallery_loaders[i]
            gall_outputs, gall_count = _extract_gallery_features(base, gall_loader, config, device, embed_dim)
            gall_label = gall_loader.dataset.test_label[:gall_count]
            query_label = loader.query_loaders[i].dataset.test_label[:query_count_list[i]]
            print(f"[eval] regdb gallery trial {i + 1}: {gall_count}/{len(gall_loader.dataset)} samples")

            if 'IR' in config.test_modality:
                query_feat_ir = query_feat_ir_list[i]
                if config.regdb_test_mode == 't-v':
                    if config.Fix_Visual:
                        distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_outputs['IR_FIX']))
                    else:
                        distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_outputs['Main']))
                    cmc_ir, mAP_ir, mINP_ir = eval_regdb(-distmat_ir, query_label, gall_label)
                else:
                    if config.Fix_Visual:
                        distmat_ir = np.matmul(gall_outputs['IR_FIX'], np.transpose(query_feat_ir))
                    else:
                        distmat_ir = np.matmul(gall_outputs['Main'], np.transpose(query_feat_ir))
                    cmc_ir, mAP_ir, mINP_ir = eval_regdb(-distmat_ir, gall_label, query_label)
                all_cmc_ir += cmc_ir
                all_mAP_ir += mAP_ir
                all_mINP_ir += mINP_ir

            if 'Fusion' in config.test_modality:
                query_feat_fusion = query_feat_fusion_list[i]
                if config.regdb_test_mode == 't-v':
                    if config.CAT_EVAL:
                        distmat_ir_f = np.matmul(query_feat_ir_cat_list[i], np.transpose(gall_outputs['Main']))
                        distmat_text_f = np.matmul(query_feat_text_cat_list[i], np.transpose(gall_outputs['Main']))
                        distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_outputs['Main'])) + distmat_ir_f + distmat_text_f
                    else:
                        distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_outputs['Main']))
                    cmc_fusion, mAP_fusion, mINP_fusion = eval_regdb(-distmat_fusion, query_label, gall_label)
                else:
                    distmat_fusion = np.matmul(gall_outputs['Main'], np.transpose(query_feat_fusion))
                    cmc_fusion, mAP_fusion, mINP_fusion = eval_regdb(-distmat_fusion, gall_label, query_label)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion

            if 'Text' in config.test_modality:
                query_feat_text = query_feat_text_list[i]
                if config.regdb_test_mode == 't-v':
                    distmat_text = np.matmul(query_feat_text, np.transpose(gall_outputs['Main']))
                    cmc_text, mAP_text, mINP_text = eval_regdb(-distmat_text, query_label, gall_label)
                else:
                    distmat_text = np.matmul(gall_outputs['Main'], np.transpose(query_feat_text))
                    cmc_text, mAP_text, mINP_text = eval_regdb(-distmat_text, gall_label, query_label)
                all_cmc_text += cmc_text
                all_mAP_text += mAP_text
                all_mINP_text += mINP_text

        if 'IR' in config.test_modality:
            all_cmc_ir /= num_trials
            all_mAP_ir /= num_trials
            all_mINP_ir /= num_trials
            result_dict['IR'] = (all_mINP_ir, all_mAP_ir, all_cmc_ir)
        if 'Fusion' in config.test_modality:
            all_cmc_fusion /= num_trials
            all_mAP_fusion /= num_trials
            all_mINP_fusion /= num_trials
            result_dict['Fusion'] = (all_mINP_fusion, all_mAP_fusion, all_cmc_fusion)
        if 'Text' in config.test_modality:
            all_cmc_text /= num_trials
            all_mAP_text /= num_trials
            all_mINP_text /= num_trials
            result_dict['Text'] = (all_mINP_text, all_mAP_text, all_cmc_text)

    elif loader.dataset == 'llcm':
        all_cmc_ir = 0
        all_mAP_ir = 0
        all_mINP_ir = 0
        all_cmc_fusion = 0
        all_mAP_fusion = 0
        all_mINP_fusion = 0
        all_cmc_text = 0
        all_mAP_text = 0
        all_mINP_text = 0
        num_trials = _eval_num_trials(config, len(loader.gallery_loaders))
        query_label = loader.query_loader.dataset.test_label[:query_count]
        query_cam = loader.query_cam[:query_count]
        for i in range(num_trials):
            gall_loader = loader.gallery_loaders[i]
            gall_outputs, gall_count = _extract_gallery_features(base, gall_loader, config, device, embed_dim)
            gall_label = gall_loader.dataset.test_label[:gall_count]
            gall_cam = loader.gall_cam_list[i][:gall_count]
            print(f"[eval] llcm gallery trial {i + 1}: {gall_count}/{len(gall_loader.dataset)} samples")

            if 'IR' in config.test_modality:
                if config.Fix_Visual:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_outputs['IR_FIX']))
                else:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_outputs['Main']))
                cmc_ir, mAP_ir, mINP_ir = eval_llcm(-distmat_ir, query_label, gall_label, query_cam, gall_cam)
                all_cmc_ir += cmc_ir
                all_mAP_ir += mAP_ir
                all_mINP_ir += mINP_ir

            if 'Fusion' in config.test_modality:
                if config.CAT_EVAL:
                    distmat_ir_f = np.matmul(query_feat_ir_f, np.transpose(gall_outputs['Main']))
                    distmat_text_f = np.matmul(query_feat_text_f, np.transpose(gall_outputs['Main']))
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_outputs['Main'])) + distmat_ir_f + distmat_text_f
                else:
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_outputs['Main']))
                cmc_fusion, mAP_fusion, mINP_fusion = eval_llcm(-distmat_fusion, query_label, gall_label, query_cam, gall_cam)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion
            if 'Text' in config.test_modality:
                distmat_text = np.matmul(query_feat_text, np.transpose(gall_outputs['Main']))
                cmc_text, mAP_text, mINP_text = eval_llcm(-distmat_text, query_label, gall_label, query_cam, gall_cam)
                all_cmc_text += cmc_text
                all_mAP_text += mAP_text
                all_mINP_text += mINP_text

        if 'IR' in config.test_modality:
            all_cmc_ir /= num_trials
            all_mAP_ir /= num_trials
            all_mINP_ir /= num_trials
            result_dict['IR'] = (all_mINP_ir, all_mAP_ir, all_cmc_ir)
        if 'Fusion' in config.test_modality:
            all_cmc_fusion /= num_trials
            all_mAP_fusion /= num_trials
            all_mINP_fusion /= num_trials
            result_dict['Fusion'] = (all_mINP_fusion, all_mAP_fusion, all_cmc_fusion)
        if 'Text' in config.test_modality:
            all_cmc_text /= num_trials
            all_mAP_text /= num_trials
            all_mINP_text /= num_trials
            result_dict['Text'] = (all_mINP_text, all_mAP_text, all_cmc_text)

    return result_dict
