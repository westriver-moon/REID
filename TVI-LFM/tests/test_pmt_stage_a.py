import os
import sys

import numpy as np
import torch
from PIL import Image


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from network.pmt_vit_adapter import PMTViTVisual
from data_loader.dataset import Test_Tri_Data
from solver.lr_scheduler import LRSchedulerWithWarmup
from tools import PMTMSEL, PMTDCL
from tools.utils import load_train_configs


def test_stage_a_config_merges_and_resolves_pmt_checkpoint_path():
    config = load_train_configs("config/stage_a/pmt_vit_stage_a.yaml")

    assert config.pretrain_choice == "PMT_VIT"
    assert config.training_mode == "RGB_IR"
    assert config.loss_names == "wrt,id"
    assert config.freeze_text_in_image_only is True
    assert config.joint_mode == "image_only"
    assert config.sysu_data_path.endswith("/")
    assert config.clip_download_root == "~/.cache/clip"
    assert config.llm_aug is False
    assert config.Feat_Filter is False
    assert config.pmt_pretrained.endswith("jx_vit_base_p16_224-80ecf9dd.pth")


def test_stage_a_control_config_is_image_only_and_uses_valid_sysu_dir():
    config = load_train_configs("config/stage_a/rn50_ori_stage_a_control.yaml")

    assert config.pretrain_choice == "RN50_ORI"
    assert config.training_mode == "RGB_IR"
    assert config.joint_mode == "image_only"
    assert config.sysu_data_path.endswith("/")


def test_stage_a_pmt_recipe_config_uses_pmt_training_recipe():
    config = load_train_configs("config/stage_a/pmt_vit_stage_a_pmt_recipe.yaml")

    assert config.pretrain_choice == "PMT_VIT"
    assert config.training_mode == "RGB_IR"
    assert config.joint_mode == "image_only"
    assert config.loss_names == "pmt_recipe"
    assert config.pmt_recipe is True
    assert config.pmt_recipe_transforms is True
    assert config.pmt_progressive_epoch == 6
    assert config.pmt_msel_weight == 0.5
    assert config.pmt_dcl_weight == 0.5
    assert config.optimizer == "AdamW"
    assert config.lrscheduler == "cosine"
    assert config.target_lr_factor == 0.01
    assert config.total_train_epoch == 24
    assert config.img_size == [256, 128]


def test_pmt_visual_is_single_branch_and_returns_token_outputs():
    model = PMTViTVisual(
        input_resolution=(288, 144),
        patch_size=(16, 16),
        stride_size=(12, 12),
        embed_dim=32,
        depth=1,
        num_heads=4,
        output_dim=64,
    )
    model.eval()

    assert hasattr(model, "vit")
    assert not hasattr(model, "rgb_vit")
    assert not hasattr(model, "ir_vit")

    dummy = torch.zeros(2, 3, 288, 144)
    with torch.no_grad():
        out_default = model(dummy)
        out_rgb_mode = model(dummy, mode="rgb")
        out_ir_mode = model(dummy, mode="ir")

    assert out_default["tokens"].shape == (2, 254, 64)
    assert out_default["features"].shape == (2, 64)
    assert out_default["raw_tokens"].shape == (2, 254, 32)
    assert torch.equal(out_default["features"], out_rgb_mode["features"])
    assert torch.equal(out_default["features"], out_ir_mode["features"])


def test_pmt_recipe_losses_accept_aligned_visible_ir_layout():
    labels_one_modality = torch.arange(4).repeat_interleave(2)
    labels = torch.cat([labels_one_modality, labels_one_modality])
    features = torch.randn(labels.numel(), 32)

    msel = PMTMSEL(num_pos=2, feat_norm="no")
    dcl = PMTDCL(num_pos=2, feat_norm="no")

    assert torch.isfinite(msel(features, labels))
    assert torch.isfinite(dcl(features, labels))


def test_cosine_scheduler_supports_per_group_target_lr_factor():
    linear = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(
        [
            {"params": [linear.weight], "lr": 0.1},
            {"params": [linear.bias], "lr": 0.01},
        ],
        lr=0.1,
    )
    scheduler = LRSchedulerWithWarmup(
        optimizer,
        milestones=(40, 60, 100),
        mode="cosine",
        warmup_epochs=1,
        total_epochs=3,
        target_lr_factor=0.01,
    )

    scheduler.last_epoch = 3
    lrs = scheduler.get_lr()
    assert abs(lrs[0] - 0.001) < 1e-12
    assert abs(lrs[1] - 0.0001) < 1e-12


def test_ir_only_test_dataset_does_not_require_text_assets(tmp_path):
    img_path = tmp_path / "query.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

    dataset = Test_Tri_Data(
        [str(img_path)],
        np.array([1]),
        data_path=str(tmp_path) + os.sep,
        transform=lambda image: image,
        load_text=False,
    )
    sample = dataset[0]

    assert "img" in sample
    assert "target" in sample
    assert "text" not in sample
    assert "text_filter" not in sample
