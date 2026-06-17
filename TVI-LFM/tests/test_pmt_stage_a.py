import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from network.pmt_vit_adapter import PMTViTVisual
from tools.utils import load_train_configs


def test_stage_a_config_merges_and_resolves_pmt_checkpoint_path():
    config = load_train_configs("config/stage_a/pmt_vit_stage_a.yaml")

    assert config.pretrain_choice == "PMT_VIT"
    assert config.training_mode == "RGB_IR"
    assert config.loss_names == "wrt,id"
    assert config.freeze_text_in_image_only is True
    assert config.llm_aug is False
    assert config.Feat_Filter is False
    assert config.pmt_pretrained.endswith("jx_vit_base_p16_224-80ecf9dd.pth")


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
