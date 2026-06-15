from __future__ import annotations

import tempfile

import numpy as np
import torch

from pmt_sysu.config.defaults import Config
from pmt_sysu.data.sampler import PMTIdentitySampler, assert_pmt_batch_layout, build_label_positions
from pmt_sysu.engine.trainer import compute_pmt_losses
from pmt_sysu.losses import DCL, MSEL, TripletLoss
from pmt_sysu.model import build_pmt_model
from pmt_sysu.utils.checkpoint import load_model_weights


def tiny_config():
    return Config(
        {
            "data": Config({"height": 32, "width": 16, "batch_size_per_modality": 8, "num_pos": 2}),
            "model": Config(
                {
                    "num_classes": 4,
                    "embed_dim": 32,
                    "patch_size": [8, 8],
                    "stride_size": [8, 8],
                    "depth": 1,
                    "num_heads": 4,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "attention_dropout": 0.0,
                    "drop_path": 0.0,
                }
            ),
            "train": Config({"progressive_epoch": 6, "triplet_margin": 0.1, "msel_weight": 0.5, "dcl_weight": 0.5}),
        }
    )


def aligned_labels(batch_size=8, num_pos=2):
    labels = torch.arange(batch_size // num_pos).repeat_interleave(num_pos)
    return labels.clone(), labels.clone()


def test_sampler_layout():
    labels = np.repeat(np.arange(8), 4)
    pos = build_label_positions(labels)
    sampler = PMTIdentitySampler(labels, labels, pos, pos, batch_size=8, num_pos=2)
    lv = torch.tensor(labels[sampler.index1[:8]])
    li = torch.tensor(labels[sampler.index2[:8]])
    assert_pmt_batch_layout(lv, li, num_pos=2, batch_size=8)


def test_model_output_shape():
    config = tiny_config()
    model = build_pmt_model(config)
    model.train()
    out = model(torch.randn(16, 3, 32, 16), return_dict=True)
    assert out["features"].shape == (16, 32)
    assert out["logits"].shape == (16, 4)
    model.eval()
    emb = model(torch.randn(16, 3, 32, 16))
    assert emb.shape == (16, 32)


def test_msel_finite():
    labels, labels_ir = aligned_labels()
    targets = torch.cat([labels, labels_ir])
    features = torch.randn(16, 32)
    loss = MSEL(num_pos=2, feat_norm="no")(features, targets)
    assert torch.isfinite(loss)


def test_dcl_finite():
    labels, labels_ir = aligned_labels()
    targets = torch.cat([labels, labels_ir])
    features = torch.randn(16, 32)
    loss = DCL(num_pos=2, feat_norm="no")(features, targets)
    assert torch.isfinite(loss)


def test_gray_stage_loss():
    config = tiny_config()
    model = build_pmt_model(config)
    labels, labels_ir = aligned_labels()
    batch = (torch.randn(8, 3, 32, 16), torch.randn(8, 3, 32, 16), labels, labels_ir)
    out = compute_pmt_losses(
        config,
        model,
        batch,
        "cpu",
        1,
        torch.nn.CrossEntropyLoss(),
        TripletLoss(0.1, feat_norm="no"),
        MSEL(2, feat_norm="no"),
        DCL(2, feat_norm="no"),
    )
    assert out["stage"] == "gray_ir"
    assert out["msel_loss"].item() == 0
    assert out["dcl_loss"].item() == 0
    assert torch.isfinite(out["loss"])


def test_rgb_stage_loss():
    config = tiny_config()
    model = build_pmt_model(config)
    labels, labels_ir = aligned_labels()
    batch = (torch.randn(8, 3, 32, 16), torch.randn(8, 3, 32, 16), labels, labels_ir)
    out = compute_pmt_losses(
        config,
        model,
        batch,
        "cpu",
        7,
        torch.nn.CrossEntropyLoss(),
        TripletLoss(0.1, feat_norm="no"),
        MSEL(2, feat_norm="no"),
        DCL(2, feat_norm="no"),
    )
    assert out["stage"] == "rgb_ir"
    assert torch.isfinite(out["msel_loss"])
    assert torch.isfinite(out["dcl_loss"])
    assert torch.isfinite(out["loss"])


def test_official_checkpoint_key_conversion():
    config = tiny_config()
    model = build_pmt_model(config)
    state = {"model_state_dict": {f"module.{k}": v.detach().clone() for k, v in model.state_dict().items()}}
    with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
        torch.save(state, handle.name)
        result = load_model_weights(model, handle.name, strict=False)
    assert len(result.unexpected_keys) == 0

