# TVI-LFM Stage A Current Project Task

Date: 2026-06-17

This is the current-project version of the Stage A engineering task. It targets the freshly cloned repository at:

```text
/home/cgv841/ybj/TVI-LFM
```

Reference code and weights may be read from:

```text
/home/cgv841/ybj/PMT-SYSU
```

Do not implement this task in `Single-experiment`. That directory is the preserved old experiment workspace and may only be used as historical reference when explicitly needed.

## Goal

Stage A answers one narrow question:

```text
Can TVI-LFM train and evaluate normally if RN50_ORI is replaced by one fully shared PMT ViT-B/16 visual backbone, while keeping the original TVI-LFM RGB_IR recipe?
```

The only intended scientific variable is the visual backbone.

Original TVI-LFM visual path:

```text
RGB independent ResNet stem ----\
                                 -> shared ResNet layer1-layer4 -> visual feature
IR independent ResNet stem -----/
```

Stage A visual path:

```text
RGB original --\
RGB augmented ---> one shared PMT ViT-B/16 -> CLS feature -> projection -> TVI feature
IR image -----/
```

There must be exactly one PMT ViT instance. RGB and IR must share:

- patch embedding
- positional embedding
- all transformer blocks
- final norm
- projection head

`mode` may stay in the function signature for compatibility, but it must not select different RGB and IR parameters.

## Strict Boundary

Stage A must keep:

- `training_mode: RGB_IR`
- `loss_names: wrt,id`
- TVI-LFM `IdentitySampler`
- TVI-LFM RGB original / RGB augmented / IR batch structure
- TVI-LFM transforms and optimizer family
- TVI-LFM scheduler, warmup, epoch settings, seed, and evaluation protocol

Stage A must not add:

- PMT gray RGB progressive stage
- PMT progressive epoch schedule
- MSEL
- DCL
- PMT margin triplet
- text fusion
- image-text contrastive loss
- prototype alignment
- mean shift
- VCM / RegDB / LLCM pretraining
- official PMT SYSU checkpoint initialization
- multi-branch patch embedding
- LASTViT adapter or LASTViT regression requirements
- dual ViT towers
- new data augmentation
- PMT-specific AdamW recipe unless added later as a separate ablation

Use only ImageNet ViT-B/16 initialization for the PMT visual backbone:

```text
/home/cgv841/ybj/PMT-SYSU/pretrained/jx_vit_base_p16_224-80ecf9dd.pth
```

## Experiments

### A0 Control

Retrain the current TVI-LFM control under the same recipe:

```text
pretrain_choice: RN50_ORI
training_mode: RGB_IR
loss_names: wrt,id
```

Do not compare A1 against old historical RN50 numbers trained with different settings.

### A1 Main

Train the PMT visual replacement:

```text
pretrain_choice: PMT_VIT
training_mode: RGB_IR
loss_names: wrt,id
```

Architecture:

```text
PMT ViT-B/16 CLS 768 -> Linear projection -> 2048 dim TVI-LFM feature
```

Keep 2048 output dimension in Stage A so that classifier shape, ReID feature shape, and downstream evaluation remain aligned with `RN50_ORI`.

## PMT ViT Parameters

Use the single-branch overlapping patch ViT structure from PMT-SYSU:

```yaml
pmt_embed_dim: 768
pmt_patch_size: [16, 16]
pmt_stride_size: [12, 12]
pmt_depth: 12
pmt_num_heads: 12
pmt_mlp_ratio: 4.0
pmt_dropout: 0.03
pmt_attention_dropout: 0.0
pmt_drop_path_rate: 0.1
prj_output_dim: 2048
```

TVI-LFM currently uses input size:

```text
H x W = 288 x 144
```

For `patch_size=16` and `stride_size=12`:

```text
num_y = floor((288 - 16) / 12) + 1 = 23
num_x = floor((144 - 16) / 12) + 1 = 11
patch tokens = 23 * 11 = 253
tokens including CLS = 254
```

Expected shapes:

```text
raw_tokens       [B, 254, 768]
projected_tokens [B, 254, 2048]
features         [B, 2048]
```

## Current Repository Facts To Respect

The fresh `TVI-LFM` clone has no `project/sysumm01` tree. Put new Stage A configs under:

```text
TVI-LFM/config/stage_a/
```

Current `main.py` replaces the entire config object when `--config_select` is not `default`. Therefore one of these must be done before using short Stage A YAML files:

1. Preferred: change `tools/utils.py::load_train_configs()` to merge `config/default.yaml` with the selected YAML, where selected keys override defaults.
2. Alternative: make every Stage A YAML a full copy of all keys from `config/default.yaml`.

Use the preferred merge approach unless there is a strong reason not to. It makes Stage A configs smaller and prevents missing-key crashes.

Current visual string handling is inconsistent:

- `core/train.py` checks `"VIT"` uppercase.
- `core/build.py` checks `"ViT"` mixed case.

Stage A must add explicit helper checks for:

```text
RN50
RN50_ORI
ViT-B/16
PMT_VIT
```

Do not rely on substring case matching for `PMT_VIT`.

Current `CLIP.dtype` does not check `visual.input_dtype`. It assumes either `Dual_Resnet` or `visual.conv1.weight`. Stage A must update it before adding `PMTViTVisual`, otherwise PMT will fail because it has no `conv1`.

Current `core/test.py` pools non-fixed visual outputs through `base_model.visual.<pooling>()`, which assumes CNN feature maps. PMT token outputs need a separate token/global feature path.

## Implementation Tasks

### 1. Add PMT ViT Code Locally

Create:

```text
TVI-LFM/network/pmt_vit.py
TVI-LFM/network/pmt_vit_adapter.py
```

Copy only Stage A required single-branch code from:

```text
PMT-SYSU/pmt_sysu/model/vision_transformer.py
```

Required components:

- `DropPath`
- `Mlp`
- `Attention`
- `Block`
- `PatchEmbedOverlap`
- `ViT`
- `resize_pos_embed`

Do not dynamically import PMT-SYSU with `sys.path`. Do not copy `MultiBranchPatchEmbedOverlap`.

Modify local `ViT.forward_features()` so it can return either CLS or full tokens:

```python
def forward_features(self, x, return_tokens=False):
    ...
    x = self.norm(x)
    if return_tokens:
        return x
    return x[:, 0]
```

```python
def forward(self, x, return_tokens=False):
    return self.forward_features(x, return_tokens=return_tokens)
```

### 2. Add A Shared PMT Visual Adapter

Implement `PMTViTVisual` in `network/pmt_vit_adapter.py`.

Required behavior:

```python
class PMTViTVisual(nn.Module):
    def __init__(..., output_dim=2048, pretrained_path=None):
        self.vit = ViT(...)
        self.projection = nn.Linear(embed_dim, output_dim, bias=False)
```

Forbidden attributes:

```text
rgb_vit
ir_vit
rgb_backbone
ir_backbone
```

Forward output must be a dict:

```python
{
    "tokens": projected_tokens,
    "features": projected_tokens[:, 0],
    "raw_tokens": raw_tokens,
    "raw_features": raw_tokens[:, 0],
}
```

The adapter must accept `mode=None` for compatibility and ignore it.

Add:

```python
@property
def input_dtype(self):
    return self.vit.patch_embed.proj.weight.dtype
```

### 3. Load ImageNet ViT Weights Strictly

The PMT adapter, not CLIP RN50 loading, must load:

```text
../PMT-SYSU/pretrained/jx_vit_base_p16_224-80ecf9dd.pth
```

Support checkpoint forms:

```python
checkpoint["model"]
checkpoint["state_dict"]
checkpoint
```

Normalize keys:

```python
key = key.removeprefix("module.")
```

Skip classifier/distillation keys:

```text
head.*
head_dist.*
dist_token
```

Resize positional embedding from ImageNet grid to TVI-LFM grid:

```text
[1, 197, 768] -> [1, 254, 768]
```

The loader must print:

- checkpoint path
- loaded key count
- skipped key count and names
- missing keys
- unexpected keys
- positional embedding resize source and target shape

Core PMT backbone missing keys must be zero:

```text
patch_embed.proj.*
cls_token
pos_embed
blocks.*
norm.*
```

The new projection head is not part of the ImageNet checkpoint and should not be counted as a PMT core missing key.

### 4. Integrate `PMT_VIT` In CLIP Wrapper

Modify:

```text
TVI-LFM/network/clip_model/clip_model.py
```

Add `PMTViTVisual` import.

In `CLIP.__init__()`, handle `visual_name == "PMT_VIT"` before existing `RN50_ORI` and default ViT branches.

For `PMT_VIT`, construct:

```python
self.visual = PMTViTVisual(
    input_resolution=image_resolution,
    patch_size=pmt_patch_size,
    stride_size=pmt_stride_size,
    embed_dim=pmt_embed_dim,
    depth=pmt_depth,
    num_heads=pmt_num_heads,
    mlp_ratio=pmt_mlp_ratio,
    drop_rate=pmt_dropout,
    attn_drop_rate=pmt_attention_dropout,
    drop_path_rate=pmt_drop_path_rate,
    output_dim=embed_dim,
    pretrained_path=pmt_pretrained,
)
```

Update `CLIP.dtype`:

```python
if hasattr(self.visual, "input_dtype"):
    return self.visual.input_dtype
elif self.visual.__class__.__name__ == "Dual_Resnet":
    ...
else:
    return self.visual.conv1.weight.dtype
```

In `build_CLIP_from_openai_pretrained()`, if `name == "PMT_VIT"`, load the CLIP RN50 checkpoint only to recover the text-side shell:

```python
weight_name = "RN50"
```

But do not load any `visual.*` CLIP weights into PMT:

```python
if self.visual_name == "PMT_VIT" and k.startswith("visual."):
    continue
```

Also fix `load_param()` wrapper handling order. The current code filters keys before checking `model` or `state_dict`; wrapper extraction should happen before filtering.

### 5. Pass PMT Config From `core/build.py`

Modify the call to `build_CLIP_from_openai_pretrained()` in:

```text
TVI-LFM/core/build.py
```

Pass:

```text
pmt_pretrained
pmt_patch_size
pmt_stride_size
pmt_embed_dim
pmt_depth
pmt_num_heads
pmt_mlp_ratio
pmt_dropout
pmt_attention_dropout
pmt_drop_path_rate
```

Add defaults in config parsing or default YAML so that `RN50_ORI` still builds without PMT-specific YAML surprises.

### 6. Add Visual Output Helpers

In `CLIP2ReID`, add explicit helpers:

```python
def _is_pmt_visual(self):
    return self.args.pretrain_choice == "PMT_VIT"

def _uses_token_visual(self):
    return self.args.pretrain_choice in ["ViT-B/16", "PMT_VIT"]

def _uses_spatial_map_visual(self):
    return "RN" in self.args.pretrain_choice
```

Add dict-aware helpers:

```python
def _slice_visual_output(self, visual_output, start, end):
    if isinstance(visual_output, dict):
        return {
            k: v[start:end] if torch.is_tensor(v) else v
            for k, v in visual_output.items()
        }
    return visual_output[start:end]

def _get_visual_tokens(self, visual_output):
    if isinstance(visual_output, dict):
        return visual_output["tokens"]
    if torch.is_tensor(visual_output) and visual_output.ndim == 3:
        return visual_output
    raise TypeError(...)

def _get_visual_embedding(self, visual_output):
    if isinstance(visual_output, dict):
        return visual_output["features"].float()
    if self._uses_token_visual():
        return visual_output[:, 0, :].float()
    if self._uses_spatial_map_visual():
        return self.base_model.visual.__getattr__(self.args.pooling)(visual_output).float().squeeze()
    raise TypeError(...)
```

Use those helpers in:

- `encode_image_feat()`
- `forward()`
- any test/eval feature extraction path that handles visual outputs

The `forward()` split must work for dict outputs:

```python
image_visual = self.base_model.encode_image(torch.cat((rgb_imgs0, rgb_imgs1, ir_imgs), dim=0), mode)
b = ir_imgs.size(0)

rgb_visual = self._slice_visual_output(image_visual, 0, 2 * b)
ir_visual = self._slice_visual_output(image_visual, 2 * b, None)

rgb_feats_map = self._get_visual_tokens(rgb_visual)
ir_feats_map = self._get_visual_tokens(ir_visual)

rgb_feats = self._get_visual_embedding(rgb_visual)
ir_feats = self._get_visual_embedding(ir_visual)
```

Keep existing Stage A loss construction:

```python
pids = torch.cat([label_rgb, label_rgb, label_ir], dim=0)
img_feats = torch.cat((rgb_feats, ir_feats), dim=0)
```

Only `id` and `wrt` losses should be active.

### 7. Train Mode Support

Modify:

```text
TVI-LFM/core/train.py
```

Use explicit checks:

```python
if config.pretrain_choice in ["RN50", "RN50_ORI"]:
    mode = "1/3"
elif config.pretrain_choice in ["ViT-B/16", "PMT_VIT"]:
    mode = None
else:
    raise ValueError(...)
```

Use `raise ValueError`, not a bare `ValueError(...)` expression.

### 8. Freeze Text For Image-Only Stage A

When:

```text
training_mode: RGB_IR
freeze_text_in_image_only: true
```

freeze text-side parameters:

- `base_model.transformer`
- `base_model.token_embedding`
- `base_model.positional_embedding`
- `base_model.ln_final`
- `base_model.text_projection`

Do not freeze:

- `base_model.visual`
- classifier
- BN neck
- image-side heads
- `logit_scale` unless it is proven unused and explicitly documented

The optimizer already skips `requires_grad=False`, so freezing before optimizer construction is sufficient.

### 9. Fix Optimizer Grouping

Modify:

```text
TVI-LFM/solver/build.py
```

The current code checks `"transformer"` before `"visual"`. For Stage A, visual transformer parameters must not accidentally receive text learning rate.

Use explicit prefix/type grouping:

```python
is_visual = key.startswith("base_model.visual") or ".visual." in key
is_text = (
    "base_model.transformer" in key
    or "base_model.token_embedding" in key
    or "base_model.positional_embedding" in key
    or "base_model.ln_final" in key
    or "base_model.text_projection" in key
)
```

Then apply:

- visual params -> `lr_visual`
- text params -> `lr_txt`
- classifier/head params -> `lr_visual * classifier_lr_factor`

Keep the existing optimizer family and hyperparameters for A0/A1.

### 10. Fix Evaluation For Token Visuals

Modify:

```text
TVI-LFM/core/test.py
```

Do not call `visual.GEM()` or other CNN pooling on PMT token outputs.

Add or reuse a unified image feature helper:

```python
visual_output = base.encode_image_featmap(img, mode=None)
feat = base._get_visual_embedding(visual_output)
```

CNN paths must remain unchanged for `RN50` and `RN50_ORI`.

`Fix_Visual` branches must also use the same helper.

### 11. Config Work

Add Stage A configs under:

```text
TVI-LFM/config/stage_a/
```

Required files:

```text
TVI-LFM/config/stage_a/rn50_ori_stage_a_control.yaml
TVI-LFM/config/stage_a/pmt_vit_stage_a.yaml
```

Preferred implementation: merge each selected YAML with `config/default.yaml`. Then these configs only need to include overrides.

Shared Stage A overrides:

```yaml
training_mode: RGB_IR
loss_names: wrt,id
llm_aug: false
Feat_Filter: false
Fix_Visual: false
Return_B4_BN: false
test_model_type: IR
test_modality: IR
fusion_way: add
prj_output_dim: 2048
pooling: GEM
img_h: 288
img_w: 144
img_size: [288, 144]
batch_size: 32
num_pos: 4
seed: 1
freeze_text_in_image_only: true
```

A0 override:

```yaml
pretrain_choice: RN50_ORI
output_path: logs/stage_a_rn50_ori
```

A1 override:

```yaml
pretrain_choice: PMT_VIT
output_path: logs/stage_a_pmt_vit
pmt_pretrained: ../PMT-SYSU/pretrained/jx_vit_base_p16_224-80ecf9dd.pth
pmt_embed_dim: 768
pmt_patch_size: [16, 16]
pmt_stride_size: [12, 12]
pmt_depth: 12
pmt_num_heads: 12
pmt_mlp_ratio: 4.0
pmt_dropout: 0.03
pmt_attention_dropout: 0.0
pmt_drop_path_rate: 0.1
```

If merge is not implemented, both YAML files must include every key currently present in `config/default.yaml`, plus the new Stage A keys.

### 12. Tests And Smoke Checks

Create lightweight tests under:

```text
TVI-LFM/tests/
```

Recommended tests:

1. `PMTViTVisual` has one `vit` and no RGB/IR duplicate tower attributes.
2. `mode="rgb"`, `mode="ir"`, and `mode=None` use the same parameters and produce same-shape outputs.
3. Dummy input `[2, 3, 288, 144]` returns:
   - `raw_tokens [2, 254, 768]`
   - `tokens [2, 254, 2048]`
   - `features [2, 2048]`
4. ImageNet checkpoint preflight loads with zero core PMT backbone missing keys.
5. `CLIP2ReID.forward()` can split the three-way batch with PMT dict outputs.
6. `RGB_IR` loss path returns `id_loss` and `wrt_loss` only.
7. Text parameters are frozen when `freeze_text_in_image_only: true`.
8. Optimizer assigns PMT visual transformer parameters to visual LR, not text LR.
9. PMT eval feature extraction does not call CNN pooling.
10. `RN50_ORI` build and forward smoke still work.

Tests should not require the full SYSU dataset. Dataset-dependent checks belong to smoke commands, not unit tests.

## Execution Order

Run commands from:

```bash
cd /home/cgv841/ybj/TVI-LFM
```

### Static Checks

```bash
python -m compileall network core solver tools config
```

### Unit Tests

```bash
pytest -q tests
```

### PMT Checkpoint Preflight

Run a CPU or single-GPU preflight that builds `PMTViTVisual`, loads ImageNet weights, and checks output shapes on dummy input.

Expected:

```text
core PMT missing keys: 0
unexpected keys: 0 or only explicitly skipped classifier/distillation keys
tokens: [B, 254, 2048]
features: [B, 2048]
```

### Training Smoke

Use the real SYSU path configured for this machine. Run a short smoke for both configs.

If a short-iteration option is added, it must be implemented in code and used for both A0 and A1. Do not put inert keys such as `train_max_iter` into YAML unless the code consumes them.

### Evaluation Smoke

Verify:

- `RN50_ORI` eval path still works.
- `PMT_VIT` eval path extracts `[N, 2048]` features.
- PMT eval never calls `GEM` on tokens.
- mAP/CMC output is written normally.

### Formal Runs

Run A0 and A1 with matching settings:

```bash
python main.py --config_select config/stage_a/rn50_ori_stage_a_control.yaml
python main.py --config_select config/stage_a/pmt_vit_stage_a.yaml
```

If batch size must be reduced for memory, reduce it in both A0 and A1 and record the change.

## Result Record Template

Record results in a Markdown file after the runs:

```markdown
# TVI-LFM Stage A Results

## Environment
- repo: /home/cgv841/ybj/TVI-LFM
- commit:
- dataset:
- gpu:
- seed:
- date:

## Configs
- A0:
- A1:

## Metrics
| experiment | backbone | mAP | mINP | R1 | R5 | R10 | R20 | best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | RN50_ORI | | | | | | | |
| A1 | PMT_VIT | | | | | | | |

## Deviations
- None, or list exact deviations.

## Notes
- Include checkpoint-load messages and any memory-related changes.
```

## Acceptance Criteria

Code-level success:

- `PMT_VIT` builds.
- PMT ImageNet checkpoint loads with zero core backbone missing keys.
- `PMT_VIT` training forward returns valid `id` and `wrt` losses.
- PMT eval returns valid feature tensors and metrics.
- `RN50_ORI` path is not broken.
- Text-side params are frozen for image-only Stage A when configured.
- Optimizer grouping keeps PMT visual params on visual LR.

Experiment-level success:

- A0 and A1 are trained/evaluated under matched settings.
- Metrics are recorded for both.
- A1 does not need to beat A0 for Stage A to be considered successful; Stage A only validates the backbone swap under controlled conditions.

## Non-Goals For This Document

Do not use this Stage A task to implement:

- PMT full reproduction
- PMT gray-to-RGB progressive training
- MSEL/DCL integration
- text-image fusion
- LASTViT
- cross-dataset pretraining
- performance chasing sweeps

Those belong to later stages after Stage A is stable.
