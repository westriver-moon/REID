# Stage A Size Audit Risks

## Findings

1. PMT recipe is not fully PMT-faithful because of `prj_output_dim: 2048`.
   Evidence: TVI-LFM config uses `prj_output_dim: 2048`, and `PMTViTVisual` projects raw 768-dim PMT tokens to 2048-dim tokens. PMT-SYSU config/model use `embed_dim: 768`.
   Impact: size comparison remains fair, but comparison against original PMT must be qualified.

2. `logit_scale` remains trainable but unused in PMT image-only losses.
   Evidence: `CLIP2ReID` creates `logit_scale`; PMT recipe returns temperature but losses do not depend on it.
   Impact: mostly harmless after `AdamWSkipEmptyGrad`, but it is unnecessary optimizer state/no-gradient handling.

3. No regular latest checkpoint/resume path is active in TVI-LFM main training.
   Evidence: `main.py` has epoch checkpoint/resume code commented; only best-eval model saving is active.
   Impact: if a run crashes before the first evaluation, there may be no resumable checkpoint. This is operational, not a scientific bias between the two active runs.

4. `AdamWSkipEmptyGrad` is an engineering compatibility patch.
   Evidence: older PyTorch AdamW fails on empty-gradient param groups; both active runs use the patched path.
   Impact: fair across the two runs, but should not be described as a research contribution.

5. LR scheduler emits PyTorch warnings because `scheduler.step(epoch)` is called before optimizer steps.
   Evidence: lightweight LR inspection reproduces the warning.
   Impact: with explicit epoch argument the closed-form LR values are as intended, but the warning can confuse log review.

## Non-issues Checked

- Text loading is disabled for `RGB_IR` + `test_modality: IR`.
- Batch shapes are correct for both sizes.
- Visible/IR label alignment is correct.
- Progressive gray stage length matches PMT-SYSU semantics: 6 epochs.
- Evaluation cadence matches PMT-SYSU semantics: every 2 epochs after epoch 2.
- Text encoder parameters are frozen for image-only training.
