# Log

1. Inspected PMT ViT, model builder, config loader, tests, and training entrypoint.
2. Added configurable multi-branch patch embedding while preserving the original single-branch PMT default.
3. Added `sysu_pmt_mbpatch.yaml` with two patch branches:
   - `[16,16] stride [12,12]`
   - `[16,8] stride [12,6]`
4. Added focused tests for MBPatch output shape and single-branch patch weight initialization.
5. Ran static check and pytest successfully.
6. Ran PMT-MBPatch preflight on the real SYSU-MM01 cache with ImageNet ViT weights.
7. Ran one-batch smoke training at epoch 1 and epoch 7.
8. Started full 24-epoch PMT-MBPatch training at `2026-06-15 20:39:07 Asia/Shanghai`.
9. Confirmed training passed epoch 1 and entered epoch 2.
10. Stopped the foreground run after epoch 2 checkpoint was saved, then fixed checkpoint RNG-state resume compatibility.
11. Verified resume with a one-batch smoke run from `outputs/pmt_sysu/mbpatch_reproduction/checkpoints/latest.pth`.
12. Restarted the full run in tmux session `pmt_mbpatch_full`; it resumed at epoch 3.

Notes:

- ImageNet ViT loading reports `Missing keys: 6; Unexpected keys: 0`, which is expected because MBPatch adds branch and fusion parameters beyond the original single-branch patch embed.
- The first branch is initialized from the original patch embedding path and the 1x1 fusion starts as anchor-branch identity.
- Full training is currently running; no final metric should be inferred until training and 10-trial evaluation finish.
