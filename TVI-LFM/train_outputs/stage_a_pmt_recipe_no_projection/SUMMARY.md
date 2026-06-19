# Stage A PMT Recipe No-Projection Run

Goal: isolate the effect of removing the PMT `768 -> 2048` projection/head.

Design:

- Base comparison: completed `A1 PMT recipe 288x144` run with `prj_output_dim: 2048`.
- New run: same 288x144 PMT-recipe settings, but `prj_output_dim: 768`.
- The PMT adapter therefore uses `projection = Identity()`.
- The classifier, BN, CLIP text projection, and evaluation embedding dimension are all 768.
- Training remains Stage A image-only: `training_mode: RGB_IR`, `joint_mode: image_only`, `test_modality: IR`.

Pre-start checks passed:

- `projection` is `Identity`.
- `model.embed_dim == 768`.
- `text_projection.shape == (512, 768)`.
- `classifier.BN.num_features == 768`.
- `classifier.classifier.weight.shape == (395, 768)`.
- Text tensors are frozen in image-only mode.
- Loader batch has no text keys and preserves PMT aligned visible/IR label layout.
- Forward smoke covered both `gray_ir` and `rgb_ir` branches with finite losses.

Environment:

- Conda environment: `clipreid`.
- GPU: physical GPU 0, observed idle before launch.
- Dataset: `/home/cgv841/datasets/SYSU-MM01/`.
- PMT ImageNet checkpoint: `/home/cgv841/ybj/PMT-SYSU/pretrained/jx_vit_base_p16_224-80ecf9dd.pth`.

Notes:

- This is not a pure PMT-SYSU run. It remains TVI-LFM/CLIP2ReID infrastructure with a 768-dimensional PMT-compatible head.
- Comparability is strongest against the existing `stage_a_pmt_vit_recipe_288x144_run2` 2048-dimensional run.

Final result:

- Completed all 24 epochs.
- Best Rank-1 checkpoint: epoch 21, Rank-1 `65.53%`, mAP `64.11%`, mINP `51.65%`.
- Best mAP / final validation: epoch 23, Rank-1 `65.44%`, mAP `64.11%`, mINP `51.58%`.
- Matched 2048-projection reference at 288x144: Rank-1 `64.71%`, mAP `62.42%`, mINP `48.98%`.
- Delta vs 2048 projection at matched 288x144: best Rank-1 `+0.82`, mAP-at-best-R1 `+1.69`, mINP-at-best-R1 `+2.67` percentage points.

Generated analysis:

- `RESULT_ANALYSIS.md`
- `plots/projection_comparison_dashboard.png`
- `plots/projection_best_metric_bars.png`
- `no_projection_train_metrics.csv`
- `no_projection_eval_metrics.csv`
- `no_projection_summary_metrics.csv`
