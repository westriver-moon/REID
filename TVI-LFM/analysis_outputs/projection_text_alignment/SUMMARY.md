# Projection Removal and TVI Text Alignment

Read-only analysis of whether removing the current PMT `768 -> 2048`
projection would make later TVI-LFM text alignment harder.

## Code Facts

- `prj_output_dim` is used as the global CLIP2ReID `embed_dim`.
- For `PMT_VIT`, the PMT visual adapter receives `output_dim=embed_dim`.
  Therefore:
  - `prj_output_dim: 2048` gives PMT `768 -> 2048`.
  - `prj_output_dim: 768` makes the PMT projection an identity.
- The CLIP text side also uses the same `embed_dim` through
  `text_projection = [transformer_width, embed_dim]`.
- TVI text fusion paths assume image and text features already have the same
  final dimension. They use addition, subtraction, cross-attention, classifier
  sharing, or contrastive-style alignment directly on this shared dimension.
- Therefore, removing the PMT projection is not by itself a direct dimension
  mismatch, as long as the whole PMT+TVI config uses `prj_output_dim: 768`.

## Likely Benefits for Later Text Alignment

1. PMT-native visual space.
   - Text alignment would target PMT's native 768-dimensional CLS feature
     rather than a randomly initialized 2048-dimensional projection.

2. Less visual-side distortion.
   - The current 2048 projection is trained by ReID losses, not by language
     alignment. It may make the visual feature more ReID-discriminative but less
     naturally compatible with CLIP text.

3. Smaller alignment space.
   - 768 dimensions mean fewer classifier, BN, attention, and projection
     parameters. This can make text-stage adaptation lighter.

4. Cleaner PMT comparison.
   - It removes a large non-PMT component from the PMT branch, making the
     PMT-to-text stage easier to interpret.

## Likely Costs and Risks

1. Existing 2048-dimensional checkpoints and scripts become incompatible.
   - A 768-dimensional PMT Stage A checkpoint must be paired with a 768-dimensional
     Stage B text config. Original TVI-LFM scripts default to RN50_ORI and
     `prj_output_dim: 2048`.

2. It no longer matches RN50/TXI-LFM head width.
   - A0 RN50_ORI remains 2048-dimensional. A 768-dimensional PMT branch is a
     cleaner PMT comparison but no longer keeps the exact same head width as RN50.

3. Text projection is still not original OpenAI CLIP dimension.
   - TVI-LFM already resizes CLIP text projection when `prj_output_dim` differs
     from OpenAI RN50's original projection dimension. This happens for 2048 and
     would also happen for 768.

4. Capacity may be lower.
   - 2048 has more representational capacity. If TVI-LFM's text fusion benefited
     from that wider space, 768 may need retuning.

## Conclusion

Removing the PMT projection should not inherently make TVI text alignment harder
from a shape-compatibility perspective. It may actually make the PMT text stage
scientifically cleaner because image and text align in a PMT-native 768-dimensional
space.

The practical requirement is that Stage B must be designed as a 768-dimensional
PMT+TVI branch, not as a reuse of the original 2048-dimensional RN50 TVI-LFM
scripts or checkpoints.

Recommended next ablation:

1. Train `A1-PMT-faithful-768` image-only.
2. Start a small Stage B smoke test from that checkpoint with:
   - `pretrain_choice: PMT_VIT`
   - `prj_output_dim: 768`
   - matching PMT image size and stride
   - `training_mode: RGB_IR_Text`
   - `Fix_Visual: true`
   - text/fusion path enabled
3. Compare against the current 2048 PMT branch with the same Stage B recipe.
