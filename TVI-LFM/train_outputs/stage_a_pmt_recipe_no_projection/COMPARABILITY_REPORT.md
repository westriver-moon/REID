# Comparability Report

Most direct comparison:

- Current run: `A1 PMT recipe 288x144, prj_output_dim=768`.
- Reference run: `A1 PMT recipe 288x144, prj_output_dim=2048`.

Comparable:

- Same dataset, split, evaluation modality, and gallery-trial averaging path.
- Same PMT backbone configuration and ImageNet initialization.
- Same PMT recipe losses and augmentation.
- Same training length and evaluation cadence.
- Same seed and batch layout.

Not comparable as a single-factor PMT-SYSU reproduction:

- The run still uses TVI-LFM/CLIP2ReID training and evaluation code.
- It does not include PMT-SYSU MBPatch.
- Text projection is resized to 768 but text is frozen and unused in Stage A.

Primary hypothesis:

- If 768 improves over 2048, the random 2048 projection/head likely hurt PMT
  optimization or retrieval representation.
- If 768 underperforms, the wider TVI-LFM head may be helping this integration,
  or the 768 branch may need retuning.
