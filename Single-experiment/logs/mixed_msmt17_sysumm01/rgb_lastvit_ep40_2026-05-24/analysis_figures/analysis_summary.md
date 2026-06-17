# Mixed MSMT17 + SYSU-MM01 RGB LastViT Analysis

- Epochs: 1-40
- Best validation mAP: 0.9128 at epoch 34
- Best validation Rank-1: 0.9394 at epoch 33
- Final all-search mAP / Rank-1: 0.8973 / 0.9295
- Final indoor-search mAP / Rank-1: 0.9543 / 0.9277
- Minimum train loss: 0.0209 at epoch 40
- Average epoch time: 401.3s

## Reading

The run converges steadily: total loss decreases from 7.3530 to 0.0209. Validation mAP climbs quickly in the first 20 epochs and peaks later at epoch 34, while the final checkpoint is slightly below the best checkpoint. Use `checkpoints/best.pth` for downstream fine-tuning or reporting unless you specifically need the last state.
