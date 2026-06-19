# Scientific Changelog

This kickoff does not add a new method beyond the Stage A plan.

Stage A controlled variable:

- A0 uses the original `RN50_ORI` visual path.
- A1 replaces the visual path with one fully shared `PMT_VIT` visual backbone initialized from ImageNet ViT-B/16.

Held fixed:

- `training_mode: RGB_IR`
- `loss_names: wrt,id`
- SYSU-MM01 data path and evaluation protocol
- TVI-LFM transforms, sampler, optimizer family, scheduler, warmup, and seed

Text assets, text fusion, PMT SYSU training losses, PMT progressive schedule, and official PMT SYSU checkpoint initialization are not used.

Runtime control update:

- A1 was shortened from the inherited 120-epoch TVI-LFM schedule to 40 epochs.
- This is a training-budget change, not a model-method change.
- A1 validation is configured from epoch 20 every 2 epochs so the shortened run still produces validation evidence.
