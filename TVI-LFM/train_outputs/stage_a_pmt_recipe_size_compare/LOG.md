# Stage A PMT Recipe Size Comparison Log

2026-06-19 CST

- Designed a controlled input-size comparison.
- GPU0 and GPU1 were free before kickoff.
- The 256x128 run is assigned to physical GPU0 with `CUDA_VISIBLE_DEVICES: "0"`.
- The 288x144 run is assigned to physical GPU1 with `CUDA_VISIBLE_DEVICES: "1"` and local `gpu_id: "0"`.
- Both runs use the same PMT recipe: progressive gray/RGB visible branch, PMT triplet, MSEL, DCL, AdamW, cosine LR, 24 epochs.
- First kickoff at 05:36:34 failed at the first AdamW optimizer step with `UnboundLocalError: local variable 'beta1' referenced before assignment`.
- Root cause: this PyTorch AdamW build fails when a per-parameter group has no gradient in the current step; PMT recipe can leave some trainable parameters unused for a given forward path.
- Applied engineering fix in `solver/build.py`: use `AdamWSkipEmptyGrad`, which skips no-gradient groups and keeps normal AdamW behavior for groups with gradients.
- A `run1` restart at 05:39:50 confirmed the same full-model failure before the SafeAdamW patch.
- Final clean kickoff uses `run2` output paths:
  - `A1R_256x128`: launcher PID `3517143`, python PID `3517164`, physical GPU0.
  - `A1R_288x144`: launcher PID `3517153`, python PID `3517169`, physical GPU1.
- At 05:45:51 both run2 jobs were still running, occupying GPU0/GPU1, with no AdamW traceback after model/data/optimizer setup.
