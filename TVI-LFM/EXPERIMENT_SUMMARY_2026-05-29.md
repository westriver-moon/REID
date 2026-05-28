# TVI-LFM Experiment Summary (2026-05-29)

This document summarizes the currently retained experiments in `TVI-LFM`, the main metric comparisons, and the conclusions we reached during the recent IR / VCM / SCHP discussion.

## 1. Scope

The experiments below are the runs that are still retained in the workspace and are representative of the current project status.

Local artifact roots:

```text
/home/cgv841/ybj/TVI-LFM/logs/mixed_msmt17_sysumm01
/home/cgv841/ybj/TVI-LFM/logs/sysumm01
/home/cgv841/ybj/TVI-LFM/logs/sysu_ir_vcm_ir
/home/cgv841/ybj/TVI-LFM/logs/schp_quality
```

Note: not every historical ablation run is still preserved in `logs/`. Some temporary or failed runs were cleaned. The conclusions from those runs are recorded here.

## 2. Retained Experiment Table

### 2.1 Final metrics

| Experiment | Task | Model / Key design | Final mAP | Rank-1 | Rank-5 | Rank-10 | Rank-20 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mixed_msmt17_sysumm01/rgb_lastvit_ep40` | SYSU RGB-related mixed training | `lastvit` + multi-branch patch embed + LAST Top-K | 0.8973 | 0.9295 | 0.9852 | 0.9902 | 0.9933 | Best retained RGB-direction result |
| `sysumm01/rgb_vitb_rgbonly_ep30` | SYSU RGB only | ViT-B RGB only baseline | 0.8412 | 0.8752 | 0.9734 | 0.9862 | 0.9924 | RGB single-dataset baseline |
| `sysumm01/ir_vitb_ironly_ep30` | SYSU IR only | ViT-B IR only baseline | 0.8305 | 0.7434 | 0.9466 | 0.9819 | 0.9952 | IR single-dataset baseline |
| `sysu_ir_vcm_ir/patch_mean_ep40` | SYSU IR + VCM IR | `patch_mean` + multi-branch patch embed + VCM tracklet K=2 | 0.8404 | 0.7525 | 0.9565 | 0.9918 | 0.9986 | Current best retained IR-direction result |
| `sysu_ir_vcm_ir/schp_part_ep40` | SYSU IR + VCM IR + SCHP | `schp_part_patch_mean` + soft SCHP part enhancement | 0.8354 | 0.7465 | 0.9538 | 0.9895 | 0.9983 | Did not beat `patch_mean` |

### 2.2 Best validation checkpoint by all-search mAP

| Experiment | Best epoch | Best mAP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mixed_msmt17_sysumm01/rgb_lastvit_ep40` | 34 | 0.9128 | 0.9371 | 0.9879 | 0.9927 | 0.9951 |
| `sysumm01/rgb_vitb_rgbonly_ep30` | 28 | 0.8543 | 0.8813 | 0.9756 | 0.9878 | 0.9935 |
| `sysumm01/ir_vitb_ironly_ep30` | 29 | 0.8364 | 0.7549 | 0.9424 | 0.9776 | 0.9936 |
| `sysu_ir_vcm_ir/patch_mean_ep40` | 34 | 0.8443 | 0.7606 | 0.9562 | 0.9866 | 0.9980 |
| `sysu_ir_vcm_ir/schp_part_ep40` | 34 | 0.8369 | 0.7482 | 0.9512 | 0.9862 | 0.9977 |

## 3. Current Best IR Configuration

The current best retained IR-direction configuration is:

```text
logs/sysu_ir_vcm_ir/patch_mean_ep40
```

This can be viewed as the current effective form of the earlier `A3.3` idea:

```text
A3.3 mixed-training recipe
+ remove LAST-style hard local filtering
+ keep the mixed SYSU IR + VCM IR setup
+ use patch_mean as the final global aggregator
```

Its core design is:

- Data: `SYSU IR + VCM IR`
- VCM usage: tracklet-level sampling
- Frames per tracklet: `K = 2`
- Batch composition:
  - SYSU IR: `8 IDs x 2 images`
  - VCM IR: `2 IDs x 2 tracklets x 2 frames`
- Backbone head:
  - ViT-B
  - multi-branch patch embedding
  - all patch mean pooling (`patch_mean`)
- Loss:
  - `sysu_id_weight = 1.0`
  - `sysu_triplet_weight = 1.0`
  - `vcm_id_weight = 0.2`
  - `vcm_triplet_weight = 0.3`
  - `tracklet_consistency_weight = 0.05`

Relevant config file:

```text
project/sysumm01/configs/sysu_ir_vcm_ir_patch_mean.yaml
```

## 4. What VCM Actually Brought

Comparing the retained IR baseline and the current best IR run:

```text
SYSU IR only                -> logs/sysumm01/ir_vitb_ironly_ep30
SYSU IR + VCM IR patch_mean -> logs/sysu_ir_vcm_ir/patch_mean_ep40
```

Delta on final metrics:

| Metric | Delta |
| --- | ---: |
| mAP | +0.0099 |
| Rank-1 | +0.0091 |
| Rank-5 | +0.0099 |
| Rank-10 | +0.0099 |
| Rank-20 | +0.0034 |

Interpretation:

- VCM IR is useful.
- The gain is not dramatic, but it is consistent across mAP / Rank-1 / Rank-5 / Rank-10 / Rank-20.
- The current best use of VCM is not "more complicated local modeling", but a relatively conservative mixed-training setup with tracklet sampling and mild VCM loss weights.

## 5. Why `patch_mean` Beat LAST / Hard Local Selection

The project originally used a LAST-style local token selection idea, but the retained best IR result is now the simpler `patch_mean` version.

The practical conclusion from the ablations was:

```text
hard local token selection did not help the IR-direction setting
```

Our working explanation is:

1. ReID identity cues are distributed across the body, not concentrated in a few highly salient patches.
2. In IR, "high-response" local regions are not always identity-discriminative.
3. Hard Top-K selection is more fragile under SYSU/VCM domain shift.
4. All-patch mean pooling is less selective, but more robust.

So the current project choice is:

```text
keep multi-branch patch embedding
remove hard LAST-style local filtering
use global all-patch mean pooling
```

## 6. SCHP Quality Validation

We performed offline SCHP parsing quality validation on mixed samples from SYSU IR, VCM IR, and SYSU RGB.

Summary:

| Subset | Count | Heuristic quality OK | Mean foreground ratio | Mean valid part count |
| --- | ---: | ---: | ---: | ---: |
| SYSU IR | 80 | 93.75% | 0.3903 | 3.625 |
| VCM IR | 80 | 92.50% | 0.3110 | 3.6125 |
| SYSU RGB | 20 | 85.00% | 0.4499 | 3.8000 |

Main observations:

- SCHP can usually find a person-like foreground region.
- `upper` / `lower` are more stable than `head` / `shoes`.
- VCM IR quality is visibly weaker than SYSU IR.
- SCHP looks usable as a soft prior, but not reliable enough as a strong structural decision maker in IR.

Reference artifact:

```text
logs/schp_quality/sysu_vcm_ir_lip
```

## 7. Why SCHP Part Fusion Did Not Win

Comparing the current best IR result and the SCHP-enhanced run:

```text
patch_mean  -> logs/sysu_ir_vcm_ir/patch_mean_ep40
SCHP part   -> logs/sysu_ir_vcm_ir/schp_part_ep40
```

Delta on final metrics:

| Metric | Delta (`SCHP - patch_mean`) |
| --- | ---: |
| mAP | -0.0050 |
| Rank-1 | -0.0060 |
| Rank-5 | -0.0027 |
| Rank-10 | -0.0023 |
| Rank-20 | -0.0003 |

Conclusion:

```text
SCHP-guided part fusion worked technically, but did not improve retrieval quality.
```

Our main interpretation:

1. SCHP is trained for RGB human parsing rather than IR ReID.
2. Fine semantic parts are less stable in IR.
3. Injecting SCHP directly into the final embedding path makes the model sensitive to parsing noise.
4. The current `patch_mean` baseline is already strong and robust, so a noisy prior has little room to help.

## 8. Recommended Current Backbone / Pipeline

For the current codebase, the recommended IR-direction backbone and training recipe is:

```text
ViT-B
+ multi-branch patch embedding
+ global all-patch mean pooling
+ SYSU IR + VCM IR mixed training
+ VCM tracklet sampling with K=2
+ conservative VCM loss weights
```

In short:

```text
the current best result comes from the mixed-training recipe,
not from more aggressive local or part-aware feature engineering
```

## 9. Recommended Next SCHP Uses

Based on the current results, SCHP is still potentially useful, but probably not as a strong part-fusion branch.

The two most promising next directions are:

### S1. Foreground-aware augmentation

Use SCHP only during training augmentation:

- background suppression / blur / erase
- foreground-preserving augmentations
- optional upper/lower occlusion augmentation

Why:

- low risk
- no test-time dependency
- avoids injecting parsing noise into the final embedding

### S2. VCM tracklet quality-aware frame sampling

Use SCHP only to score VCM frames:

- prefer frames with better foreground quality
- sample K frames from higher-quality subsets of a tracklet
- or use soft weighted frame sampling

Why:

- matches the actual strength of VCM (video / tracklet structure)
- uses SCHP as a data-quality prior rather than a representation prior

## 10. Final Recommendation

If we need one stable IR-direction baseline for further work, it should be:

```text
sysu_ir_vcm_ir/patch_mean_ep40
```

If we continue exploring SCHP, the recommended order is:

1. foreground-aware augmentation
2. VCM quality-aware frame sampling
3. only then consider more advanced semantic integration

At the moment, the evidence does **not** support keeping hard LAST-style local token filtering or SCHP part fusion as the mainline default for IR training.
