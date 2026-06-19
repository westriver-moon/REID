# Stage A PMT Recipe Size Comparison Commands

Started from repository root:

```bash
cd /home/cgv841/ybj/TVI-LFM
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid

python main.py --config_select config/stage_a/pmt_vit_stage_a_pmt_recipe_256x128.yaml
python main.py --config_select config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144.yaml
```

The two commands are launched in parallel because GPU0 and GPU1 were both free at kickoff.

Launcher logs:

```text
/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_size_compare/a1r_256x128_launcher.log
/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_size_compare/a1r_288x144_launcher.log
/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_size_compare/a1r_256x128_run1_launcher.log
/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_size_compare/a1r_288x144_run1_launcher.log
/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_size_compare/a1r_256x128_run2_launcher.log
/home/cgv841/ybj/TVI-LFM/train_outputs/stage_a_pmt_recipe_size_compare/a1r_288x144_run2_launcher.log
```

Active clean run output dirs:

```text
/home/cgv841/ybj/TVI-LFM/logs/stage_a_pmt_vit_recipe_256x128_run2/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe
/home/cgv841/ybj/TVI-LFM/logs/stage_a_pmt_vit_recipe_288x144_run2/sysu/Base/Baseline_train[RGB_IR]_pmt_recipe
```
