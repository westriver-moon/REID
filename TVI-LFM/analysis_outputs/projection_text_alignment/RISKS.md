# Risks

1. High: accidental mixed-dimensional Stage B.
   - If Stage A is trained at 768 but Stage B is launched with default
     `prj_output_dim: 2048`, checkpoint loading will fail or silently omit
     important weights depending on the load path.

2. High: comparing 768 PMT+TVI to 2048 RN50 TVI without labeling the head change.
   - Removing the projection improves PMT fidelity but changes the A0/A1
     comparability surface.

3. Medium: text projection quality.
   - OpenAI CLIP text projection is resized to the configured embedding
     dimension. This was already true for 2048 and would also be true for 768.
     The text stage should be verified with a smoke run before full training.

4. Medium: fusion modules scale with embedding dimension.
   - Cross-attention and classifier modules will be smaller at 768. This is not
     a bug, but it changes capacity and may require LR or epoch retuning.

5. Low: no direct add/cross-attention shape mismatch when configured correctly.
   - The code constructs both text and visual outputs with the same global
     `embed_dim`.
