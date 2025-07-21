### Acknowledgement
This evaluation is based on the repo SimplerEnv-OpenVLA and SimplerEnv.


### example

    ORIGSIZE=1 DENOISING_STEPS=20 bash scripts/pick_coke_can_visual_matching.sh  dita -0.8 results/dita-0.8 0
    ORIGSIZE=1 DENOISING_STEPS=20 bash scripts/pick_coke_can_variant_agg.sh  dita -0.8 results/dita-0.8 0



> We follow the OpenVLA instruction to use [rlds_modify](https://github.com/kpertsch/rlds_dataset_mod) to resize the image. Therefore, during the inference, we also resize the image with distortion. Resizing the image to 224*224 during training might not be reasonable.

> For complex tasks, we observe it is able to achieve slightly better results with more DDIM denosing steps.
