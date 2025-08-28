# ATT-ASPP-UNet for Fetal Abdominal Circumference Estimation

This repository contains the implementation of the **Attention-ASPP-UNet** model for fetal abdominal circumference (AC) estimation from ultrasound images. The project includes full training, evaluation, ablation study, and error analysis pipelines.

A baseline implementation is also provided, used for comparison against the proposed model in terms of segmentation accuracy and AC measurement performance.

---

## Code Structure and Descriptions

| File / Script                        | Description |
|-------------------------------------|-------------|
| `attention_aspp_unet_pipeline_stage.py` | Main pipeline for training, post-correction, and inference using the Attention-ASPP-UNet model. |
| `test_ablation.py`                 | Performs ablation studies to assess the contribution of different model components. |
| `eval_segmentation_batch.py`       | Computes segmentation performance metrics (e.g., Dice, IoU) for batch predictions. |
| `analyze_ac.py`                    | Calculates abdominal circumference (AC) and compares it across methods or ground truth. |
| `vis_error_analysis.py`            | Visualizes and analyzes failure cases to identify typical error patterns. |
| `convert_to_png.py`                | Converts `.mha` files from the original dataset to `.png` format for preprocessing and model training. The dataset is balanced with a **20% proportion of negative samples**. |

---

## Baseline Code

The repository also includes a baseline model implementation used to generate comparative results for segmentation and AC estimation. All evaluation scripts can process both the baseline and proposed model outputs for consistent comparison.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

