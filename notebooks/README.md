# Final Colab notebooks

1. `01_model_a_scratch_full_yelp.ipynb` — completed Model A workflow; the added final cell exports text-free test predictions for the paired comparison.
2. `02_model_b_distilbert_full_yelp.ipynb` — full-data DistilBERT fine-tuning. Requires access to the exact Model A tokenized Drive cache and a CUDA GPU.
3. `03_export_final_artifacts.ipynb` — exports both trained models, tokenizers, metrics, histories, and paired test predictions to one local artifact ZIP.

Run Model A evaluation/export before Model B comparison. Do not use the archived 150K notebook or result files as the final full-data Model B experiment.
