# ABAW8 EMI challenge

-----------------------
This project contains ml expirements with ml models for Emotional Mimicry Intensity task by ABAW 8 challenge. 
The Hume-Vidmimic dataset is used.

### Reproduction of the Most successful Pipeline

The only needed notebooks:

* ipynb_new/frame_encodign.ipynb
* ipynb_new/single_mod_models/audio_model_v2_unfrozen_enc.ipynb
* ipynb_new/single_mod_models/text_model_v3_encoder_finetune.ipynb
* ipynb_new/multi_mod_models/multimodal_late_fusion_finetuend_v1.ipynb

Execute cells in first three files, then execute the last one providing paths for saved embeddings and models saved
during execution of previous notebooks.
