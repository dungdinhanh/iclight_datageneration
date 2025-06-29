#!bin/bash


# This script is used to generate the source of px files for the lbm project
# the background is used from fashionpedia dataset
# the foreground is used from the studio px dataset
# ground truth is used from studio px dataset


cmd="accelerate launch lbm_dataset_generate_ap.py --image_folder /home/ubuntu/data/data_2D/px_wh_54k/images --mask_folder /home/ubuntu/data/data_2D/px_wh_54k/images \
 --source_folder /home/ubuntu/data/data_2D_processed/px_wh_54k/source_b_fashioinpedia/ --background_dir /home/ubuntu/data/data_2D_processed/fashionpedia_46k/background \
 --metadata_csv /home/ubuntu/data/data_2D_processed/px_wh_54k/record_blipcap_p5_filtered_oversized_with_background.csv --image_width 640 --image_height 832"
 echo ${cmd}
 eval ${cmd}
 