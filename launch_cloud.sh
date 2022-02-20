# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage:
# Follow instructions at the Before you begin section of https://cloud.google.com/ai-platform/training/docs/custom-containers-training#before_you_begin
# chmod +x launch_cloud.sh
# ./launch_cloud.sh gol4ts.yaml job_name
# ./launch_cloud.sh count5x5_2ts.yaml job_name

# This launches a new training on google cloud.
# If you want to do something else, you can use these commands as examples.

CONFIG_NO_EXT="$(basename $1 .yaml)"
JOB_NAME="${CONFIG_NO_EXT}_$2_"$(($(date +%s)-1601800000)) ;
echo launching $JOB_NAME

BUCKET_ID=visualize-nn-imagination-runs
PROJECT_ID=visualize-nn-imagination
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME

# Have extra quota in us-west1, us-central1, asia-east1
# australia-southeast1, europe-west1, us-east1, europe-west4
gcloud ai-platform jobs submit training $JOB_NAME \
  --project $PROJECT_ID \
  --package-path train/ \
  --module-name train.train \
  --region australia-southeast1	\
  --python-version 3.7 \
  --runtime-version 2.2 \
  --job-dir $JOB_DIR \
  --config configs/$1 \
  # -- --adver_train_steps=50000 --adver_weight=1.0 --batch_size=295 --early_pred_state_metric_val=1.0 --early_stop_step=200000 --early_task_metric_val=0.1 --eval_data_size=10000 --eval_interval=5000 --game_timesteps=3 --grid_size=2 --learning_rate=0.0017268333329049197 --lr_decay_rate_per1M_steps=0.23956616873864867 --max_dec_train_steps=10000 --max_train_steps=400000 --patch_size=2 --reg_amount=0.0 --target_pred_state_metric_val=0.01 --target_task_metric_val=0.001 --task=count --use_autoencoder=1 --use_task_autoencoder=1 --board_size=10 --random_board_prob=0.1 --adver_decoder_layers=2 --decoder_counter_strides=1 --decoder_layers=3 --decoder_task_layers=6 --dropout_rate=0.0 --encoded_size=34 --encoder_layers=2 --model_timesteps=3 --timestep_layers=4 --use_residual=0 --use_rnn=1
echo "python3 -m tensorboard.main --logdir $JOB_DIR"
