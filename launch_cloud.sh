# Usage:
# Follow instructions at the Before you begin section of https://cloud.google.com/ai-platform/training/docs/custom-containers-training#before_you_begin
# chmod +x launch_cloud.sh
# ./launch_cloud.sh cc_4ts.yaml job_name
# ./launch_cloud.sh cc_count5x5_2ts.yaml job_name

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
  --config $1 \
  -- \
  --batch_size=432 \
  --count_cells=0 \
  --dec_enc_loss_amount=0.0 \
  --early_pred_state_metric_val=1.0 \
  --early_stop_step=100000 \
  --early_task_metric_val=1.0 \
  --eval_data_size=10000 \
  --eval_interval=5000 \
  --game_timesteps=3 \
  --learning_rate=0.00029995352381101767 \
  --lr_decay_rate_per1M_steps=0.9 \
  --max_train_steps=80000 \
  --reg_amount=0.0 \
  --target_pred_state_metric_val=0.01 \
  --target_task_metric_val=0.001 \
  --use_autoencoder=1 \
  --use_task_autoencoder=1 \
  --board_size=20 \
  --random_board_prob=0.1 \
  --decoder_counter_layers=2 \
  --decoder_counter_strides=2 \
  --decoder_layers=1 \
  --dropout_rate=0.0 \
  --encoded_size=21 \
  --encoder_layers=2 \
  --job_dir=/Users/wichersn/visualize_nn_runs/cc_different_ts_game3_model2_save_9933351/9 \
  --model_timesteps=2 \
  --timestep_layers=3 \
  --use_residual=1 \
  --use_rnn=1

echo "python3 -m tensorboard.main --logdir $JOB_DIR"