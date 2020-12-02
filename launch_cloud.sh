# Usage:
# Follow instructions at the Before you begin section of https://cloud.google.com/ai-platform/training/docs/custom-containers-training#before_you_begin
# chmod +x launch_cloud.sh
# ./launch_cloud.sh cc_4ts.yaml job_name

# This launches a new training on google cloud.
# If you want to do something else, you can use these commands as examples.

CONFIG_NO_EXT="$(basename $1 .yaml)"
JOB_NAME="${CONFIG_NO_EXT}_$2_"$(($(date +%s)-1601800000)) ;
echo launching $JOB_NAME

BUCKET_ID=visualize-nn-imagination-runs
PROJECT_ID=visualize-nn-imagination
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME

# Have extra quota in us-west1, us-central1, asia-east1
gcloud ai-platform jobs submit training $JOB_NAME \
  --project $PROJECT_ID \
  --package-path train/ \
  --module-name train.train \
  --region asia-east1	\
  --python-version 3.7 \
  --runtime-version 2.2 \
  --job-dir $JOB_DIR \
  --config $1 \

echo "python3 -m tensorboard.main --logdir $JOB_DIR"