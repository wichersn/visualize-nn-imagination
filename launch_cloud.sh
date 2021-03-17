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

echo "python3 -m tensorboard.main --logdir $JOB_DIR"