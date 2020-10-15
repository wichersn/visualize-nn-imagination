# Launch on cloud
    chmod +x launch_cloud.sh
    ./launch_cloud.sh job_name
    
You might have to change the `--region` flag in `launch_cloud` to get it to schedule. Sometimes jobs stay `Queued` for 10 hours. See [here](https://cloud.google.com/compute/docs/gpus) for regions with p100s or whatever gpu type is in `cloud_config.yaml`
    
# Train local
     python3 -m train.train --job_dir something
     
# Tensorboard
Replace these paths with the ones you want to see

    python3 -m tensorboard.main --logdir_spec 0:gs://visualize-nn-imagination-runs/full_hp_gpu_603721,1:gs://visualize-nn-imagination-runs/full_hp_gpu_p100_605262
    
# Get images
This will add the images to a `img` directory in `gs://visualize-nn-imagination-runs/full_hp_gpu_p100_605262`
    
    python3 -m train.save_model_result_imgs --job_dir gs://visualize-nn-imagination-runs/full_hp_gpu_p100_605262 --is_hp_serach_root
    
Download them with 

    gsutil cp -r gs://visualize-nn-imagination-runs/full_hp_gpu_603721/imgs/ some_local_dir
    
