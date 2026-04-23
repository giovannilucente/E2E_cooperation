# Cooperative E2E models
Repository forked from [CoLMDriver](https://github.com/cxliu0314/colmdriver).


## Installation

### Env
#### Step 1: Basic Installation for colmdriver
Get code and create pytorch environment.
```Shell
git clone https://github.com/giovannilucente/E2E_cooperation.git
conda create --name colmdriver python=3.7 cmake=3.22.1
conda activate colmdriver
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudnn -c conda-forge

python -m pip install -r opencood/requirements.txt
python -m pip install -r simulation/requirements.txt
```

#### Step 2: Download and setup CARLA 0.9.10.1.
```Shell
chmod +x simulation/setup_carla.sh
./simulation/setup_carla.sh
python -m easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
mkdir external_paths
ln -s ${PWD}/carla/ external_paths/carla_root
# If you already have a Carla, just create a soft link to external_paths/carla_root
```

The file structure should be:
```Shell
|--CoLMDriver
    |--external_paths
        |--carla_root
            |--CarlaUE4
            |--Co-Simulation
            |--Engine
            |--HDMaps
            |--Import
            |--PythonAPI
            |--Tools
            |--CarlaUE4.sh
            ...
```

Note: we choose the setuptools==41 to install because this version has the feature `easy_install`. After installing the carla.egg you can install the lastest setuptools to avoid No module named distutils_hack.

Steps 3,4,5 are for perception module.

#### Step 3: Install Spconv 
```Shell
python -m pip install spconv-cu113
```

#### Step 4: Set up
```Shell
# Set up
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace 
```

#### Step 5: Install pypcd
```Shell
# go to another folder
cd ..
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ..
```


## Evaluation on Interdrive benchmark

### Evaluation of baselines
Setup and get ckpts.

| Methods   | TCP | CoDriving               |
|-----------|---------|---------------------------|
| Installation Guide  | [github](https://github.com/OpenDriveLab/TCP)  | [github](https://github.com/CollaborativePerception/V2Xverse) |
| Checkpoints     |  [google drive](https://drive.google.com/file/d/1D-10aMUAOPk1yiOr-PvSOJMS_xi_eR7U/view?usp=sharing)  |  [google drive](https://drive.google.com/file/d/1Izg9wZ3ktR-mwn7J_ZqxrwBmtI1YJ6Xi/view?usp=sharing)   |

The downloaded checkpoints should follow this structure:
```Shell
|--CoLMDriver
    |--ckpt
        |--codriving
            |--perception
            |--planning
        |--TCP
            |--new.ckpt
```

Evaluate TCP, CoDriving on Interdrive benchmark:

```Shell
# Start CARLA server, if port 2000 is already in use, choose another
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=2000 -prefer-nvidia

# Evaluate TCP on Interdrive
bash scripts/eval/eval_mode.sh 0 2000 tcp ideal Interdrive_all

# Evaluate CoDriving on Interdrive
bash scripts/eval/eval_mode.sh 0 2000 codriving ideal Interdrive_all
```

- CARLA processes may fail to stop，please kill them in time.


## <span id="dataset"> Dataset
The dataset for training CoLMDriver is obtained from [V2Xverse](https://github.com/CollaborativePerception/V2Xverse), which contains experts behaviors in CARLA. You may get the dataset in two ways:
- Download from [this huggingface repository](https://huggingface.co/datasets/gjliu/V2Xverse).
- Generate the dataset by yourself, following this [guidance](https://github.com/CollaborativePerception/V2Xverse).

The dataset should be linked/stored under `external_paths/data_root/` follow this structure:
```Shell
|--data_root
    |--weather-0
        |--data
            |--routes_town{town_id}_{route_id}_w{weather_id}_{datetime}
                |--ego_vehicle_{vehicle_id}
                    |--2d_bbs_{direction}
                    |--3d_bbs
                    |--actors_data
                    |--affordances
                    |--bev_visibility
                    |--birdview
                    |--depth_{direction}
                    |--env_actors_data
                    |--lidar
                    |--lidar_semantic_front
                    |--measurements
                    |--rgb_{direction}
                    |--seg_{direction}
                    |--topdown
                |--rsu_{vehicle_id}
                |--log
            ...
```

## <span id="train"> Training

### Perception module
Our perception module follows [CoDriving](https://github.com/CollaborativePerception/V2Xverse).
To train perception module from scratch or a continued checkpoint, run the following commonds:
```Shell
# Single GPU training
python opencood/tools/train.py -y opencood/hypes_yaml/v2xverse/colmdriver_multiclass_config.yaml [--model_dir ${CHECKPOINT_FOLDER}]

# DDP training
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y opencood/hypes_yaml/v2xverse/colmdriver_multiclass_config.yaml [--model_dir ${CHECKPOINT_FOLDER}]

# Offline testing of perception
python opencood/tools/inference_multiclass.py --model_dir ${CHECKPOINT_FOLDER}
```
The training outputs can be found at `opencood/logs`.
Arguments Explanation:
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune or continue-training. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder. In this case, ${CONFIG_FILE} can be `None`,
- `--nproc_per_node` indicate the GPU number you will use.

### Planning module
Given a checkpoint of perception module, we freeze its parameters and train the down-stream planning module in an end-to-end paradigm. The planner gets BEV perception feature and occupancy map as input and targets to predict the future waypoints of ego vehicle.

Train the planning module with a given perception checkpoint on multiple GPUs:
```Shell
# Train planner
bash scripts/train/train_planner_e2e.sh $GPU_ids $num_GPUs $perception_ckpt $planner_config $planner_ckpt_resume $name_of_log $save_path

# Example
bash scripts/train/train_planner_e2e.sh 0,1 2 ckpt/colmdriver/percpetion covlm_cmd_extend_adaptive_20 None log ./ckpt/colmdriver_planner

# Offline test
bash scripts/eval/eval_planner_e2e.sh 0,1 ckpt/colmdriver/percpetion covlm_cmd_extend_adaptive_20 ckpt/colmdriver/waypoints_planner/epoch_26.ckpt ./ckpt/colmdriver_planner
```
