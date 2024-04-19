import os 


def runs(env, MAPS, ALGOS, UNITS, seed_max, parallel, cuda_device): 
    for map in MAPS: 
        for algo in ALGOS: 
            print(f"env is {env}, map is {map}, algo is {algo}, max seed is {seed_max}")
            for units in UNITS: 
                exp = units  
                print(f"exp is: {exp}") 
                for seed in range(seed_max): 
                    print(f"seed is: {seed}") 
                    command = f"CUDA_VISIBLE_DEVICES={cuda_device} python ../train/train_smac.py --env_name {env} --algorithm_name {algo} --experiment_name {exp} \
                        --map_name {map} --seed {seed} --units {units} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
                        --num_env_steps 20000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32"
                    if parallel: command += " &" 
                    os.system(command) 


runs(
    env="StarCraft2v2", 
    MAPS = ["10gen_protoss", "10gen_terran", "10gen_zerg"], 
    ALGOS = ["rmappo"], 
    UNITS = ["5v5", "10v10", "10v11", "20v20", "20v23"], 
    seed_max = 3, 
    parallel = True, 
    cuda_device=0
)

runs(
    env="StarCraft2v2", 
    MAPS = ["10gen_protoss", "10gen_terran", "10gen_zerg"], 
    ALGOS = ["mappo"], 
    UNITS = ["5v5", "10v10", "10v11", "20v20", "20v23"], 
    seed_max = 3, 
    parallel = True, 
    cuda_device=1
)

runs(
    env="StarCraft2v2", 
    MAPS = ["10gen_protoss", "10gen_terran", "10gen_zerg"], 
    ALGOS = ["happo"], 
    UNITS = ["5v5", "10v10", "10v11", "20v20", "20v23"], 
    seed_max = 3, 
    parallel = True, 
    cuda_device=3
)
