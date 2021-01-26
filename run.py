import ray
from multiagent.fullobs_collect_treasure import FullObsCollectTreasureEnv
from ray import tune
from ray.tune.registry import register_env

from maac import MAACTrainer

if __name__ == '__main__':
    ray.init(address='auto')


    def fullobs_collect_treasure(args):
        return FullObsCollectTreasureEnv()


    register_env('fullobs_collect_treasure', fullobs_collect_treasure)

    config = {
        "framework": "torch",
        "env": "fullobs_collect_treasure",
        "num_gpus": .5,
        "num_cpus_for_driver": 4,
    }

    tune.run(MAACTrainer,
             name="maac",
             config=config,
             metric='episode_reward_mean',
             mode='max',
             )
