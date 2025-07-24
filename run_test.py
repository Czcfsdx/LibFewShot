# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "./results/DN4-miniImageNet--ravi-Conv64F-5-1-Dec-01-2021-06-05-20"
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "0",
    "workers": 8,
    "n_gpu": 1,
    "test_episode": 600,
    # "episode_size": 1,
    # "test_query": 1,
    "shot_num": 1,
    "test_shot": 1,
}


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    if config["ensemble"] and config["ensemble_kwargs"]["name"] == "quickboost":
        config = Config(
            os.path.join(
                config["ensemble_kwargs"]["other"]["pretrain_model_path"], "config.yaml"
            ),
            config,
        ).get_config_dict()

        # config["ensemble_kwargs"]["other"]["test_standalone"] = False
        if not config["ensemble_kwargs"]["other"]["test_standalone"]:
            # use pretrain model test config to override quickboost test config
            model_config = Config(
                os.path.join(
                    config["ensemble_kwargs"]["other"]["pretrain_model_path"],
                    "config.yaml",
                )
            ).get_config_dict()
            config["test_shot"] = model_config["test_shot"]
            config["test_way"] = model_config["test_way"]
            config["test_query"] = model_config["test_query"]
            config["test_epoch"] = model_config["test_epoch"]
            config["test_episode"] = model_config["test_episode"]

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
