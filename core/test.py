# -*- coding: utf-8 -*-
import os
import builtins
from logging import getLogger
from time import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

import core.model as arch
from core.data import get_dataloader
from core.utils import (
    init_logger_config,
    prepare_device,
    init_seed,
    create_dirs,
    AverageMeter,
    count_parameters,
    ModelType,
    TensorboardWriter,
    mean_confidence_interval,
    get_instance,
)


class Test(object):
    """
    The tester.

    Build a tester from config dict, set up model from a saved checkpoint, etc. Test and log.
    """

    def __init__(self, rank, config, result_path=None):
        self.rank = rank
        self.config = config
        self.config["rank"] = rank
        self.result_path = result_path
        self.distribute = self.config["n_gpu"] > 1
        self.viz_path, self.state_dict_path = self._init_files(config)
        self.logger = self._init_logger()
        self.device, self.list_ids = self._init_device(rank, config)
        self.writer = self._init_writer(self.viz_path)
        self.test_meter = self._init_meter()
        print(config)
        self.quickboost = False
        self.quickboost_test_standalone = False
        if config["ensemble"] and config["ensemble_kwargs"]["name"] == "quickboost":
            self.quickboost = True
            self.ensemble_way, self.ensemble_name = self._init_ensemble(config)
            self.state_dict_path = os.path.join(
                config["ensemble_kwargs"]["other"]["pretrain_model_path"],
                "checkpoints",
                "model_best.pth"
            )
            if self.ensemble_way.test_standalone:
                self.quickboost_test_standalone = True
            self.model, self.model_type = self._init_model(config)
            self.ensemble_way.emb_func = self.model.emb_func
        else:
            self.model, self.model_type = self._init_model(config)
        self.test_loader = self._init_dataloader(config)

    def test_loop(self):
        """
        The normal test loop: test and cal the 0.95 mean_confidence_interval.
        """

        if self.quickboost:
            self.ensemble_way.load_model(os.path.join(self.result_path, "checkpoints"))
            if self.quickboost_test_standalone:
                print("QuickBoost start testing standalone ...")
            else:
                print(f"QuickBoost start testing with {self.ensemble_way.pretrain_model_name} ...")

        total_accuracy = 0.0
        total_h = np.zeros(self.config["test_epoch"])
        total_accuracy_vector = []

        for epoch_idx in range(self.config["test_epoch"]):
            print("============ Testing on the test set ============")
            _, accuracies = self._validate(epoch_idx)
            test_accuracy, h = mean_confidence_interval(accuracies)
            print("Test Accuracy: {:.3f}\t h: {:.3f}".format(test_accuracy, h))
            total_accuracy += test_accuracy
            total_accuracy_vector.extend(accuracies)
            total_h[epoch_idx] = h

        aver_accuracy, h = mean_confidence_interval(total_accuracy_vector)
        print("Aver Accuracy: {:.3f}\t Aver h: {:.3f}".format(aver_accuracy, h))
        print("............Testing is end............")

        if self.writer is not None:
            self.writer.close()
            if self.distribute:
                dist.barrier()
        elif self.distribute:
            dist.barrier()

    def _validate(self, epoch_idx):
        """
        The test stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        if not self.quickboost_test_standalone:
            # switch to evaluate mode
            self.model.eval()
            if self.distribute:
                self.model.module.reverse_setting_info()
            else:
                self.model.reverse_setting_info()
        meter = self.test_meter
        meter.reset()
        episode_size = self.config["episode_size"]
        accuracies = []

        end = time()
        if self.quickboost_test_standalone:
            enable_grad = False
        else:
            enable_grad = self.model_type != ModelType.METRIC
        log_scale = self.config["episode_size"]
        with torch.set_grad_enabled(enable_grad):
            loader = self.test_loader
            for batch_idx, batch in enumerate(zip(*loader)):
                if self.rank == 0:
                    self.writer.set_step(
                        int(
                            (
                                epoch_idx * len(self.test_loader)
                                + batch_idx * episode_size
                            )
                            * self.config["tb_scale"]
                        )
                    )

                meter.update("data_time", time() - end)

                # calculate the output
                calc_begin = time()
                if self.quickboost_test_standalone:
                    output, acc = self.ensemble_way.test([elem for each_batch in batch for elem in each_batch])
                else:
                    output, acc = self.model([elem for each_batch in batch for elem in each_batch])
                    if self.quickboost:
                        output, acc = self.ensemble_way.test([elem for each_batch in batch for elem in each_batch], output)
                accuracies.append(acc)
                meter.update("calc_time", time() - calc_begin)

                # measure accuracy and record loss
                meter.update("acc", acc)

                # measure elapsed time
                meter.update("batch_time", time() - end)

                if ((batch_idx + 1) * log_scale % self.config["log_interval"] == 0) or (
                    batch_idx + 1
                ) * episode_size >= max(map(len, loader)) * log_scale:
                    info_str = (
                        "Epoch-({}): [{}/{}]\t"
                        "Time {:.3f} ({:.3f})\t"
                        "Calc {:.3f} ({:.3f})\t"
                        "Data {:.3f} ({:.3f})\t"
                        "Acc@1 {:.3f} ({:.3f})".format(
                            epoch_idx,
                            (batch_idx + 1) * log_scale,
                            max(map(len, loader)) * log_scale,
                            meter.last("batch_time"),
                            meter.avg("batch_time"),
                            meter.last("calc_time"),
                            meter.avg("calc_time"),
                            meter.last("data_time"),
                            meter.avg("data_time"),
                            meter.last("acc"),
                            meter.avg("acc"),
                        )
                    )
                    print(info_str)
                end = time()

        if not self.quickboost_test_standalone:
            if self.distribute:
                self.model.module.reverse_setting_info()
            else:
                self.model.reverse_setting_info()
        return meter.avg("acc"), accuracies

    def _init_files(self, config):
        """
        Init result_path(log_path, viz_path) from the config dict.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (viz_path, checkpoints_path).
        """
        if self.result_path is not None:
            result_path = self.result_path
        elif self.config["ensemble"]:
            result_dir = "{}-{}-{}-{}".format(
                config["ensemble_kwargs"]["name"],
                config["data_root"].split("/")[-1],
                config["way_num"],
                config["shot_num"],
            )
            result_path = os.path.join(config["result_root"], result_dir)
        else:
            result_dir = "{}-{}-{}-{}-{}".format(
                config["classifier"]["name"],
                # you should ensure that data_root name contains its true name
                config["data_root"].split("/")[-1],
                config["backbone"]["name"],
                config["way_num"],
                config["shot_num"],
            )
            result_path = os.path.join(config["result_root"], result_dir)
        log_path = os.path.join(result_path, "log_files")
        viz_path = os.path.join(log_path, "tfboard_files")

        if self.config["ensemble"]:
            init_logger_config(
                config["log_level"],
                log_path,
                config["ensemble_kwargs"]["name"],
                "",
                rank=self.rank,
                is_train=False
                )
        else:
            init_logger_config(
                config["log_level"],
                log_path,
                config["classifier"]["name"],
                config["backbone"]["name"],
                rank=self.rank,
                is_train=False
            )

        state_dict_path = os.path.join(result_path, "checkpoints", "model_best.pth")

        if self.rank == 0:
            create_dirs([result_path, log_path, viz_path])
            

        return viz_path, state_dict_path

    def _init_logger(self):
        self.logger = getLogger(__name__)

        # Hack print
        def use_logger(msg, level="info"):
            if self.rank != 0:
                return
            if level == "info":
                self.logger.info(msg)
            elif level == "warning":
                self.logger.warning(msg)
            else:
                raise ("Not implemente {} level log".format(level))

        builtins.print = use_logger

        return self.logger

    def _init_dataloader(self, config):
        """
        Init the Test dataloader.

        Args:
            config (dict): Parsed config file.

        Returns:
            Dataloader: Test_loader.
        """
        self._check_data_config()
        distribute = self.distribute
        test_loader = get_dataloader(config, "test", self.model_type, distribute)

        return test_loader

    def _check_data_config(self):
        """
        Check the config params.
        """
        # check: episode_size >= n_gpu and episode_size != 0
        assert (
            self.config["episode_size"] >= self.config["n_gpu"]
            and self.config["episode_size"] != 0
        ), "episode_size {} should be >= n_gpu {} and != 0".format(
            self.config["episode_size"], self.config["n_gpu"]
        )

        # check: episode_size % n_gpu == 0
        assert (
            self.config["episode_size"] % self.config["n_gpu"] == 0
        ), "episode_size {} % n_gpu {} != 0".format(
            self.config["episode_size"], self.config["n_gpu"]
        )

        # check: episode_num % episode_size == 0
        # assert (
        #     self.config["train_episode"] % self.config["episode_size"] == 0
        # ), "train_episode {} % episode_size  {} != 0".format(
        #     self.config["train_episode"], self.config["episode_size"]
        # )

        assert (
            self.config["test_episode"] % self.config["episode_size"] == 0
        ), "test_episode {} % episode_size  {} != 0".format(
            self.config["test_episode"], self.config["episode_size"]
        )

    def _init_model(self, config):
        """
        Init model (backbone+classifier) from the config dict and load the best checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        emb_func = get_instance(arch, "backbone", config)
        model_kwargs = {
            "way_num": config["way_num"],
            "shot_num": config["shot_num"] * config["augment_times"],
            "query_num": config["query_num"],
            "test_way": config["test_way"],
            "test_shot": config["test_shot"] * config["augment_times"],
            "test_query": config["test_query"],
            "emb_func": emb_func,
            "device": self.device,
        }
        model = get_instance(arch, "classifier", config, **model_kwargs)

        print(model)
        print("Trainable params in the model: {}.".format(count_parameters(model)))
        print("Loading the state dict from {}.".format(self.state_dict_path))
        state_dict = torch.load(self.state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)

        # 加载检查点
        checkpoint = torch.load(self.state_dict_path, map_location="cpu")

        # 兼容不同的保存格式
        if isinstance(checkpoint, dict) and 'params' in checkpoint:
            # GitHub下载的模型格式：包含'params'键
            print("检测到GitHub模型格式，从'params'键加载参数")
            state_dict = checkpoint['params']
        else:
            # 原有的直接保存格式
            print("检测到标准模型格式")
            state_dict = checkpoint

        # 处理参数名称映射（GitHub模型使用encoder.前缀，本地期望emb_func.前缀）
        new_state_dict = {}
        needs_mapping = False

        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                # 将encoder.替换为emb_func.
                new_key = key.replace('encoder.', 'emb_func.')
                new_state_dict[new_key] = value
                needs_mapping = True
            else:
                new_state_dict[key] = value

        if needs_mapping:
            print("应用参数名称映射：encoder.* -> emb_func.*")
            state_dict = new_state_dict

        # 加载模型参数
        try:
            model.load_state_dict(state_dict)
            print("模型参数加载成功")
        except Exception as e:
            print(f"严格模式加载失败: {e}")
            print("尝试非严格模式加载...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"缺少的键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"意外的键: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            print("模型参数加载成功（非严格模式）")

        if self.distribute:
            # higher order grad of BN in multi gpu will conflict with syncBN
            # FIXME MAML with multi GPU is conflict with syncBN
            if not (
                self.config["classifier"]["name"] in ["MAML"]
                and self.config["n_gpu"] > 1
            ):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            else:
                print(
                    "{} with multi GPU will conflict with syncBN".format(
                        self.config["classifier"]["name"]
                    ),
                    level="warning",
                )
            model = model.to(self.rank)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True,
            )

            return model, model.module.model_type
        else:
            model = model.to(self.device)

            return model, model.model_type

    def _init_device(self, rank, config):
        """
        Init the devices from the config file.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of devices and list_ids.
        """
        init_seed(config["seed"], config["deterministic"])
        device, list_ids = prepare_device(
            rank,
            config["device_ids"],
            config["n_gpu"],
            backend="nccl"
            if "dist_backend" not in self.config
            else self.config["dist_backend"],
            dist_url="tcp://127.0.0.1:" + str(config["port"])
            if "dist_url" not in self.config
            else self.config["dist_url"],
        )
        torch.cuda.set_device(self.rank)

        return device, list_ids

    def _init_meter(self):
        """
        Init the AverageMeter of test stage to cal avg... of batch_time, data_time, calc_time and acc.

        Returns:
            AverageMeter: Test_meter.
        """
        test_meter = AverageMeter(
            "test", ["batch_time", "data_time", "calc_time", "acc"], self.writer
        )

        return test_meter

    def _init_writer(self, viz_path):
        """
        Init the tensorboard writer.

        Return:
            writer: tensorboard writer
        """
        if self.rank == 0:
            writer = TensorboardWriter(viz_path)
            return writer
        else:
            return None

    def _init_ensemble(self, config):
        """
        Init ensemble way from the config dict

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the ensemble way instance and ensemble way name (str)
        """
        ensemble_kwargs = config["ensemble_kwargs"]["other"]
        ensemble_kwargs["device"]= self.device
        ensemble_kwargs["way_num"]= config["way_num"]
        ensemble_kwargs["shot_num"]= config["shot_num"] * config["augment_times"]
        ensemble_kwargs["query_num"]= config["query_num"]
        ensemble_kwargs["test_way"]= config["test_way"]
        ensemble_kwargs["test_shot"]= config["test_shot"] * config["augment_times"]
        ensemble_kwargs["test_query"]= config["test_query"]
        name = config["ensemble_kwargs"]["name"]
    # if ensemble_kwargs["pretrain_encoder_path"] is None:
        #     ensemble_kwargs["emb_func"] = get_instance(arch, "backbone", ensemble_kwargs)
        # else:
        #     ensemble_kwargs["emb_func"] = get_instance(arch, "backbone", ensemble_kwargs)
        #     ensemble_kwargs["emb_dict"] = os.path.join(ensemble_kwargs["pretrain_model_path"], "checkpoints", "emb_func_best.pth")


        import core.ensemble as ways
        ensemble = get_instance(ways, "ensemble_kwargs", config, **ensemble_kwargs)

        return ensemble, name
