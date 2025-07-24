# QuickBoost 使用指南

## 数据集准备

目前的实现只能在 miniImageNet-ravi 的数据集上进行训练和测试

请准备好该数据集（参考 Libfewshot 文档），并在配置文件中设置好对应的路径（`config/headers/data.yaml`）

## 预训练的事项准备

准备好 `./QuickBoost/` 目录下的 `name2idx_test.json` `embedding_resnet18_64classes_test.pkl` `embedding_resnet18_64classes.pkl` 文件，如果要自定义路径，修改 `config/QuickBoost.yaml` 中的对应路径

准备好要在测试时集成的模型并在 `config/QuickBoost.yaml` 中设置好对应路径，读取模型时会假设该路径下的文件结构和 Libfewshot 生成的 `results/model` 的文件结构一致，实际只需要使用到 `results/model/config.yaml` `results/model/checkpoints/model_best.pth` 这些文件 (直接用 `results/model` 作为 `pretrain_model_path` 是最好的)

> `pretrain_model_name` 这个配置只用在日志中，不影响实际结果

## 训练 QuickBoost

`run_trainer.py` 中 `Config` 内的路径改为 `./config/QuickBoost.yaml` (或其他自定义后的 QuickBoost 配置), 之后运行该文件，就会在结果路径（默认为`./results/`）中训练并保存对应的结果，包括配置和随机森林模型

## 测试 QuickBoost

`run_test.py` 中设置好 `PATH` 为上面的训练产生的结果文件夹路径，直接运行即可测试，其中：
- `test_standalone` 这个配置决定了是单独测试 QuickBoost (`True` 时) 还是测试和模型（`pretrain_model`）集成 model + QuickBoost (`False` 时)
- 可以直接修改结果路径中的 `config.yaml` 进行修改，也可以在 `run_test.py` 中修改 `config["ensemble_kwargs"]["other"]["test_standalone"]` 的值（后者会覆盖前者）
