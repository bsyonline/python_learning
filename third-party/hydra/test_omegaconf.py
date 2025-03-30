from omegaconf import OmegaConf

# 加载多个配置文件
dataset_config = OmegaConf.load("third-party/hydra/conf/dataset.yaml")
model_config = OmegaConf.load("third-party/hydra/conf/model.yaml")
evaluation_config = OmegaConf.load("third-party/hydra/conf/evaluation.yaml")

# 合并配置
merged_config = OmegaConf.merge(dataset_config, model_config, evaluation_config)
merged_config_yaml = OmegaConf.to_yaml(merged_config)


print(f"dataset name: {merged_config.dataset.name}")
print(f"model name: {merged_config.model.name}, for task: {merged_config.model.task}")
print(f"evaluation metrics: {', '.join(merged_config.evaluation.metrics)}")
print(merged_config_yaml)
# 保存合并后的配置
#OmegaConf.save(merged_config, "merged_config.yaml")