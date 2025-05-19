import hydra
from omegaconf import DictConfig, OmegaConf

# python3 hydra_load_conf_mutil_env.py --multirun model=qwen2_7b,qwen2_14b dataset=wudao,alpaca
@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()