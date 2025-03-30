import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(config_path='conf', config_name='hydra_config.yaml')
def train(cfg: DictConfig):
    # 打印配置信息
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # 保存配置
    # output_dir = os.getcwd()  # Hydra 会自动切换到输出目录
    # OmegaConf.save(cfg, os.path.join(output_dir, "merged_config.yaml"))
    # print(f"\nConfiguration saved to: {output_dir}/merged_config.yaml")
    
if __name__ == '__main__':
    train()
 