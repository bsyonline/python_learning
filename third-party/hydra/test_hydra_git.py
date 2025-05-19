import subprocess

import hydra
from omegaconf import DictConfig, OmegaConf
import os

def get_git_info():
    """获取当前Git commit信息"""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            cwd=os.getcwd()
        ).strip().decode('utf-8')
        return {
            'commit': commit_hash,
            'branch': subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=os.getcwd()
            ).strip().decode('utf-8')
        }
    except Exception as e:
        print(f"Warning: Unable to get git info: {e}")
        return None

def load_config_from_git(config_path):
    """从Git仓库加载配置文件"""
    try:
        # 获取配置文件内容
        config_content = subprocess.check_output(
            ['git', 'show', f'HEAD:{config_path}'],
            cwd=os.getcwd()
        ).decode('utf-8')
        return OmegaConf.create(config_content)
    except Exception as e:
        print(f"Warning: Unable to load config from git: {e}")
        return None

@hydra.main(config_path='conf', config_name='hydra_config.yaml')
def train(cfg: DictConfig):
    # 打印配置信息
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # 获取Git信息
    git_info = get_git_info()
    if git_info:
        print("\nGit Info:")
        print(f"Commit: {git_info['commit']}")
        print(f"Branch: {git_info['branch']}")

        # 保存Git信息到配置文件
        with open('git_info.txt', 'w') as f:
            f.write(f"Commit: {git_info['commit']}\n")
            f.write(f"Branch: {git_info['branch']}\n")
    
    # 从Git加载配置示例
    git_config = load_config_from_git('conf/hydra_config.yaml')
    if git_config:
        print("\nConfig from Git:")
        print(OmegaConf.to_yaml(git_config))

    # 保存配置
    # output_dir = os.getcwd()  # Hydra 会自动切换到输出目录
    # OmegaConf.save(cfg, os.path.join(output_dir, "merged_config.yaml"))
    # print(f"\nConfiguration saved to: {output_dir}/merged_config.yaml")
    
if __name__ == '__main__':
    train()
 