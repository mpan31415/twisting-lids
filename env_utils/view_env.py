import isaacgym
import torch
import datetime
import hydra
import gym
import os
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


@hydra.main(config_name="config", config_path="../minimal_stable_PPO/configs")
def main(config: DictConfig):

    ###### define device ######
    # my_device = "cuda:0"
    my_device = "cpu"

    # use the same device for sim and rl
    config.sim_device = my_device
    config.rl_device = my_device
    config.seed = set_seed(config.seed)

    # We want to view the environment
    config.headless = False

    # small number of envs
    config.num_envs = 9

    print("Graphics device id", config.graphics_device_id)
    envs = isaacgym_task_map[config.task_name](
        cfg=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        rl_device=config.rl_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
        virtual_screen_capture=False,
        force_render=True,
    )

    print("Observation space is", envs.observation_space)
    print("Action space is", envs.action_space)

    obs = envs.reset()
    for _ in range(200000):
        random_actions = (
            2.0
            * torch.rand((config.num_envs,) + envs.action_space.shape, device=my_device)
            - 1.0
        )
        random_actions = torch.zeros_like(random_actions, device=my_device)
        envs.step(random_actions)


if __name__ == "__main__":
    main()
