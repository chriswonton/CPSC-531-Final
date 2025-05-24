import gym
from gym.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv

class InventoryRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.milestones = {
            'log': 1,
            'planks': 2,
            'stick': 4,
            'crafting_table': 4,
            'wooden_pickaxe': 8,
            'cobblestone': 16,
            'furnace': 32,
            'stone_pickaxe': 32,
            'iron_ore': 64,
            'iron_ingot': 128,
            'iron_pickaxe': 256,
            'diamond': 1024,
            'diamond_shovel': 2045
        }
        self.prev_inv = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_inv = dict(obs['inventory'])
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = 0.0
        inv = obs['inventory']
        for item, value in self.milestones.items():
            if inv.get(item, 0) > self.prev_inv.get(item, 0):
                reward += value
        self.prev_inv = dict(inv)
        return obs, reward, done, info

def make_env():
    base = gym.make('MineRLObtainDiamondShovel-v0')
    base = TimeLimit(base, max_episode_steps=18000)
    return InventoryRewardWrapper(base)

env = DummyVecEnv([make_env])
