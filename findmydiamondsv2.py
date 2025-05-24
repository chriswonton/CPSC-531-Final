class InventoryRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.milestones = [
            'log', 'planks', 'stick', 'crafting_table', 'wooden_pickaxe',
            'cobblestone', 'furnace', 'stone_pickaxe', 'iron_ore',
            'iron_ingot', 'iron_pickaxe', 'diamond', 'diamond_shovel'
        ]
        self.prev_inv = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # copy of the starting inventory counts
        self.prev_inv = dict(obs['inventory'])
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = 0.0
        inv = obs['inventory']
        # +1 for every milestone whose count just increased
        for item in self.milestones:
            if inv.get(item, 0) > self.prev_inv.get(item, 0):
                reward += 1.0
        self.prev_inv = dict(inv)
        return obs, reward, done, info

# build your custom-wrapped, time-limited env
def make_env():
    base = gym.make('MineRLObtainDiamondShovel-v0')
    # enforce 18000 max steps if needed
    base = TimeLimit(base, max_episode_steps=18000)
    return InventoryRewardWrapper(base)

env = DummyVecEnv([make_env])