import minerl
import gym

env = gym.make("MineRLObtainDiamond-v0")

try:
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.noop()
        action['forward'] = 1
        action['attack'] = 1
        obs, reward, done, _ = env.step(action)
        env.render()
except KeyboardInterrupt:
    print("MineRL Interrupted! Closing...")
finally:
    env.close()
    print("Environment closed cleanly.")