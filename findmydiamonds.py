import minerl
import gym
import logging
import time
import random

logging.basicConfig(level=logging.INFO)
env = gym.make("MineRLObtainDiamondShovel-v0")

def has_item(obs, item_name, count=1):
    return int(obs['inventory'].get(item_name, 0)) >= count

try:
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print("\n--- Simple Agent: Collect Logs & Craft Sticks ---\n")

    mining_steps = 0  # 0 means not currently mining

    while not done:
        action = env.action_space.no_op()
        inventory = obs["inventory"]

        if not has_item(obs, "log", 2) and mining_steps == 0:
            mining_steps = random.randint(20, 40) 

        # --- Mining mode ---
        if mining_steps > 0:
            action["forward"] = 1
            action["attack"] = 1
            # action["camera"] = [random.uniform(-1, 1), random.uniform(-0.5, 0.5)]
            mining_steps -= 1

        # --- Crafting logic ---
        elif any("log" in k for k in inventory):
            action["craft"] = "planks" if not has_item(obs, "planks", 2) else "stick"

        # --- Occasional jump to avoid terrain traps ---
        if random.random() < 0.02:
            action["jump"] = 1

        obs, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward
        step_count += 1

        if step_count % 10 == 0:
            non_zero = {k: int(v) for k, v in inventory.items() if int(v) > 0}
            print(f"[Step {step_count}] Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print("Inventory:", non_zero)
            print("Mining steps left:", mining_steps)
            print("-" * 40)

except KeyboardInterrupt:
    print("Interrupted! Closing...")
finally:
    env.close()
    print("Environment closed.")


    