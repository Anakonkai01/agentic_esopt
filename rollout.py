def rollout(env, agent, verbose=False):
    observation = env.reset()

    trajectory = []
    total_reward = 0.0

    done = False

    while not done:
        action, raw_response = agent.act(observation)

        # Model không generate được action parseable
        if action is None:
            env.steps_taken += 1

            timeout = env.steps_taken >= env.max_steps

            trajectory.append({
                "observation": observation,
                "raw_response": raw_response,
                "action": None,
                "reward": 0.0,
                "invalid": True,
            })

            done = timeout

            if verbose:
                print("\nMODEL:")
                print(raw_response)
                print("PARSED ACTION: None")

            continue

        next_observation, reward, done, info = env.step(action)

        trajectory.append({
            "observation": observation,
            "raw_response": raw_response,
            "action": action,
            "reward": reward,
            "info": info,
        })

        if verbose:
            print("\n======================")
            print("Observation:")
            for row in observation:
                print(row)

            print("\nQwen:")
            print(raw_response)

            print("\nAction:")
            print(action)

            print("\nReward:")
            print(reward)

            print("Info:")
            print(info)

        observation = next_observation
        total_reward += reward

    return {
        "reward": total_reward,
        "success": total_reward > 0,
        "trajectory": trajectory,
        "length": len(trajectory),
    }
    
    
    
def evaluate_population(
    env_fn,
    agent,
    es_optimizer,
    population_size,
):
    seeds = es_optimizer.sample_seeds(
        population_size
    )

    rewards = []

    for seed in seeds:
        es_optimizer.apply_perturbation(seed)

        try:
            env = env_fn()
            
            res = rollout(
                env=env,
                agent=agent,
                verbose=False,
            )
            reward = res["reward"]
        finally:
            es_optimizer.revert_perturbation(seed)

        rewards.append(reward)

    return seeds, rewards 