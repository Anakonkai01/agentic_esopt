import numpy as np
import torch




# objective -(theta - 5)^2
def objective(theta): 
    """
    compute objective 
    input: theta/weight R^d 
    return a scalar value 
    """
    return -np.sum((theta - 5)**2)



# noise sampling 
def noise_sampling(seed, dim):
    rng = np.random.default_rng(seed=seed)
    return rng.normal(
        loc=0.0,
        scale=1.0,
        size= dim
    )


# population evaluation 
def g_hat_estimator(rewards, seeds, G, theta_dim): 
    direction = np.zeros(shape=theta_dim)
    r_z_score = reward_normalization(rewards) # (G, )
    for i, seed in enumerate(seeds):

        epsilons = noise_sampling(seed=seed, dim=theta_dim) # (d)
        direction += epsilons * r_z_score[i]

    return 1 / G * direction # (d)
    

# reward normalization 
def reward_normalization(rewards): 
    rewards = np.array(rewards)
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    return (rewards - mean_r) / (std_r + 0.001)

def apply_perturbation(theta, sigma, seed, theta_dim):
    epsilon_vectors = noise_sampling(seed=seed, dim=theta_dim)
    perturbed_theta = theta + sigma * epsilon_vectors
    
    return perturbed_theta
    
def revert_perturbation(theta, sigma, seed, theta_dim):
    epsilon_vectors = noise_sampling(seed=seed, dim=theta_dim)
    reversed_theta = theta - sigma * epsilon_vectors
    
    return reversed_theta
 
# training loop 
def train(theta, G, T, sigma=0.01, lr=0.1, base_seed=0): 
    theta_dim = theta.shape[0]
    for t in range(T): 
        # sample epsilon 
        rewards = []
        seeds = []
        for g in range(G): 
            seed = base_seed + t * G + g
            seeds.append(seed)
            perturbed_theta = apply_perturbation(theta=theta, sigma=sigma, seed=seed, theta_dim=theta_dim) # (d)
            revert_theta = revert_perturbation(theta=perturbed_theta, sigma=sigma, seed=seed, theta_dim=theta_dim)
            print(np.allclose(revert_theta, theta))
            
            rewards.append(objective(perturbed_theta))
            
        update_direction = g_hat_estimator(rewards=rewards, seeds=seeds, G=G, theta_dim=theta_dim)
            
            
        
        print(theta, update_direction)
        theta = theta + lr * update_direction
        
    return theta 
            
# main / config



def main(): 
    theta = np.array([1, 2, 3])
    G = 100
    T = 100
    new_theta = train(theta, G=G, T=T)
    
    
    
main()