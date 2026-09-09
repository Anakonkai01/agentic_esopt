import torch 
import torch.nn as nn 
import math 

class ESOptimizer: 
    def __init__(self, model, sigma=1e-3, alpha=5e-4, seed=0): 
        self.model = model
        self.sigma = sigma
        self.alpha = alpha
        
        self.seed_generator = torch.Generator(device="cpu")
        self.seed_generator.manual_seed(seed)
        
   
    def sample_seeds(self, population_size): 
        seeds = torch.randint(
            low=0, 
            high= 2 ** 32 - 1,
            size= (population_size,),
            generator=self.seed_generator,
            dtype=torch.int64,
        )
        return seeds.tolist()
    
    def _make_noise_generator(self, seed, device): 
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator
    
    def apply_perturbation(self, seed, sigma=None): 
        if sigma is None: 
            sigma = self.sigma
        
        device = next(self.model.parameters()).device
        generator = self._make_noise_generator(
            seed=seed,
            device=device
        )

        with torch.inference_mode(): 
            for param in self.model.parameters(): 
                eps = torch.randn_like(param, generator=generator)
                param.add_(eps * sigma)

    def revert_perturbation(self, seed, sigma=None): 
        if sigma is None:
            sigma = self.sigma

        device = next(self.model.parameters()).device

        generator = self._make_noise_generator(
            seed=seed,
            device=device
        )        

        with torch.inference_mode(): 
            for param in self.model.parameters(): 
                eps = torch.randn_like(param, generator=generator)
                param.sub_(eps * sigma)
                
    def normalize_rewards(self, rewards): 
        # rewards: list 
        rewards = torch.tensor(rewards, dtype=torch.float32)

        mean = torch.mean(rewards)
        std = torch.std(rewards, correction=0)

        norm = (rewards - mean) / (std + 1e-8)
        return norm
    
    def apply_es_update(self, seeds, normalized_rewards): 
        with torch.inference_mode():
            for i, seed_i in enumerate(seeds): 
                self.seed_generator.manual_seed(seed_i)
                for param in self.model.parameters(): 
                    eps = torch.randn_like(param, generator=self.seed_generator)
                    d = 1 / (len(seeds)) * self.alpha * eps * normalized_rewards[i]
                    param.add_(d)
                
                
    def step(self, seeds, rewards): 
        normalized_rewards = self.normalize_rewards(rewards)
        self.apply_es_update(seeds, normalized_rewards)
        return {
            "raw_rewards": rewards,
            "normalized_rewards": normalized_rewards,
        }
    
    
    @staticmethod 
    def cosine_sigma(step, total_steps, sigma_start, sigma_end): 
        return sigma_end + (sigma_start - sigma_end) * (1 + math.cos(math.pi * step/total_steps)) / 2
    
