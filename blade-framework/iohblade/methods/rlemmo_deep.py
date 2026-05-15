from iohblade.method import Method
from iohblade.solution import Solution
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch

class IOHBladeLoggerCallback(BaseCallback):
    """
    Custom callback to link Stable Baselines 3 training progress 
    directly to the iohblade ExperimentLogger.
    """
    def __init__(self, problem, verbose=0):
        super().__init__(verbose)
        self.problem = problem
        self.best_f = float('inf')

    def _on_step(self) -> bool:
        try:
            # Ask all 10 parallel environments for their current best score
            best_fs = self.training_env.get_attr('best_f')
            current_best = min(best_fs)
            
            # If we found a new global best across all cores, log it!
            if current_best < self.best_f:
                self.best_f = current_best
                if hasattr(self.problem, 'logger') and self.problem.logger:
                    # iohblade maximizes, so we pass negative fitness
                    sol = Solution(code=f"# DeepRLEMMO step {self.num_timesteps}")
                    sol.set_scores(-self.best_f)
                    self.problem.logger.log_individual(sol)
        except Exception:
            # Fail silently so logging issues don't crash the heavy training job
            pass
        return True


class DeepRLEMMO(Method):
    """
    Deep RL Implementation of RLEMMO using Stable Baselines 3 (PPO).
    """
    def __init__(self, problem_builder, num_envs=10, pop_size=100, name="DeepRLEMMO", **kwargs):
        super().__init__(None, budget=1, name=name)
        self.problem_builder = problem_builder
        self.num_envs = num_envs
        self.pop_size = pop_size

    def __call__(self, problem) -> Solution:
        print(f"[{self.name}] Initializing Vectorized Environments...")
        
        def env_maker():
            from lens_rl_env import LensPopulationEnv
            return LensPopulationEnv(self.problem_builder, pop_size=self.pop_size)
            
        vec_env = make_vec_env(env_maker, n_envs=self.num_envs)

        print(f"[{self.name}] Initializing PPO Agent on GPU/CPU...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tensorboard log removed, using custom callback instead
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1,
            device=device,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4
        )

        # Initialize our custom IOHBlade logger hook
        logger_callback = IOHBladeLoggerCallback(problem)

        total_timesteps = problem.budget_factor 
        print(f"[{self.name}] Commencing Training/Search for {total_timesteps} steps...")
        
        # Pass the callback into the learning loop
        model.learn(total_timesteps=total_timesteps, callback=logger_callback)

        best_fs = vec_env.get_attr('best_f')
        best_f = min(best_fs)

        
        res_sol = Solution(code=f"# DeepRLEMMO Search\n# Best Fitness: {best_f}")
        res_sol.set_scores(-best_f, feedback=f"DeepRLEMMO best fitness: {best_f}")
        return res_sol

    def to_dict(self):
        """Required by the iohblade framework for logging."""
        return {
            "method_name": self.name,
            "budget": self.budget,
            "pop_size": self.pop_size,
            "num_envs": self.num_envs
        }