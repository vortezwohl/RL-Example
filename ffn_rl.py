import torch
from torch import nn
from deeplotx import MultiHeadFeedForward
from stable_baselines3.common.policies import ActorCriticPolicy


class MyActorCriticPolicy(nn.Module):
    def __init__(self, feature_dim: int, policy_output_dim: int, value_output_dim: int, device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.latent_dim_pi = policy_output_dim
        self.latent_dim_vf = value_output_dim
        self.policy_net = nn.Sequential(
            MultiHeadFeedForward(feature_dim=feature_dim, num_heads=3, device=device, dtype=dtype),
            nn.Linear(in_features=feature_dim, out_features=policy_output_dim, device=torch.device(device), dtype=dtype)
        )
        self.value_net = nn.Sequential(
            MultiHeadFeedForward(feature_dim=feature_dim, num_heads=3, device=device, dtype=dtype),
            nn.Linear(in_features=feature_dim, out_features=value_output_dim, device=torch.device(device), dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net.forward(x), self.value_net.forward(x)

    def forward_actor(self, x: torch.Tensor):
        return self.policy_net.forward(x)

    def forward_critic(self, x: torch.Tensor):
        return self.value_net.forward(x)


class MyRLModel(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MyActorCriticPolicy(self.features_dim, 64, 64)


# %%
import gymnasium
from stable_baselines3 import PPO

env = gymnasium.make("CartPole-v1", render_mode="human")
ppo = PPO(MyRLModel, env, verbose=1)
ppo.learn(total_timesteps=3000, progress_bar=True)
# %%
vec_env = ppo.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = ppo.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    if done:
        obs = vec_env.reset()
