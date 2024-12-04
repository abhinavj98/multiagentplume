from typing import Dict, List, Optional, Tuple, Type, Union
from torch import nn
import torch as th
from gym import spaces
from stable_baselines3.common.torch_layers import MlpExtractor
class MultiAgentMLPExtractor(nn.Module):
    """Initilizes multiper MLP extractors and concatenates the output of each extractor."""

    def __init__(
        self,
        features_dim: int,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        num_agents: int = 1,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        self.num_agents = num_agents
        self.features_dim = features_dim
        self.features_dim_per_agent = self.features_dim // self.num_agents
        self.extractors = nn.ModuleList([MlpExtractor(self.features_dim_per_agent, net_arch, activation_fn, device) for _ in range(self.num_agents)])
        # Save dim, used to create the distributions
        self.latent_dim_pi = self.extractors[0].latent_dim_pi*self.num_agents
        self.latent_dim_vf = self.extractors[0].latent_dim_vf*self.num_agents
        # self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.latent_dim_pi)
        # self.action_net = nn.ModuleList([self.action_dist.proba_distribution_net(latent_dim=self.latent_dim_pi) for _ in range(self.num_agents)])
        # self.value_net = nn.ModuleList([nn.Linear(self.latent_dim_vf, 1) for _ in range(self.num_agents)])

    def forward(self, observations: th.Tensor) -> th.Tensor:
        actor = self.forward_actor(observations)
        critic = self.forward_critic(observations)
        return actor, critic
    def forward_actor(self, observations: th.Tensor) -> th.Tensor:
        observations_per_agent = th.split(observations, self.features_dim_per_agent, dim=1)
        # print(observations_per_agent.shape)
        extracted_features = [extractor.forward_actor(obs) for obs, extractor in zip(observations_per_agent, self.extractors)]
        assert len(extracted_features) == self.num_agents
        return th.cat(extracted_features, dim=1)

    def forward_critic(self, observations: th.Tensor) -> th.Tensor:
        observations_per_agent = th.split(observations, self.features_dim_per_agent, dim=1)
        extracted_features = [extractor.forward_critic(obs) for obs, extractor in zip(observations_per_agent, self.extractors)]
        return th.cat(extracted_features, dim=1)

# class MultiAgentActionNet(nn.Module):
#     def __init__(self, latent_dim: int, action_space: spaces.Space, num_agents: int = 1):
#         super().__init__()
#         self.num_agents = num_agents
#         self.action_space = action_space
#         self.action_nets = nn.ModuleList([self.action_dist.proba_distribution_net(latent_dim=latent_dim) for _ in range(self.num_agents)])
#
#     def forward(self, latent_pi: th.Tensor) -> th.Tensor:
#         observations_per_agent = th.split(latent_pi, self.num_agents, dim=1)

class MultiAgentNet(nn.Module):
    def __init__(self, latent_dim: int, output_dim, num_agents: int = 1):
        super().__init__()
        self.num_agents = num_agents
        self.latent_dim = latent_dim
        self.value_nets = nn.ModuleList([nn.Linear(latent_dim//self.num_agents, output_dim) for _ in range(self.num_agents)])

    def forward(self, latent_vf: th.Tensor) -> th.Tensor:
        observations_per_agent = th.split(latent_vf, self.latent_dim//self.num_agents, dim=1)
        return th.cat([value_net(obs) for obs, value_net in zip(observations_per_agent, self.value_nets)], dim=1)

class MultiAgentValueNet(nn.Module):
    def __init__(self, latent_dim: int, output_dim, num_agents: int = 1):
        super().__init__()
        self.num_agents = num_agents
        self.latent_dim = latent_dim
        self.value_net = nn.Linear(latent_dim, output_dim)#nn.ModuleList([nn.Linear(latent_dim//self.num_agents, output_dim) for _ in range(self.num_agents)])

    def forward(self, latent_vf: th.Tensor) -> th.Tensor:
        # observations_per_agent = th.split(latent_vf, self.latent_dim//self.num_agents, dim=1)
        return self.value_net(latent_vf) #h.mean(th.cat([value_net(obs) for obs, value_net in zip(observations_per_agent, self.value_nets)], dim=1), dim = 1)