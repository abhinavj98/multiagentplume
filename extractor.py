# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#
#
#
# class CombinedExtractor(BaseFeaturesExtractor):
#     """
#     Combined features extractor for Dict observation spaces.
#     Builds a features extractor for each key of the space. Input from each space
#     is fed through a separate submodule (CNN or MLP, depending on input shape),
#     the output features are concatenated and fed through additional MLP network ("combined").
#
#     :param observation_space:
#     :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
#         256 to avoid exploding network sizes.
#     :param normalized_image: Whether to assume that the image is already normalized
#         or not (this disables dtype and bounds checks): when True, it only checks that
#         the space is a Box and has 3 dimensions.
#         Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
#     """
#
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         cnn_output_dim: int = 256,
#         normalized_image: bool = False,
#     ) -> None:
#         # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
#         super().__init__(observation_space, features_dim=1)
#
#         extractors: Dict[str, nn.Module] = {}
#
#         total_concat_size = 0
#         for key, subspace in observation_space.spaces.items():
#             if is_image_space(subspace, normalized_image=normalized_image):
#                 extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
#                 total_concat_size += cnn_output_dim
#             else:
#                 # The observation key is a vector, flatten it if needed
#                 extractors[key] = nn.Flatten()
#                 total_concat_size += get_flattened_obs_dim(subspace)
#
#         self.extractors = nn.ModuleDict(extractors)
#
#         # Update the features dim manually
#         self._features_dim = total_concat_size
#
#     def forward(self, observations: TensorDict) -> th.Tensor:
#         encoded_tensor_list = []
#
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         return th.cat(encoded_tensor_list, dim=1)
#
#
# class MultiAgentFeaturesExtractor(BaseFeaturesExtractor):
#     """Similar to CombinedExtractor but for multi-agent environments. Each agent has its own features conacatenated first
#     and then all agents are concatenated together. THis is useful for multi-agent environments where each agent has its own
#     observation space. For example keys starting with 'agent_0', 'agent_1', etc."
#     """
#
#         def __init__(
#             self,
#             observation_space: spaces.Dict,
#             features_dim: int = 256,
#             normalized_image: bool = False,
#         ) -> None:
#             super().__init__(observation_space, features_dim=features_dim)
#
#             agent_extractors: Dict[str, nn.Module] = {}
#             agent_features_dim: Dict[str, int] = {}
#
#             for agent_id, agent_space in observation_space.spaces.items():
#                 agent_extractor = CombinedExtractor(agent_space, cnn_output_dim=features_dim, normalized_image=normalized_image)
#                 agent_extractors[agent_id] = agent_extractor
#                 agent_features_dim[agent_id] = agent_extractor.features_dim
#
#             self.agent_extractors = nn.ModuleDict(agent_extractors)
#             self.agent_features_dim = agent_features_dim
#
#             # Update the features dim manually
#             self._features_dim = sum(agent_features_dim.values())
#
#         def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
#             agent_encoded_tensor_list = []
#
#             for agent_id, extractor in self.agent_extractors.items():
#                 agent_encoded_tensor_list.append(extractor(observations[agent_id]))
#
#             return th.cat(agent_encoded_tensor_list, dim=1)
#