from stable_baselines3.ppo.policies import ActorCriticPolicy
from cbp import CBP

class CPPO_Policy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        **kwargs
    ):
        super(CPPO_Policy, self).__init__(observation_space, action_space, lr_schedule, **(kwargs|{'optimizer_kwargs':{}})) # remove optimizer_kwargs, as __init__ initializes Adam with them, which throws errors
        self.optimizer_kwargs = kwargs['optimizer_kwargs']
        assert len(self.mlp_extractor.shared_net) == 0, 'no shared layers between policy and value function allowed' # not used in the paper, might try to implement it later
        
        policy_linears, policy_activations = self._handle_sequential(self.mlp_extractor.policy_net)
        value_linears, value_activations = self._handle_sequential(self.mlp_extractor.value_net)
        
        self.optimizer = CBP(self.parameters(),
                             linear_layers=[policy_linears, value_linears],
                             activation_layers=[policy_activations, value_activations],
                             output_linears=[self.action_net, self.value_net],
                             lr=lr_schedule(1),
                             **self.optimizer_kwargs)
    
    def _handle_sequential(self, sequential):
        linears = []
        activations = []
        for i, layer in enumerate(sequential):
            if i%2 == 0: # Linear Layer
                linears.append(layer)
            else:        # Activation Layer
                activations.append(layer)
        return linears, activations