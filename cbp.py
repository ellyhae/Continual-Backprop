import torch
from cadam import CAdam
import math
from stable_baselines3.common.utils import get_device

class CBP(CAdam):
    '''
    Open questions:
        How should batches be dealth with?
            For now I calculate the mean over the batch and handle that like in the sequential case
        How many features are actually replaced every iteration? Their n_l and rho don't seem to work, as 256 * 10**-4 < 1. Is this supposed to be a probability?
            # For now math.ceil is used, so every iteration 1 usit is replaced. This doesn't make sense, since when n_l < m then the features are just replaced in order as they mature.
            Changed to using n_l * rho as a probability of replacing the worst performing feature
    '''
    def __init__(self,
                 params,                 # all parameters to be optimized by Adam
                 linear_layers,          # List[List[Linear]], a list of linearities for each separate network (policy, value, ...), in the order they are executed
                 activation_layers,      # List[List[Activation]], a list of activation layers for each separate network (policy, value, ...), in the order they are executed. Forward hooks are added to these
                 output_linears,         # List[Linear], a list of each network's last Linear layer
                 eta=0.99,               # running average discount factor
                 m=int(5e3),             # maturity threshold, only features with age > m are elligible to be replaced
                 rho=10**-4,             # replacement rate, controls how frequently features are replaced                                                    # TODO: change description
                 sample_weights=None,    # functiion, take size and device as input and return a tensor of the given size with newly initialized weights
                 eps=1e-8,               # small additive value to avoid division by zero
                 device = 'auto',
                 **kwargs):
        super(CBP, self).__init__(params, **kwargs)
        self.linear_layers = linear_layers
        self.activation_layers = activation_layers
        self.cbp_vals = {}
        self.output_linears = output_linears
        self.eta = eta
        self.m = m
        self.rho = rho
        
        self.dev = get_device(device)
        
        assert len(self.linear_layers) == len(self.activation_layers)
        for linears, activations in zip(self.linear_layers, self.activation_layers):
            self._add_hooks(linears, activations)
        
        if sample_weights is None:
            def sample_weights(size, device):
                sample = torch.empty(size, device=device)
                torch.nn.init.kaiming_uniform_(sample, a=math.sqrt(5))
                return sample
            reset_weights = lambda w: torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.sample_weights = sample_weights
        self.eps = eps
    
    @torch.no_grad()
    def step(self):
        super(CBP, self).step()
        for linears, output_linear in zip(self.linear_layers, self.output_linears): # cycle through models
            # TODO change such that abs is not called twice on each weight Tensor
            for current_linear, next_linear in zip(linears, linears[1:] + [output_linear]): # cycle through layers
                cbp_vals = self.cbp_vals[current_linear]
                
                pre_w = current_linear.weight.abs().sum(1).detach_().add_(self.eps) # avoid division by zero
                post_w = next_linear.weight.abs().sum(0).detach_()
                
                y = (cbp_vals['h'] - cbp_vals['fhat']).abs_().mul_(post_w).div_(pre_w)
                cbp_vals['u'].mul_(self.eta).add_((1-self.eta)*y)
                
                uhat = cbp_vals['u'] / (1 - self.eta**cbp_vals['age'])
                
                eligible = cbp_vals['age'] > self.m
                if eligible.any() and torch.rand(1) < len(uhat)*self.rho:  # use n_l* rho as a probability of replacing a single feature
                    ascending = uhat.argsort()
                    r = ascending[eligible[ascending]]   # sort eligible indices according to their utility
                    #r = r[:math.ceil(uhat.shape[0]*self.rho)]  # choose top k worst performing features    # using ceil because otherwise nothing ever gets reset int(256*10**-4)=0
                    r = r[[0]]  # choose the worst feature
                    
                    current_linear.weight.index_copy_(0, r, self.sample_weights((len(r), current_linear.weight.shape[1]), device=current_linear.weight.device))
                    next_linear.weight.index_fill_(1, r, 0.)
                    
                    cbp_vals['u'].index_fill_(0, r, 0.)
                    cbp_vals['f'].index_fill_(0, r, 0.)
                    cbp_vals['age'].index_fill_(0, r, 0)
                    
                    ### Adam resets
                    pre_state = self.state[current_linear.weight]
                    pre_state['step'].index_fill_(0, r, 0)
                    pre_state['exp_avg'].index_fill_(0, r, 0.)
                    pre_state['exp_avg_sq'].index_fill_(0, r, 0.)
                    
                    post_state = self.state[next_linear.weight]
                    post_state['step'].index_fill_(1, r, 0)
                    post_state['exp_avg'].index_fill_(1, r, 0.)
                    post_state['exp_avg_sq'].index_fill_(1, r, 0.)
                    
    def _hook_gen(self, linear_layer):
        num_units = linear_layer.weight.shape[0]
        self.cbp_vals[linear_layer] = {
            'age':  torch.zeros(num_units, dtype=int, device=self.dev), 
            'h':    torch.zeros(num_units, device=self.dev),
            'f':    torch.zeros(num_units, device=self.dev),
            'fhat': torch.zeros(num_units, device=self.dev),
            'u':    torch.zeros(num_units, device=self.dev)
        }
        def hook(mod, inp, out):
            if mod.training:
                with torch.no_grad():
                    # NOTE Seems CBP is only described for sequential input with gradient updates at each step.
                    #      Since PPO is based on batched environment data, changes have to be made
                    #      I will therefore work with means over the baches
                    cbp_vals = self.cbp_vals[linear_layer]
                    cbp_vals['age'].add_(1)
                    cbp_vals['h'] = out.mean(0).detach_()
                    cbp_vals['fhat'] = self.cbp_vals[linear_layer]['f'] / (1 - self.eta**cbp_vals['age'])
                    cbp_vals['f'].mul_(self.eta).add_((1-self.eta)*cbp_vals['h'])
        return hook
    
    def _add_hooks(self, linears, activations):
        assert len(linears) == len(activations
                                  )
        for lin, act in zip(linears, activations):
            act.register_forward_hook(self._hook_gen(lin))