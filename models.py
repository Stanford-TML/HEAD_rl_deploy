import torch
import numpy as np
from typing import Optional, Union


class RunningMeanStd(torch.nn.Module):
    def __init__(self, dim: int, clamp: float=0):
        super().__init__()
        self.epsilon = 1e-5
        self.clamp = clamp
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, x, unnorm=False):
        mean = self.mean.to(torch.float32)
        var = self.var.to(torch.float32)+self.epsilon
        if unnorm:
            if self.clamp:
                x = torch.clamp(x, min=-self.clamp, max=self.clamp)
            return mean + torch.sqrt(var) * x
        x = (x - mean) * torch.rsqrt(var)
        if self.clamp:
            return torch.clamp(x, min=-self.clamp, max=self.clamp)
        return x
    
    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        var, mean = torch.var_mean(x, dim=0, unbiased=True)
        count = x.size(0)
        count_ = count + self.count
        delta = mean - self.mean
        m = self.var * self.count + var * count + delta**2 * self.count * count / count_
        self.mean.copy_(self.mean+delta*count/count_)
        self.var.copy_(m / count_)
        self.count.copy_(count_)

    def reset_counter(self):
        self.count.fill_(1)

class DiagonalPopArt(torch.nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, bias: torch.Tensor, momentum:float=0.1):
        super().__init__()
        self.epsilon = 1e-5

        self.momentum = momentum
        self.register_buffer("m", torch.zeros((dim,), dtype=torch.float64))
        self.register_buffer("v", torch.full((dim,), self.epsilon, dtype=torch.float64))
        self.register_buffer("debias", torch.zeros(1, dtype=torch.float64))

        self.weight = weight
        self.bias = bias

    def forward(self, x, unnorm=False):
        debias = self.debias.clip(min=self.epsilon)
        mean = self.m/debias
        var = (self.v - self.m.square()).div_(debias)
        if unnorm:
            std = torch.sqrt(var)
            return (mean + std * x).to(x.dtype)
        x = ((x - mean) * torch.rsqrt(var)).to(x.dtype)
        return x

    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        running_m = torch.mean(x, dim=0)
        running_v = torch.mean(x.square(), dim=0)
        new_m = self.m.mul(1-self.momentum).add_(running_m, alpha=self.momentum)
        new_v = self.v.mul(1-self.momentum).add_(running_v, alpha=self.momentum)
        std = (self.v - self.m.square()).sqrt_()
        new_std_inv = (new_v - new_m.square()).rsqrt_()

        scale = std.mul_(new_std_inv)
        shift = (self.m - new_m).mul_(new_std_inv)

        self.bias.data.mul_(scale).add_(shift)
        self.weight.data.mul_(scale.unsqueeze_(-1))

        self.debias.data.mul_(1-self.momentum).add_(1.0*self.momentum)
        self.m.data.copy_(new_m)
        self.v.data.copy_(new_v)


class Discriminator(torch.nn.Module):
    def __init__(self, disc_dim, latent_dim=256):
        super().__init__()
        self.rnn = torch.nn.GRU(disc_dim, latent_dim, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32)
        )
        if self.rnn is not None:
            i = 0
            for n, p in self.mlp.named_parameters():
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.)
                elif "weight" in n:
                    gain = 1 if i == 2 else 2**0.5 
                    torch.nn.init.orthogonal_(p, gain=gain)
                    i += 1
        self.ob_normalizer = RunningMeanStd(disc_dim)
        self.all_inst = torch.arange(0)
        
    def forward(self, s, seq_end_frame, normalize=True):
        if normalize: s = self.ob_normalizer(s)
        if self.rnn is None:
            s = s.view(s.size(0), -1)
        else:
            n_inst = s.size(0)
            if n_inst > self.all_inst.size(0):
                self.all_inst = torch.arange(n_inst, 
                    dtype=seq_end_frame.dtype, device=seq_end_frame.device)
            s, _ = self.rnn(s)
            s = s[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))]
        return self.mlp(s)

class ACModel(torch.nn.Module):

    class Critic(torch.nn.Module):
        def __init__(self, state_dim, goal_dim, value_dim=1, latent_dim=256, use_rnn=True, ob_horizon = 2,
                     smaller_nn=False, concate_s_g=False,
                     privilaged_obs_dim=0):
            super().__init__()
            goal_dim += privilaged_obs_dim # Add the privilaged observation
            if use_rnn:
                self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)
            else:
                self.rnn = None

            self.concate_s_g = concate_s_g
            
            if (not smaller_nn) and (not concate_s_g):
                embed_goal_dim = ob_horizon * state_dim if self.rnn is None else latent_dim
                mlp_input_dim = embed_goal_dim
            elif (not smaller_nn) and (concate_s_g):
                raise ValueError("A bad combination of smaller_nn and concate_s_g")
            elif smaller_nn and (not concate_s_g):
                embed_goal_dim = latent_dim if use_rnn else ob_horizon * state_dim
                mlp_input_dim = embed_goal_dim
            elif smaller_nn and concate_s_g:
                embed_goal_dim = latent_dim
                if use_rnn:
                    mlp_input_dim = embed_goal_dim + latent_dim
                else:
                    mlp_input_dim = ob_horizon * state_dim + latent_dim
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(mlp_input_dim, 1024 if smaller_nn else 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(1024 if smaller_nn else 2048, 512 if smaller_nn else 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(512 if smaller_nn else 2048, 256 if smaller_nn else 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(256 if smaller_nn else 1024, 256 if smaller_nn else 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(256 if smaller_nn else 1024, value_dim)
            )
            self.embed_goal = torch.nn.Sequential(
                torch.nn.Linear(goal_dim, latent_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(latent_dim, embed_goal_dim)
            )
            i = 0
            for n, p in self.mlp.named_parameters():
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.)
                elif "weight" in n:
                    torch.nn.init.uniform_(p, -0.0001, 0.0001)
                    i += 1
            self.all_inst = torch.arange(0)

        def forward(self, s, seq_end_frame, g=None):
            if self.rnn is None:
                s = s.view(s.size(0), -1)
            else:
                n_inst = s.size(0)
                if n_inst > self.all_inst.size(0):
                    self.all_inst = torch.arange(n_inst, 
                        dtype=seq_end_frame.dtype, device=seq_end_frame.device)
                s, _ = self.rnn(s)
                s = s[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))]
            if g is not None:
                g1 = self.embed_goal(g)
                if self.concate_s_g:
                    s = torch.cat((s, g1), -1)
                else:
                    s = s + g1

            return self.mlp(s)


    class Actor(torch.nn.Module):
        def __init__(self, state_dim, act_dim, goal_dim, latent_dim=256, 
                     init_mu=None, init_sigma=None, use_rnn = True, ob_horizon = 2,
                     smaller_nn=False, concate_s_g=False):
            super().__init__()
            if use_rnn:
                self.rnn = torch.nn.GRU(state_dim, latent_dim, batch_first=True)
            else:
                self.rnn = None

            self.concate_s_g = concate_s_g

            if (not smaller_nn) and (not concate_s_g):
                embed_goal_dim = ob_horizon * state_dim if self.rnn is None else latent_dim
                mlp_input_dim = embed_goal_dim
            elif (not smaller_nn) and (concate_s_g):
                raise ValueError("A bad combination of smaller_nn and concate_s_g")
            elif smaller_nn and (not concate_s_g):
                embed_goal_dim = latent_dim if use_rnn else ob_horizon * state_dim
                mlp_input_dim = embed_goal_dim
            elif smaller_nn and concate_s_g:
                embed_goal_dim = latent_dim
                if use_rnn:
                    mlp_input_dim = embed_goal_dim + latent_dim
                else:
                    mlp_input_dim = ob_horizon * state_dim + latent_dim

            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(mlp_input_dim, 1024 if smaller_nn else 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(1024 if smaller_nn else 2048, 512 if smaller_nn else 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(512 if smaller_nn else 2048, 256 if smaller_nn else 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(256 if smaller_nn else 1024, 256 if smaller_nn else 1024),
                torch.nn.ReLU()
            )
            self.embed_goal = torch.nn.Sequential(
                torch.nn.Linear(goal_dim, latent_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(latent_dim, embed_goal_dim)
            )
            self.mu = torch.nn.Linear(256 if smaller_nn else 1024, act_dim)
            self.log_sigma = torch.nn.Linear(256 if smaller_nn else 1024, act_dim)
            with torch.no_grad():
                if init_mu is not None:
                    if torch.is_tensor(init_mu):
                        mu = torch.ones_like(self.mu.bias)*init_mu
                    else:
                        mu = np.ones(self.mu.bias.shape, dtype=np.float32)*init_mu
                        mu = torch.from_numpy(mu)
                    self.mu.bias.data.copy_(mu)
                    torch.nn.init.uniform_(self.mu.weight, -0.00001, 0.00001)
                if init_sigma is None:
                    torch.nn.init.constant_(self.log_sigma.bias, -3)
                    torch.nn.init.uniform_(self.log_sigma.weight, -0.0001, 0.0001)
                else:
                    if torch.is_tensor(init_sigma):
                        log_sigma = (torch.ones_like(self.log_sigma.bias)*init_sigma).log_()
                    else:
                        log_sigma = np.log(np.ones(self.log_sigma.bias.shape, dtype=np.float32)*init_sigma)
                        log_sigma = torch.from_numpy(log_sigma)
                    self.log_sigma.bias.data.copy_(log_sigma)
                    torch.nn.init.uniform_(self.log_sigma.weight, -0.00001, 0.00001)
                self.all_inst = torch.arange(0)

        def forward(self, s, seq_end_frame, g=None):
            s0 = s
            if self.rnn is None:
                s = s.view(s.size(0), -1)
            else:
                n_inst = s.size(0)
                if n_inst > self.all_inst.size(0):
                    self.all_inst = torch.arange(n_inst, 
                        dtype=seq_end_frame.dtype, device=seq_end_frame.device)
                s, _ = self.rnn(s)
                s = s[(self.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))]
            s1 = s
            if g is not None:
                g1 = self.embed_goal(g)
                if self.concate_s_g:
                    s = torch.cat((s, g1), -1)
                else:
                    s = s + g1

            latent = self.mlp(s)
            mu = self.mu(latent)
            log_sigma = self.log_sigma(latent)
            sigma = torch.exp(log_sigma) + 1e-8
            return torch.distributions.Normal(mu, sigma)


    def __init__(self, state_dim: int, act_dim: int, goal_dim: int=0, value_dim: int=1, 
        normalize_value: bool=False,
        init_mu:Optional[Union[torch.Tensor, float]]=None, init_sigma:Optional[Union[torch.Tensor, float]]=None,
        use_rnn = True,
        ob_horizon = 2,
        future_horizon = 3,
        smaller_nn = False,
        concate_s_g = False,
        privilaged_obs_dim = 0
    ):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.actor = self.Actor(state_dim, act_dim, self.goal_dim, init_mu=init_mu, init_sigma=init_sigma, use_rnn=use_rnn, ob_horizon=ob_horizon, smaller_nn=smaller_nn, concate_s_g=concate_s_g)
        self.critic = self.Critic(state_dim, goal_dim, value_dim, use_rnn=use_rnn, ob_horizon = ob_horizon, smaller_nn=smaller_nn, concate_s_g=concate_s_g, privilaged_obs_dim=privilaged_obs_dim)
        self.ob_normalizer = RunningMeanStd(state_dim, clamp=5.0)
        
        self.actor_obs_index = None
        assert self.actor_obs_index is None, "We should handle the actor_obs_index in the Env.obs()"

        if normalize_value:            
            self.value_normalizer = DiagonalPopArt(value_dim, 
                self.critic.mlp[-1].weight, self.critic.mlp[-1].bias)
        else:
            self.value_normalizer = None
            
    def observe(self, obs, privileged_obs=None, norm=True):
        if self.goal_dim > 0:
            ## For NO FEET CASES
            # obs is [1, 489] for no feet cases -- in Isaac Gym
            # self.goal_dim = 27, self.state_dim = 231
            s = obs[:, :-self.goal_dim]
            g = obs[:, -self.goal_dim:]
        else:
            s = obs
            g = None
        s = s.view(*s.shape[:-1], -1, self.state_dim)
        return self.ob_normalizer(s) if norm else s, g, privileged_obs

    def process_actor_goal(self, g):
        goal = g[:,:-1].view(g.shape[0],-1,3) # ignore binary flag
        g_ = goal[:,self.actor_obs_index,:].view(g.shape[0],-1)
        return torch.hstack([g_, g[:,[-1]]])

    def eval_(self, s, s_privileged, seq_end_frame, g, unnorm):
        # s.shape = torch.Size([1024, 2, 231]); g.shape = torch.Size([1024, 28])
        # Image of the shape of s_privileged torch.Size([1024, 2, 40])
        # The concated_s shape is torch.Size([1024, 2, 271])
        concated_g = torch.cat((g, s_privileged), -1) if s_privileged is not None else g
        v = self.critic(s, seq_end_frame, concated_g)
        if unnorm and self.value_normalizer is not None:
            v = self.value_normalizer(v, unnorm=True)
        return v

    def act(self, obs, seq_end_frame, stochastic=None, unnorm=False, privileged_obs=None):
        if stochastic is None:
            stochastic = self.training
        s, g, s_privileged = self.observe(obs, privileged_obs)    # s.shape = torch.Size([1024, 2, 245]); g.shape = torch.Size([1024, 46])
        if self.actor_obs_index is not None:
            ag = self.process_actor_goal(g) # torch.Size([1024, 28])
        else:
            ag = g
        # seq_end_frame.shape = torch.Size([1024])
        pi = self.actor(s, seq_end_frame, ag)
        if stochastic:
            a = pi.sample()
            lp = pi.log_prob(a)
            if g is not None:
                g = g[...,:self.goal_dim]
            return a, self.eval_(s, s_privileged, seq_end_frame, g, unnorm), lp
        else:
            return pi.mean

    def evaluate(self, obs, seq_end_frame, unnorm=False, privileged_obs=None):
        s, g, s_privileged = self.observe(obs, privileged_obs)
        if g is not None:
            g = g[...,:self.goal_dim]
        return self.eval_(s, s_privileged, seq_end_frame, g, unnorm)
    
    def forward(self, obs, seq_end_frame, unnorm=False, privileged_obs=None):
        s, g, s_privileged = self.observe(obs, privileged_obs)
        if self.actor_obs_index is not None:
            ag = self.process_actor_goal(g)
        else:
            ag = g
        pi = self.actor(s, seq_end_frame, ag)
        if g is not None:
            g = g[...,:self.goal_dim]
        return pi, self.eval_(s, s_privileged, seq_end_frame, g, unnorm)

    def load_actor_dict(self, state_dict):
        actor_state_dict = {
            key.replace("actor.", ""): value 
            for key, value in state_dict.items() 
            if key.startswith("actor.")
        }
        ob_normalizer_state_dict = {
            key.replace("ob_normalizer.", ""): value 
            for key, value in state_dict.items() 
            if key.startswith("ob_normalizer.")
        }
        self.actor.load_state_dict(actor_state_dict)
        self.ob_normalizer.load_state_dict(ob_normalizer_state_dict)
