import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation
from .actor_critic_recurrent import Memory


class Actor(nn.Module):
    def __init__(self, 
                    num_proprioception,
                    height_measurements_size,
                    
                    propr_rnn_type = 'gru',
                    propr_rnn_hidden_size = 256,
                    propr_rnn_num_layers = 1,
                    
                    height_encoder_output_dim=32,
                    height_rnn_type='gru',
                    height_rnn_hidden_size=256,
                    height_rnn_num_layers=1,
                    
                    mlp_hidden_dims=[256, 256, 256],
                    mlp_output_dim = 12,
                    activation='elu'):
        super().__init__()
        
        self.num_proprioception = num_proprioception
        self.height_measurements_size = height_measurements_size
        
        self.proprioception_memory = Memory(
            input_size=num_proprioception, 
            type=propr_rnn_type,
            num_layers=propr_rnn_num_layers,
            hidden_size=propr_rnn_hidden_size
        )
        print(f"Proprioception RNN: {self.proprioception_memory}")

        # self.height_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=2),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.ELU(),
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.ELU(),
        #     nn.Flatten(),
        #     nn.Linear(640, height_encoder_output_dim),
        #     nn.ELU(),
        # )
        
        # self.height_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=2),
        #     nn.Conv2d(in_channels=8, out_channels=32, kernel_size=2, stride=2), 
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
        #     nn.ELU(),
        #     nn.Flatten(),
        #     nn.Linear(32 * 6 * 6, height_encoder_output_dim),
        #     nn.ELU(),
        # )
        
        self.height_encoder = nn.Sequential(
            # 输入：1×33×21
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Conv3×3, 16 ch
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),                          # 尺寸减半: ~16×16

            # 空洞卷积扩大感受野
            nn.Conv2d(16, height_encoder_output_dim, kernel_size=3, dilation=2, padding=2),  # Conv3×3, dilation=2, height_encoder_output_dim ch
            nn.BatchNorm2d(height_encoder_output_dim),
            nn.ReLU(inplace=True),

            # 直接全局平均池化到 1×1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # → [B, height_encoder_output_dim]
        )
        
        print(f"Height Encoder: {self.height_encoder}")
        
        self.height_memory = Memory(
            input_size = height_encoder_output_dim, 
            type = height_rnn_type, 
            num_layers = height_rnn_num_layers,
            hidden_size = height_rnn_hidden_size
        )
        print(f"Height RNN: {self.height_memory}")
        
        activation = get_activation(activation)
        mlp_input_dim = propr_rnn_hidden_size + height_rnn_hidden_size
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim, mlp_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(mlp_hidden_dims)):
            if l == len(mlp_hidden_dims) - 1:
                actor_layers.append(nn.Linear(mlp_hidden_dims[l], mlp_output_dim))
            else:
                actor_layers.append(nn.Linear(mlp_hidden_dims[l], mlp_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
        print(f"Actor MLP: {self.actor}")


    def reset(self, dones=None):
        self.proprioception_memory.reset(dones)
        self.height_memory.reset(dones)
        
    def forward(self, observations, 
                masks=None, 
                propri_hidden_states=None, 
                height_hidden_states=None):
        proprioception = observations[..., :self.num_proprioception]
        # print(f"Proprioception shape: {proprioception.shape}")
        
        propr_latent = self.proprioception_memory(
            proprioception, masks=masks, hidden_states=propri_hidden_states
        )
        # print(f"Proprioception latent shape: {propr_latent.shape}")
        
        height_measurements = observations[..., self.num_proprioception:].reshape(
            -1, 1, self.height_measurements_size[0], self.height_measurements_size[1]
        ) # merge steps and num_envs dimension
        # print(f"Height measurements shape: {height_measurements.shape}")
        
        height_embedding = self.height_encoder(height_measurements)
        # print(f"Height embedding shape: {height_embedding.shape}")
        
        height_embedding = height_embedding.view(
            *proprioception.shape[:-1], -1
        )
        # print(f"Height embedding reshaped: {height_embedding.shape}")
        
        height_latent = self.height_memory(
            height_embedding, masks=masks, hidden_states=height_hidden_states
        )
        # print(f"Height latent shape: {height_latent.shape}")
        
        input_a = torch.cat((propr_latent, height_latent), dim=-1).squeeze(0)
        # print(f"Input to actor shape: {input_a.shape}")

        return self.actor(input_a)
    
    def get_hidden_states(self):
        if self.proprioception_memory.hidden_states is None or self.height_memory.hidden_states is None:
            return None
        
        # hidden_states = torch.cat(
        #     [self.proprioception_memory.hidden_states, self.height_memory.hidden_states], dim=-1
        # )
        # return hidden_states
        
        return (self.proprioception_memory.hidden_states, self.height_memory.hidden_states)


class ActorCriticScandots(nn.Module):
    is_recurrent = True
    def __init__(self, 
                    num_actor_obs,
                    num_critic_obs,
                    num_actions,
                    actor_hidden_dims=[256, 256, 256],
                    critic_hidden_dims=[256, 256, 256],
                    activation='elu',
                    num_proprioception = 45,
                    height_measurements_size = (12, 11), 
                    
                    propr_rnn_type = 'gru',
                    propr_rnn_hidden_size = 256,
                    propr_rnn_num_layers = 1,
                    
                    height_encoder_output_dim=32,
                    height_rnn_type='gru',
                    height_rnn_hidden_size=256,
                    height_rnn_num_layers=1,
                    
                    init_noise_std=1.0,
                    **kwargs
                ):
        if kwargs:
            print("ActorCriticScandots.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)
        super(ActorCriticScandots, self).__init__()
        
        assert num_actor_obs == num_proprioception + height_measurements_size[0] * height_measurements_size[1], \
            f"num_actor_obs {num_actor_obs} should be equal to num_proprioception {num_proprioception} + height_measurements_size {height_measurements_size[0]} * {height_measurements_size[1]}"
        
        self.propr_rnn_hidden_size = propr_rnn_hidden_size
        self.height_rnn_hidden_size = height_rnn_hidden_size
        
        # policy network
        print("------------------------------------------------------")
        print("Policy network info:")
        print("------------------------------------------------------")
        self.actor = Actor(
            num_proprioception=num_proprioception,
            height_measurements_size=height_measurements_size,
            propr_rnn_type=propr_rnn_type,
            propr_rnn_hidden_size=propr_rnn_hidden_size,
            propr_rnn_num_layers=propr_rnn_num_layers,
            height_encoder_output_dim=height_encoder_output_dim,
            height_rnn_type=height_rnn_type,
            height_rnn_hidden_size=height_rnn_hidden_size,
            height_rnn_num_layers=height_rnn_num_layers,
            mlp_hidden_dims=actor_hidden_dims,
            mlp_output_dim=num_actions,
            activation=activation
        )
        
        # value network
        print("------------------------------------------------------")
        print("Value network info:")
        print("------------------------------------------------------")
        self.critic = Actor(
            num_proprioception=num_proprioception,
            height_measurements_size=height_measurements_size,
            propr_rnn_type=propr_rnn_type,
            propr_rnn_hidden_size=propr_rnn_hidden_size,
            propr_rnn_num_layers=propr_rnn_num_layers,
            height_encoder_output_dim=height_encoder_output_dim,
            height_rnn_type=height_rnn_type,
            height_rnn_hidden_size=height_rnn_hidden_size,
            height_rnn_num_layers=height_rnn_num_layers,
            mlp_hidden_dims=critic_hidden_dims,
            mlp_output_dim=1,
            activation=activation
        )
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    def reset(self, dones=None):
        self.actor.reset(dones)
        self.critic.reset(dones)

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations, masks=None, hidden_states=None):
        # hidden_states: [num_layers, batch, hidden_dim]
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("hidden_states should not be None in batch mode")
            # propr_hidden_states = hidden_states[:, :, :self.propr_rnn_hidden_size]
            # height_hidden_states = hidden_states[:, :, self.propr_rnn_hidden_size:]
            propr_hidden_states = hidden_states[0]
            height_hidden_states = hidden_states[1]
        else:
            # no hidden states
            propr_hidden_states = None
            height_hidden_states = None
        # forward pass through actor
        mean = self.actor(
            observations, 
            masks=masks, 
            propri_hidden_states=propr_hidden_states, 
            height_hidden_states=height_hidden_states
        )
        # update the distribution
        self.distribution = Normal(mean, mean*0. + self.std)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("hidden_states should not be None in batch mode")
            # propr_hidden_states = hidden_states[:, :self.propr_rnn_hidden_size]
            # height_hidden_states = hidden_states[:, self.propr_rnn_hidden_size:]
            propr_hidden_states = hidden_states[0]
            height_hidden_states = hidden_states[1]
        else:
            # no hidden states
            propr_hidden_states = None
            height_hidden_states = None
        value = self.critic(
            critic_observations,
            masks=masks, 
            propri_hidden_states=propr_hidden_states, 
            height_hidden_states=height_hidden_states
        )
        return value

    def get_hidden_states(self):
        # hidden states in actor
        actor_hidden_states = self.actor.get_hidden_states()
        # hidden states in critic
        critic_hidden_states = self.critic.get_hidden_states()
        return actor_hidden_states, critic_hidden_states

if __name__ == "__main__":
    # test the class
    acs = ActorCriticScandots(
        num_actor_obs = 45 + 12 * 11,
        num_critic_obs = 45 + 12 * 11,
        num_actions = 12,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation='elu',
        num_proprioception = 45,
        height_measurements_size = (12, 11), 
        
        propr_rnn_type = 'gru',
        propr_rnn_hidden_size = 256,
        propr_rnn_num_layers = 1,
        
        height_encoder_output_dim=32,
        height_rnn_type='gru',
        height_rnn_hidden_size=256,
        height_rnn_num_layers=1,
        
        init_noise_std=1.0,
    )
    
    print("--------------------------------------------------")
    print("Test inference mode (collection)")
    obs = torch.randn(4096, 45 + 12 * 11)   # [num_envs, obs_size]
    a = acs.act(obs)
    print(f"Action shape: {a.shape}")   #   [4096, 12]
    
    
    print("--------------------------------------------------")
    print("Test inference mode (batch)")
    obs_batch = torch.randn(24, 233, 45 + 12 * 11)  # [time_steps, num_traj, obs_size]
    masks = torch.ones(24, 233, dtype=torch.bool) # [time_steps, num_traj]
    
    # hidden_states = torch.randn(1, 233, 512) # [num_layers, batch, hidden_dim]
    hidden_states = (torch.randn(1, 233, 256), torch.randn(1, 233, 256))
    a_batch = acs.act(obs_batch, masks=masks, hidden_states=hidden_states)
    print(f"Action shape: {a_batch.shape}")   #   [24, 233, 12]