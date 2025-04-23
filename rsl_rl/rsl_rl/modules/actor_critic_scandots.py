import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation
from .actor_critic_recurrent import Memory


class Actor(nn.Module):
    def __init__(self, 
                    num_proprioception,
                    height_measurements_size,
                    
                    cnn_channels=[16, 32, 32],
                    cnn_kernel_sizes=[3, 2, 2],
                    cnn_strides=[2, 2, 1],
                    cnn_padding=[0, 0, 0],
                    cnn_embedding_dim=32,
                    
                    rnn_type='gru',
                    rnn_hidden_size=256,
                    rnn_num_layers=1,
                    
                    mlp_hidden_dims=[512, 256, 128],
                    mlp_output_dim = 12,
                    mlp_activation='elu'):
        super().__init__()
        
        self.num_proprioception = num_proprioception
        self.height_measurements_size = height_measurements_size
        
        assert len(cnn_channels) == len(cnn_kernel_sizes) == len(cnn_strides) == len(cnn_padding), \
            f"cnn_channels {cnn_channels}, cnn_kernel_sizes {cnn_kernel_sizes}, cnn_strides {cnn_strides}, cnn_padding {cnn_padding} should have the same length"
        self.height_encoder = []
        for i in range(len(cnn_channels)):
            if i == 0:
                input_channels = 1
            else:
                input_channels = cnn_channels[i - 1]
            self.height_encoder.append(nn.Conv2d(input_channels, cnn_channels[i], kernel_size=cnn_kernel_sizes[i], stride=cnn_strides[i], padding=cnn_padding[i]))
            self.height_encoder.append(nn.ELU())
            self.height_encoder.append(nn.MaxPool2d(kernel_size=2, stride=1))
        self.height_encoder.append(nn.Flatten())
        self.height_encoder.append(nn.Linear(2304, cnn_embedding_dim))
        self.height_encoder.append(nn.ELU())
        self.height_encoder = nn.Sequential(*self.height_encoder)
        
        print(f"Height Encoder: {self.height_encoder}")
        
        self.memory = Memory(
            input_size = num_proprioception + cnn_embedding_dim,
            type = rnn_type,
            num_layers = rnn_num_layers,
            hidden_size = rnn_hidden_size
        )
        
        mlp_activation = get_activation(mlp_activation)
        mlp_input_dim = rnn_hidden_size
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim, mlp_hidden_dims[0]))
        actor_layers.append(mlp_activation)
        for l in range(len(mlp_hidden_dims)):
            if l == len(mlp_hidden_dims) - 1:
                actor_layers.append(nn.Linear(mlp_hidden_dims[l], mlp_output_dim))
            else:
                actor_layers.append(nn.Linear(mlp_hidden_dims[l], mlp_hidden_dims[l + 1]))
                actor_layers.append(mlp_activation)
        self.actor = nn.Sequential(*actor_layers)
        
        print(f"Actor MLP: {self.actor}")


    def reset(self, dones=None):
        self.memory.reset(dones)
        
    def forward(self, observations, 
                masks=None, 
                hidden_states=None):
        proprioception = observations[..., :self.num_proprioception]
        # print(f"Proprioception shape: {proprioception.shape}")
        
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
        
        # Concatenate proprioception and height embedding
        memory_input = torch.cat((proprioception, height_embedding), dim=-1)
        
        input_a = self.memory(
            memory_input, 
            masks=masks, 
            hidden_states=hidden_states
        ).squeeze(0)

        return self.actor(input_a)
    
    def get_hidden_states(self):
        if self.memory.hidden_states is None:
            return None
        return self.memory.hidden_states


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
                    height_measurements_size = (33, 21), 
                    
                    cnn_channels=[16, 32, 32],
                    cnn_kernel_sizes=[3, 2, 2],
                    cnn_strides=[2, 2, 1],
                    cnn_padding=[0, 0, 0],
                    cnn_embedding_dim=32,
                    
                    rnn_type='lstm',
                    rnn_hidden_size=256,
                    rnn_num_layers=1,
                    
                    init_noise_std=1.0,
                    **kwargs
                ):
        if kwargs:
            print("ActorCriticScandots.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)
        super(ActorCriticScandots, self).__init__()
        
        assert num_actor_obs == num_proprioception + height_measurements_size[0] * height_measurements_size[1], \
            f"num_actor_obs {num_actor_obs} should be equal to num_proprioception {num_proprioception} + height_measurements_size {height_measurements_size[0]} * {height_measurements_size[1]}"
        
         
        # policy network
        print("------------------------------------------------------")
        print("Policy network info:")
        print("------------------------------------------------------")
        self.actor = Actor(
            num_proprioception=num_proprioception,
            height_measurements_size=height_measurements_size,
            cnn_channels=cnn_channels,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides,
            cnn_padding=cnn_padding,
            cnn_embedding_dim=cnn_embedding_dim,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            mlp_hidden_dims=actor_hidden_dims,
            mlp_output_dim=num_actions,
            mlp_activation=activation
        )
        
        # value network
        print("------------------------------------------------------")
        print("Value network info:")
        print("------------------------------------------------------")
        self.critic = Actor(
            num_proprioception=num_proprioception,
            height_measurements_size=height_measurements_size,
            cnn_channels=cnn_channels,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides,
            cnn_padding=cnn_padding,
            cnn_embedding_dim=cnn_embedding_dim,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            mlp_hidden_dims=actor_hidden_dims,
            mlp_output_dim=1,
            mlp_activation=activation
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
        # forward pass through actor
        mean = self.actor(
            observations, 
            masks=masks, 
            hidden_states=hidden_states
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
        value = self.critic(
            critic_observations,
            masks=masks, 
            hidden_states=hidden_states
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
        num_actor_obs = 45 + 33*21,
        num_critic_obs = 45 + 33*21,
        num_actions = 12,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation='elu',
        num_proprioception = 45,
        height_measurements_size = (33, 21), 
        
        cnn_channels=[16, 32, 32],
        cnn_kernel_sizes=[2, 2, 1],
        cnn_strides=[2, 1, 1],
        cnn_padding=[0, 0, 0],
        cnn_embedding_dim=32,
        
        rnn_type='gru',
        rnn_hidden_size=256,
        rnn_num_layers=1,
        
        init_noise_std=1.0,
    )
    
    print("--------------------------------------------------")
    print("Test inference mode (collection)")
    obs = torch.randn(4096, 45 + 33*21)   # [num_envs, obs_size]
    a = acs.act(obs)
    print(f"Action shape: {a.shape}")   #   [4096, 12]
    
    
    print("--------------------------------------------------")
    print("Test inference mode (batch)")
    obs_batch = torch.randn(24, 233, 45 + 33*21)  # [time_steps, num_traj, obs_size]
    masks = torch.ones(24, 233, dtype=torch.bool) # [time_steps, num_traj]
    
    hidden_states = torch.randn(1, 233, 256) # [num_layers, batch, hidden_dim]
    a_batch = acs.act(obs_batch, masks=masks, hidden_states=hidden_states)
    print(f"Action shape: {a_batch.shape}")   #   [24, 233, 12]