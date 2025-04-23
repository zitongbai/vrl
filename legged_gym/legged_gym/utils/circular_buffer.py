import numpy as np
import torch
from typing import Tuple, Optional

class CircularBuffer:
    def __init__(self, 
                 time_steps: int,
                 num_envs: int,
                 shape: Tuple[int, ...],
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device("cpu")):
        """initializes a circular buffer for storing tensor data
        This buffer is designed to store data in a circular manner, allowing for efficient
        overwriting of old data when new data is added. It is particularly useful for
        reinforcement learning applications where data is collected in batches over time.
        
        Args:
            time_steps: time steps, the maximum number of time steps to store
            num_envs: number of environments, the number of environments to store data for
            shape: shape of the data to be stored, e.g. (3, 64, 64) for an image
            dtype: data type of the tensor, default is torch.float32
            device: device to store the tensor, default is CPU
        """
        self.time_steps = time_steps
        self.num_envs = num_envs
        self.shape = shape
        self.dtype = dtype
        self.device = device
        
        self.buffer = torch.zeros(
            (time_steps, num_envs, *shape),
            dtype=dtype,
            device=device
        )
        
        self.current_idx = 0
    
    def push(self, data: torch.Tensor, reset_env=None) -> None:
        """push data to buffer
        
        Args:
            data: (num_envs, *shape)
        """
        assert data.shape == (self.num_envs, *self.shape), \
            f"Data shape {data.shape} does not match buffer shape {(self.num_envs, *self.shape)}"
            
        self.buffer[self.current_idx] = data
        if reset_env is not None:
            self.buffer[:, reset_env, ...] = data[reset_env, ...]
        # Update the current index
        self.current_idx = (self.current_idx + 1) % self.time_steps
        
    def get_all(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get all steps of data from the buffer
        Args:
            env_ids: environment IDs, if None, return all envs
        """
        if env_ids is None:
            return torch.cat([
                self.buffer[self.current_idx:, ...],
                self.buffer[:self.current_idx, ...]
            ], dim=0)
        else:
            return torch.cat([
                self.buffer[self.current_idx:, env_ids, ...],
                self.buffer[:self.current_idx, env_ids, ...]
            ], dim=0)
            
    def clear(self) -> None:
        """clear the buffer"""
        self.buffer.zero_()
        self.current_idx = 0
        
    def __len__(self) -> int:
        """get the number of elements in the buffer"""
        return self.time_steps
    
if __name__ == "__main__":
    # Test the CircularBuffer class
    
    time_steps = 24
    num_envs = 4096
    shape = (3,)
    device = torch.device("cuda:0")
    
    buffer = CircularBuffer(time_steps, num_envs, shape, device=device)
    
    init_data = torch.randn((num_envs, *shape), device=device)
    buffer.push(init_data, reset_env=torch.arange(num_envs, device=device))

    print(buffer.buffer[0:10, 0:2, ...])
    
    data = torch.zeros((num_envs, *shape), device=device, dtype=torch.float32)
    buffer.push(data, reset_env=torch.tensor([3, ], dtype=torch.long, device=device))
    buffer.push(data, reset_env=torch.tensor([3, ], dtype=torch.long, device=device))
    print(buffer.buffer[0:10, 0:2, ...])
    
    all_buffer = buffer.get_all()
    print(all_buffer[:, 0:2, ...])