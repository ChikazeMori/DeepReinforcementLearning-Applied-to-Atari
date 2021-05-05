# import libraries
import gym
import numpy as np
from gym import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import random
import math
from collections import namedtuple

class utilities():
    
    def __init__(self):
        super(utilities, self).__init__()
        self.steps_done = 0
        

    def get_location(screen_width):
        world_width = env.x_threshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.0)

    def get_screen(env):
        resize = T.Compose([T.ToPILImage(),
                    T.Resize(160, interpolation=Image.CUBIC),
                    T.ToTensor()])
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))

        # Convert to float, rescale, convert to torch tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Resize, and add a batch dimension
        return resize(screen).unsqueeze(0)
    
    def select_action(state,steps_done,policy_net):
        # hyperparameters
        BATCH_SIZE = 128
        GAMMA = 0.999
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200

        
        #number of actions
        n_actions = 9

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)