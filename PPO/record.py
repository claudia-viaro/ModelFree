import torch
import numpy as np
import torch.nn as nn


class RecValues:
    def __init__(self, episodes):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.s_LogReg = []
        self.r_LogReg = []
        self.Xa_pre = []
        self.Xa_post = np.zeros(size = (episodes, 2000))
 
  
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.s_LogReg[:]
        del self.r_LogReg[:]
        del self.Xa_pre[:]
        del self.Xa_post[:]
    
    
    @property
    def nb_entries(self):
        return len(self.states)
