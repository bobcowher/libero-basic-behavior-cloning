import contextlib
import io
import os
import sys
import numpy as np
import h5py
import inspect
from dataloader import DataLoader 
from agent import Agent

agent = Agent()

agent.train(epochs=100000, batch_size=32)

agent.close()

