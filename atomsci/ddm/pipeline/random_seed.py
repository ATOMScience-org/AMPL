""" Used to set random seed from parameter_parser for reproducibility. """
import numpy as np 
import uuid 
import random
import torch
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)-15s %(message)s') 
#----------------------------------------------------------------------------------
class RandomStateGenerator:
    """
    A class to manage random state and seed generation for reproducible randomness.

    Attributes:
        params: Additional parameters.
        seed: The seed for the random state.
        random_state: The random state generator.
    """
    def __init__(self, params=None, seed=None):
        self.params = params
        if seed is not None:
            self.seed = seed
        elif self.params.seed is not None: 
            self.seed = self.params.seed
        else:
            self.seed = uuid.uuid4().int % (2**32)
        self.set_seed(self.seed)
    
    def set_seed(self, seed):
        log = logging.getLogger('ATOM')
        log.warning("The global seed is being set to %d, for reproducibility. Note that this action will synchronize the randonmess across all libraries which may impact the randomness of other parts of the pipeline.", seed)
        """Set the seed for all relevant libraries."""
        
        global _seed, _random_state
        _seed = seed

        _random_state = np.random.default_rng(_seed)
        
        # set seed for numpy
        np.random.default_rng(_seed)
        
        # needed for deepchem
        np.random.seed(_seed)
        
        # set seed for random
        random.seed(_seed)
        
        # set seed for PyTorch
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # set seed for tensorflow 
        tf.random.set_seed(_seed)
        
        self.random_state = _random_state

    def get_seed(self):
        """Returns the seed when called"""
        return self.seed
    
    def get_random_state(self):
        """Returns the random state when called"""
        return self.random_state