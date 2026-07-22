import numpy as np
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from typing import Optional
from pathlib import Path

class SARSOPPlanner(Policy):
    def __init__(
        self,
        environment:Environment,
        discount_factor: float,
        epsilon_convergence: float,
        maximun_iteration: int,
        name:str,
        delta_threshold:float,
        time_out_in_seconds: Optional[int] = None,
        debug:bool =False,
        log_path:Optional[Path]=None,
        use_queue_logger:bool=False
    ):
        #MAKE all the check about the possible errors
        super.__init__(
            environment = environment,
            discount_factor=discount_factor,
            name=name,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger
        )

        self.epsilon_convergence=epsilon_convergence
        self.maximun_iteration=maximun_iteration
        self.delta_threshold=delta_threshold
        self.time_out_in_seconds=time_out_in_seconds

    def initialize_gamma(self):
        """Initializes the lower-bound alpha-vector set with a blind policy vector."""
        #S_size = self.environment.
        #min_reward = min(self.pomdp['min_reward'])
        #alpha_init = np.full(S_size, min_reward / (1 - self.gamma))
        #return [alpha_init]
