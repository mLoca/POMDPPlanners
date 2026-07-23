import numpy as np
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.environment.environment import SpaceType
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from typing import List, Optional
from pathlib import Path

class SARSOPPlanner(Policy):
    def __init__(
        self,
        environment:"Environment",
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

        super().__init__(
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
        self.alpha_vectors: List[np.ndarray] = []

        self.compatibility_check()
        self.initialize_alpha()

    def compatibility_check(self):
        if self.environment.reward_range is None:
            raise ValueError(
                f"Policy {self.name} is not compatible with the environment {self.environment.name} because the policy requires the environment to have a defined reward range. Please use an environment with a defined reward range."
            )
        if self.environment.__getattribute__("states") is None:
            raise ValueError(
                f"Policy {self.name} is not compatible with the environment {self.environment.name} because the policy requires the environment to have a defined state space."
            )

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(action_space=SpaceType.DISCRETE, observation_space=SpaceType.MIXED)

    def initialize_alpha(self):
        """Initializes the lower-bound alpha-vector set with a blind policy vector."""

        state_size = len(self.environment.__getattribute__("states"))
        min_reward = min(self.environment.reward_range) # type: ignore

        #TODO: evaluate if the alphavector should be a class on its own, or if it should be a numpy array.
        alpha_init = np.full(state_size, min_reward / (1 - self.discount_factor))
        self.alpha_vectors = [alpha_init]

    def plan(self):
        """Plans a policy using the SARSOP algorithm."""
        return 

