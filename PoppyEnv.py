import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tqdm import tqdm
from pypot.creatures import PoppyTorso
from pypot.primitive.move import Move, MovePlayer
import torch
from utils.blazepose import blazepose_skeletons


class PoppyEnv(gym.Env):
    """Custom Gym Environment for the Poppy Torso robot."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PoppyEnv, self).__init__()
        print("Hello, I am Poppy!")

        # Initialize connection to Poppy
        from pypot import vrep
        vrep.close_all_connections()
        self.poppy = PoppyTorso(simulator='vrep')

        # Initialize state variables
        self.current_step = 0
        self.num_steps = 0
        self.target_loaded = False
        self.done = False
        self.episodes = 0
        self.restart_every_n_episodes = 1000

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array(
            [0, -180]), high=np.array([180, 0]), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32)

    # def seed(self, seed=None):
    #     np.random.seed(seed)

    def step(self, action):
        """Executes one step within the environment."""
        action_l, action_r = action  # Decode action
        if self.current_step > 125:
            self.move_left_arm(action_l)
        else:
            self.move_right_arm(action_r)
        obs = self.get_obs()  # Retrieve observations
        # Calculate reward based on target distance
        dis = self.calculate_distance(obs)
        reward = np.exp(-10 * dis) if dis <= 0.3 else 0
        self.current_step += 5
        self.done = (self.current_step >= self.num_steps)
        if self.done:
            self.episodes += 1
        info = {'episode': self.episodes,
                'step': self.current_step, 'reward': reward}
        return np.float32(obs), reward, self.done, info

    def reset(self, seed=None, **kwargs):
        """Resets the environment to an initial state."""
        if seed is not None:
            np.random.seed(seed)
        self.set_initial_positions()
        self.current_step = 0
        self.done = False
        if not self.target_loaded:
            self.get_target()
            self.target_loaded = True
        self.num_steps = self.targets.shape[0]
        obs = self.get_obs()
        return np.float32(obs)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_obs(self):
        """Retrieves the current state of the robot."""
        return np.concatenate([self.poppy.l_arm_chain.position, self.poppy.r_arm_chain.position])

    # def calculate_distance(self, obs):
    #     """Calculates the distance between the current and target positions."""
    #     if self.current_step <= 125:
    #         return np.linalg.norm(obs[3:] - np.array(self.targets[self.current_step].flatten())[3:])
    #     else:
    #         return np.linalg.norm(obs[:3] - np.array(self.targets[self.current_step].flatten())[:3])

    def calculate_distance(self, obs):
        if self.current_step >= len(self.targets):
            return 0  # Avoid indexing beyond the list

        target = self.targets[self.current_step]
        print(self.targets.shape)
        print(type(target))

        # Debug print to check structures
        print(f"obs: {obs}, type of obs: {type(obs)}")
        print(f"target: {target}, type of target: {type(target)}")

        if self.current_step <= 125:
            # Ensure both obs and target can be indexed/sliced
            if isinstance(obs, np.ndarray) and isinstance(target, np.ndarray):
                if obs.shape[0] > 3 and target.shape[0] > 3:
                    return np.linalg.norm(obs[3:] - target[3:])
                else:
                    print(
                        "obs or target does not have enough elements to index [3:]")
            else:
                print("obs or target is not an array-like structure as expected")
        else:
            # Similar checks for other conditions
            if isinstance(obs, np.ndarray) and isinstance(target, np.ndarray):
                if obs.shape[0] > 0 and target.shape[0] > 0:
                    return np.linalg.norm(obs[:3] - target[:3])
                else:
                    print(
                        "obs or target does not have enough elements to index [:3]")
            else:
                print("obs or target is not an array-like structure as expected")

        return 0  # Default return if conditions fail

    def set_initial_positions(self):
        """Sets the initial positions of all joints."""
        joint_pos = {
            'l_elbow_y': 90.0, 'head_y': 0.0, 'r_arm_z': 0.0, 'head_z': 0.0,
            'r_shoulder_x': 0.0, 'r_shoulder_y': 0.0, 'r_elbow_y': 90.0,
            'l_arm_z': 0.0, 'abs_z': 0.0, 'bust_y': 0.0, 'bust_x': 0.0,
            'l_shoulder_x': 0.0, 'l_shoulder_y': 0.0
        }
        for motor in self.poppy.motors:
            motor.goto_position(joint_pos[motor.name], 1, wait=True)

    def move_left_arm(self, action_l):
        """Moves the left arm based on the action."""
        for motor in self.poppy.l_arm_chain.motors:
            if motor.name == 'l_shoulder_x':
                motor.goto_position(action_l, 1, wait=True)
            elif motor.name == 'l_elbow_y':
                motor.goto_position(90.0, 1, wait=True)
            else:
                motor.goto_position(0.0, 1, wait=True)

    def move_right_arm(self, action_r):
        """Moves the right arm based on the action."""
        for motor in self.poppy.r_arm_chain.motors:
            if motor.name == 'r_shoulder_x':
                motor.goto_position(action_r, 1, wait=True)
            elif motor.name == 'r_elbow_y':
                motor.goto_position(90.0, 1, wait=True)
            else:
                motor.goto_position(0.0, 1, wait=True)

    # def targets_from_skeleton(self, skeletons, topology, lengths):
    #     targets = []
    #     all_positions = []
    #     for skeleton in skeletons:
    #         joint_targets = []
    #         joint_positions = []
    #         for i, length in enumerate(lengths):
    #             if i < len(skeleton) - 1:
    #                 position = skeleton[i] + length * 0.1
    #                 joint_positions.append(position)
    #                 joint_targets.append(
    #                     position + np.random.normal(0, 0.01, size=position.shape))
    #         targets.append(joint_targets)
    #         all_positions.append(joint_positions)
    #     return np.array(targets), np.array(all_positions)

    def targets_from_skeleton(self, skeletons, topology, lengths):
        targets = []
        all_positions = []

        for skeleton in skeletons:
            joint_targets = []
            joint_positions = []

            for i, length in enumerate(lengths):
                if i < len(skeleton):  # Ensure valid indexing
                    # Calculate a target as a simple transformation of the skeleton point
                    if topology and i < len(topology):
                        parent_index = topology[i]
                        # Just a dummy transformation
                        position = skeleton[parent_index] + length * 0.1
                    else:
                        position = skeleton[i] + length * 0.1

                    joint_positions.append(position.tolist())
                    # This needs to be a vector, not a scalar
                    joint_targets.append(position.tolist())

            targets.append(joint_targets)
            all_positions.append(joint_positions)

        # Convert to numpy arrays for consistency
        targets = np.array([np.array(t) for t in targets])
        all_positions = np.array([np.array(p) for p in all_positions])

        return targets, all_positions

    def get_target(self):
        """Extracts motion targets from video and prepares them for use."""
        self.skeletons = blazepose_skeletons('mai1.mov')
        self.topology = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.poppy_lengths = torch.tensor(
            [0.0, 0.07, 0.18, 0.19, 0.07, 0.18, 0.19, 0.12, 0.08, 0.07, 0.05, 0.1, 0.15, 0.13, 0.1, 0.15, 0.13])
        targets, all_positions = self.targets_from_skeleton(
            self.skeletons, self.topology, self.poppy_lengths)
        interpolated_targets = self.interpolate_targets(targets)
        smoothed_targets = self.moving_average(interpolated_targets, n=15)
        self.targets = smoothed_targets

    def moving_average(self, a, n=3):
        """Applies a moving average filter to the data."""
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def interpolate_targets(self, targets, factor=10):
        """Interpolates between given target positions."""
        original_length = targets.shape[0]
        new_length = original_length + (original_length - 1) * factor
        new_targets = np.zeros((new_length,) + targets.shape[1:])
        for i in range(original_length - 1):
            for j in range(factor + 1):
                t = j / float(factor)
                new_targets[i * (factor + 1) + j] = (1 - t) * \
                    targets[i] + t * targets[i + 1]
        new_targets[-1] = targets[-1]
        return new_targets
