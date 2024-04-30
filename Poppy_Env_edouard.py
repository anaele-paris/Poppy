from pypot.primitive.move import MovePlayer
from pypot.primitive.move import Move
import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tqdm import tqdm


import numpy as np
from utils.skeleton import *
from utils.quaternion import *
from utils.blazepose import blazepose_skeletons
from utils.video_capturing import preprocess_skeletons, moving_average, interpolate_skeletons
import os
from pypot.creatures import PoppyTorso
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# from pypot.creatures.ik import IKChain


class PoppyEnv(gym.Env):

    def __init__(self, goals=2, terminates=True):

        print("Hello, I am Poppy!")

        # connection to Poppy (Starts an instance on Coppelia sim, Poppy should appear in the simulator)
        from pypot import vrep
        vrep.close_all_connections()
        self.poppy = PoppyTorso(simulator='vrep')

        # define Poppy's topology and lengths
        self.topology = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.poppy_lengths = torch.Tensor([
            0.0,
            0.07,
            0.18,
            0.19,
            0.07,
            0.18,
            0.19,
            0.12,
            0.08,
            0.07,
            0.05,
            0.1,
            0.15,
            0.13,
            0.1,
            0.15,
            0.13
        ])

        # Define observation space: possible (x,y,z) positions for Poppy's wrists)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32)  # pas convaincue utilité d'applatir

        # Define Poppy's moovements limits (possible angles for each joints)
        self.joints_limits = {
            # pivots to the left (>0) or to the right (<0) around the z-axis
            'abs_z': (-90.0, 90.0),
            # 'abs_z': (0.0, 0.0),
            # bends forward (>0) or backward (<0)
            'bust_y': (-27.0, 22.0),
            # 'bust_y': (0.0, 0.0),
            # leans to the left (>0) or to the right (<0)
            'bust_x': (-20.0, 20.0),
            # 'bust_x': (0.0, 0.0),

            'r_shoulder_x': (0.0, 180.0),     # lifts elbow up or down
            # moves shoulder forward or backward
            'r_shoulder_y': (-210.0, 65.0),
            # rotates the arm around the shoulder-elbow axis
            'r_arm_z': (-50.0, 60.0),
            # straightens arms (90 full extension => -60 bent elbow)
            'r_elbow_y': (-60.0, 90.0),

            'l_shoulder_x': (0.0, 180.0),     # lifts elbow up or down
            # moves shoulder forward or backward
            'l_shoulder_y': (-210.0, 65.0),
            # rotates the arm around the shoulder-elbow axis
            'l_arm_z': (-50.0, 60.0),
            'l_elbow_y': (-60, 90),         # bends the elbow

            # fix - points the head down (>0) or up(<0) (min -20, max 20)                                   #
            'head_y': (0.0, 0.0),
            # fix - rotates the head to the left (>0) or to the right (<0)  (min -90, max 90)
            'head_z': (0.0, 0.0),
        }
        self.low_limits = np.array(
            [lim[0] for lim in self.joints_limits.values()], dtype=np.float32)
        self.high_limits = np.array(
            [lim[1] for lim in self.joints_limits.values()], dtype=np.float32)

        # Define the action space using these limits
        self.action_space = spaces.Box(
            low=self.low_limits, high=self.high_limits, dtype=np.float32)

        # Define variables for training
        self.current_step = 0
        self.num_steps = 0
        self.target_loaded = False

        self.done = False
        self.infos = []

        self.episodes = 0  # used for resetting the sim every so often
        self.restart_every_n_episodes = 1000

        super().__init__()

    # def seed(self, seed=None):
    #     return [np.random.seed(seed)]

    def poppy_goto(self, joints_to_move, wait_for=3):
        '''function to move poppy to a specific position
        Input : dictionary of an action defined by joints to move (specify only joints to move)
        joints_to_move = {
                    # (-90.0 , 90.0)    pivots to the left (>0) or to the right (<0) around the z-axis
                    'abs_z': 0.0,
                    # (-27.0 , 22.0)    bends forward (>0) or backward (<0)
                    'bust_y': 0.0,
                    # (-20.0, 20.0)     leans to the left (>0) or to the right (<0)
                    'bust_x': 0.0,

                    #  (0 , 180)      lifts elbow up or down
                    'l_shoulder_x': 0.0,
                    # (-210.0 , 65.0) moves shoulder forward or backward
                    'l_shoulder_y': 0.0,
                    # (-50.0, 60.0)   rotates the arm around the shoulder-elbow axis
                    'l_arm_z':  0.0,
                    #  (-60.0, 90.0)  straightens arms (90 full extension => -60 bent elbow)
                    'l_elbow_y': 0.0,

                    #  (0 , 180)      lifts elbow up or down
                    'r_shoulder_x': 0.0,
                    # (-210.0 , 65.0) moves shoulder forward or backward
                    'r_shoulder_y': 0.0,
                    # (-50.0, 60.0)   rotates the arm around the shoulder-elbow axis
                    'r_arm_z':  0.0,
                    #  (-60.0, 90.0)  straightens arms (90 full extension => -60 bent elbow)
                    'r_elbow_y': 0.0,

                    #Optionnal (will be reset to 0)
                    # fix - points the head down (>0) or up(<0) (min -20, max 20)                                   #
                    'head_y': 0.0,
                    # fix - rotates the head to the left (>0) or to the right (<0)  (min -90, max 90)
                    'head_z': 0.0,
                    }
        Out put : None
        Note poppy_goto(joints_reset) is equivalent to poppy_reset()
        '''
        for i, m in enumerate(self.poppy.motors):
            if not np.isnan(joints_to_move[i]):
                # wait=False to allow parallel movements
                m.goto_position(joints_to_move[i], 3, wait=False)

        # wait for moovements to be executed
        time.sleep(wait_for)

    def get_observation(self):
        # Get observations
        targets_obs = np.r_[self.poppy.l_arm_chain.position,
                            self.poppy.r_arm_chain.position]
        l_joints_obs = self.poppy.l_arm_chain.joints_position
        r_joints_obs = self.poppy.r_arm_chain.joints_position
        # targets_obs = np.r_[targets_obs[0], targets_obs[1]]
        return targets_obs, l_joints_obs, r_joints_obs

    def get_poppy_skeletons(self, skeletons):
        '''transforms a human skeleton to poppy's skeleton
        Input : human preprocessed skeleton
        Output : Poppy's skeletons
        '''

        # Get skeletons shape
        n_frames, n_joints, _ = skeletons.shape

        # Measure skeleton bone lengths
        source_lengths = torch.Tensor(n_frames, n_joints)
        for child, parent in enumerate(self.topology):
            source_lengths[:, child] = torch.sqrt(
                torch.sum(
                    (skeletons[:, child] - skeletons[:, parent])**2,
                    axis=-1
                )
            )

        # Find the corresponding angles
        source_offsets = torch.zeros(n_frames, n_joints, 3)
        source_offsets[:, :, -1] = source_lengths
        quaternions = find_quaternions(
            self.topology, source_offsets, skeletons)

        # Use these quaternions in the forward kinematics with the Poppy skeleton
        target_offsets = torch.zeros(n_frames, n_joints, 3)
        target_offsets[:, :, -
                       1] = self.poppy_lengths.unsqueeze(0).repeat(n_frames, 1)
        poppy_skeletons = forward_kinematics(
            self.topology,
            torch.zeros(n_frames, 3),
            target_offsets,
            quaternions
        )[0]

        return poppy_skeletons

    def get_targets_from_skeleton(self, target_skeletons, end_effector_indices=None):
        '''Extracts the (x,y,z) coordinates of target skeleton joints from list of indices for targets
        Inputs :
            skeletons : list of skeletons (n_frames, n_joints, 3)
            indices : list of indices for target joints
            - indices = [13, 16] # left hand, right hand (Default)
            # Chest, head, left hand, left elbow, left shoulder, right hand, right elbow
            - indices = [8, 10, 13, 12, 11, 16, 15]
            # left hand, left elbow, right hand, right elbow
            - indices = = [13, 12, 16, 15]
        Output :
            targets : list of targets (n_frames, n_targets, 3) in skeleton like format
        '''
        if end_effector_indices is None:
            end_effector_indices = [13, 16]
        return target_skeletons[:, end_effector_indices]

    def reset(self, seed=None, **kwargs):
        '''reset Poppy to the initial state'''
        if seed is not None:
            np.random.seed(seed)
        joint_pos = {'l_elbow_y': 0.0,  # if 90 will reset with arms straight along hips
                     'head_y': 0.0,
                     'r_arm_z': 0.0,
                     'head_z': 0.0,
                     'r_shoulder_x': 0.0,
                     'r_shoulder_y': 0.0,
                     'r_elbow_y': 0.0,  # if 90 will reset with arms straight along hips
                     'l_arm_z': 0.0,
                     'abs_z': 0.0,
                     'bust_y': 0.0,
                     'bust_x': 0.0,
                     'l_shoulder_x': 0.0,
                     'l_shoulder_y': 0.0
                     }

        for m in self.poppy.motors:
            # wait=False to allow parallel movements
            m.goto_position(joint_pos[m.name], 1, wait=False)

        self.current_step = 0
        self.done = False

        if self.target_loaded == False:
            self.get_target()
            self.target_loaded = True

        self.num_steps = self.targets.shape[0]
        obs = np.r_[self.poppy.l_arm_chain.position,
                    self.poppy.r_arm_chain.position]

        info = {}

        return np.float32(obs), info

    def reward(self, obs, target, threshold=0.3, alpha=10):
        '''Basic Reward function for the PoppyEnv returns value between [0, 1] based on the distance
        between the end effectors observations and their targets
        Note 1 : exponetial (- alpha * distance) transforms a distance [0 , inf] into a similarity score [0, 1]
        Input :
            obs : observations (end effectors positions) format (2x3 or 1x6, left first)
            target : targets (end effectors positions) same format as observations
            threshold : distance threshold for reward to be non zero
            alpha : hyperparameter to tune for similarity score transformation
        Output : scale'''

        # anaele : 0.3 and alpha =10 are hyperparameters to be tuned
        # Anaele : transform a distance into a similarity score between 0 and 1 (similar)

        # flatten targets and observations
        # obs = obs.flatten()
        # target = target.flatten()
        target = self.get_targets_from_skeleton(
            np.expand_dims(target.numpy(), axis=0))

        # calculate the error distance for left and right end_effector
        l_dis = np.linalg.norm(obs[:3] - target[0][0])
        r_dis = np.linalg.norm(obs[3:] - target[0][1])

        # calculate reward
        l_reward = np.exp(- alpha * l_dis) if l_dis <= threshold else 0
        r_reward = np.exp(- alpha * r_dis) if r_dis <= threshold else 0
        reward = (l_reward**2 + r_reward**2)/2

        return reward

    ###########################################################
    # Pour l'entrainement du modèle
    ###########################################################

    def step(self, action):
        '''entrainement d'un step avec calcul error (distance) et reward
        Input : action dictionary in the format {"joint_name": angle}
        Output :
            obs : observations (end effectors positions) format (2x3 or 1x6, left first)
            reward : reward value
            done : True if the episode is finished
            info : dictionary with episode, step and reward values
        '''

        # move the robot and get observations
        # we will use only end effectors position for now,
        # obs, _, _ = self.poppy_goto(action)
        self.poppy_goto(action)
        obs, _, _ = self.get_observation()

        # calculate the reward (based on similarity of end effectors positions with targets)
        # Note: reward function could penalise moovements that Poppy cannot do using l_joints_obs and r_joints_obs....
        reward = self.reward(obs, self.targets[self.current_step])
        print("reward : ", reward)

        # Update current step, info and episode
        self.current_step += 5
        print("current step : ", self.current_step)

        info = {'episode': self.episodes,
                'step': self.current_step,
                'reward': reward}
        self.infos.append(info)

        self.done = (self.current_step >= self.num_steps)
        if self.done:
            self.episodes += 1
        print("episode : ", self.episodes)

        truncated = False
        info = {}

        return np.float32(obs), reward, self.done, truncated, info

    # use this function to get the targets
    # Anaele : load video, extract position with blazepose and preprocess human skeleton to poppy skeleton
    # def get_target_from_video(self, video, path):
    #     '''Extract keypoints from video with blazepose pre-trained model, preprocess huan skeleton
    #     and apply to poppy to get poppy skeleton'''

    #     # Extract keypoints from video with blazepose pre-trained model
    #     self.skeletons = blazepose_skeletons(path+video)

    #     # Preprocessing avec angle personalisé
    #     # alpha= "auto" pour redressement automatique ou alpha = np.pi/4 pour angle personalisé (à implémenter)
    #     self.skeletons = preprocess_skeletons(self.skeletons, self.topology, ref_joint=0, alpha=np.pi/4,
    #                                           smoothing_n=4, interpolation_factor=1)  # Anaele : il faut faire le preprocessing

    #     targets, all_positions = self.get_targets_from_skeleton(
    #         self.skeletons, self.topology, self.poppy_lengths)  # Anaele transforme aux dimension de poppy (à faire après le smoothing)

    #     interpolated_targets = self.interpolate_targets(targets)

    #     smoothed_targets = self.moving_average(interpolated_targets, n=15)

    #     self.targets = smoothed_targets

    # extraction of the targets
    def get_target(self):
        '''Get target from skeletons saved in ./resources/sample_poppy_skeletons/'''
        path = "./resources/sample_poppy_skeletons/"
        files = os.listdir(path)
        for file in files:
            if file.endswith("poppy_skeletons.pt"):
                print("loading targets from : ", file)
                self.targets = torch.load(path+file)
                break

        # return self.get_targets_from_skeleton(self,
