
import time
import numpy as np
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import threading

# Import environment
import gymnasium as gym
from gymnasium import spaces
from pypot.creatures import PoppyTorso

# Import custom functions
# from utils.skeleton import *
# from utils.quaternion import *
from utils.blazepose import blazepose_skeletons
from utils.video_capturing import preprocess_skeletons

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
        # adjusted observation space to Poppy's abilities :
        # left wrist x = [-0.2, 0.5] y = [-0.4, 0.4] z = [-0.2, 0.6]  !!! x for left and right wrist inverted
        # right wrist x = [-0.5, 0.2] y = [-0.4, 0.4] z = [-0.2, 0.6]
        self.low_limits = np.array([ -0.2, -0.40, -0.2, -0.5, -0.40, 0.2], dtype=np.float32)
        self.high_limits = np.array([ 0.5,  0.40,  0.6, 0.2,  0.40, 0.6], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.low_limits, high=self.high_limits, dtype=np.float32)
        # observation space by default
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)  

        # Define Poppy's moovements limits (possible angles for each joints)
        # all bust and head joints are fix (no moovement allowed) to speed up training
        # certain joints are limited to speed up training
        self.joints_limits = {
            'l_elbow_y': (-45, 90),         # (-60, 90) bends the elbow
            'head_y': (0.0, 0.0),           # fix (min -90, max 90) - points the head down (>0) or up(<0)  
            'r_arm_z': (-45.0, 55.0),       # (-50.0, 60.0) rotates the arm around the shoulder-elbow axis 
            'head_z': (0.0, 0.0),           # fix (min -90, max 90) - rotates the head to the left (>0) or to the right (<0)  
            'r_shoulder_x': (-180.0, 5.0), # (-180.0, 15.0) lifts elbow up or down
            'r_shoulder_y': (-180.0, 50.0), # (-210.0, 65.0) moves shoulder forward or backward
            'r_elbow_y': (-45.0, 90.0),     # (-60.0, 90.0)straightens arms (90 full extension => -60 bent elbow)
            'l_arm_z': (-30.0, 30.0),       # (-50.0, 60.0) rotates the arm around the shoulder-elbow axis
            'abs_z': (0.0, 0.0),            # fix (-90.0, 90.0) pivots to the left (>0) or to the right (<0) around the z-axis
            'bust_y': (0.0, 0.0),           # fix (min -27, max 22) - bends forward (>0) or backward (<0)
            'bust_x': (0.0, 0.0),           # fix (min -20, max 20) - leans to the left (>0) or to the right (<0)
            'l_shoulder_x': (-5.0, 180.0),  # (-15.0, 180.0)lifts elbow up or down
            'l_shoulder_y': (-180.0, 65.0), # (-180.0, 65.0) moves shoulder forward or backward
          }

        self.low_limits = np.array(
            [lim[0] for lim in self.joints_limits.values()], dtype=np.float32)
        self.high_limits = np.array(
            [lim[1] for lim in self.joints_limits.values()], dtype=np.float32)

        # Define the action space using these limits
        self.action_space = spaces.Box(
            low=self.low_limits, high=self.high_limits, dtype=np.float32)

        # Define variables for training
        #self.obs = np.r_[self.poppy.l_arm_chain.position,
        #            self.poppy.r_arm_chain.position]
        # self.joints_obs = np.zeros(13)
        self.current_step = 0
        self.num_steps = 0 # must be set to self.targets.shape[0] when target is defined
        self.target_loaded = False

        self.done = False
        self.infos = []

        self.episodes = 0  # used for resetting the sim every so often
        # self.restart_every_n_episodes = 1000 # to be implemented

        super().__init__()

    ###########################################################
    # To Operate Poopy robot in Coppelia Simulator
    ###########################################################

    def get_mooves_array_from_dict(self, actions_dict):
        '''Get an array of moovements from a dictionary of moovements
        Input : dictionary of moovements (may contain from 1 to 13 angles, depending on the moovements to be executed)
        Output : array of moovements'''
        
        # Re-order mooves according to Poppy's motors index
        order = ['l_elbow_y','head_y','r_arm_z','head_z',
                 'r_shoulder_x','r_shoulder_y','r_elbow_y','l_arm_z',
                 'abs_z','bust_y','bust_x','l_shoulder_x','l_shoulder_y']
        # Create array of moovements, following the order of Poppy's motors (give 0 if motor is not in the dictionary)
        mooves_array = [actions_dict[m] if m in actions_dict.keys() else 0 for m in order]

        return mooves_array
    
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
            m.goto_position(joint_pos[m.name], 1, wait=True)

        # wait 3 seconds for moovements to be executed
        # time.sleep(3) # quote if wait=True

        self.current_step = 0
        self.done = False

        if self.target_loaded == False:
            self.get_target()
            self.target_loaded = True
            self.num_steps = self.targets.shape[0]-1 # must be set to self.targets.shape[0] when target is defined

        # self.num_steps = self.targets.shape[0]-1

        obs = np.r_[self.poppy.l_arm_chain.position,
                    self.poppy.r_arm_chain.position]

        info = {}

        return np.float32(obs), info

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

        for i, m in enumerate(self.poppy.motors):
            if not np.isnan(joints_to_move[i]):
                # wait=False to allow parallel movements (unquote time.sleep(wait_for) to wait for movements to be executed)
                m.goto_position(joints_to_move[i], 3, wait=True)

        # wait for moovements to be executed
        # time.sleep(wait_for)


    def poppy_goto_threads(self, joints_to_move):
        ''' NOT WORKING YET
        function to move poppy to a specific position using threads to use simultaneous moves
        and still be able to wait all moves are finished
            
        move_complete_events = [threading.Event() for _ in self.poppy.motors]  # Define move_complete_events

        for i, m in enumerate(self.poppy.motors):
            if not np.isnan(joints_to_move[i]):
                # wait=False to allow parallel movements (unquote time.sleep(wait_for) to wait for movements to be executed)
                m.goto_position(joints_to_move[i], 3, wait=False)

        # wait for all moovements to be executed
        # wait for all movements to be executed
        for event in move_complete_events:
            event.wait()
        # time.sleep(wait_for)'''

    ###########################################################
    # TO TRAIN THE MODEL
    ###########################################################

    def get_observation(self):
        '''Get observations from PoppyEnv
        Output : 
            - observations (end effectors positions) format (2x3 or 1x6, left first)
            - joints observed position : array (dim=13) with bust (3 motors), head (2 motors), left arm (4 motors), right arm (4)
        '''
        # Get observations
        targets_obs = np.r_[self.poppy.l_arm_chain.position,
                            self.poppy.r_arm_chain.position]
        l_joints_obs = self.poppy.l_arm_chain.joints_position
        r_joints_obs = self.poppy.r_arm_chain.joints_position

        # get joints observed position infollowing order: bust (3), head (2), left arm (4), right arm (4) and x,y,z coordinatess
        joints_pos = np.r_[l_joints_obs[2], l_joints_obs[1], l_joints_obs[0], # bust x,y,z
                           self.poppy.head_y.present_position, self.poppy.head_z.present_position, # head
                           l_joints_obs[4], l_joints_obs[3] , l_joints_obs[5:], # left arm motors shoulder x,y,z and elbow
                           r_joints_obs[4], r_joints_obs[3] , r_joints_obs[5:]] # right arm motors shoulder x,y,z and elbow
        
        return targets_obs, joints_pos

    def reward(self, obs, target, joints = None, threshold=0.3, alpha=2):
        '''Basic Reward function for the PoppyEnv returns value between [0, 1] based on the distance
        between the end effectors observations and their targets
        Note 1 : exponetial (- alpha * distance) transforms a distance [0 , inf] into a similarity score [0, 1]
        Input :
            obs : observations (end effectors positions) format (2x3 or 1x6, left first)
            target : targets (end effectors positions) same format as observations
            threshold : distance threshold for reward to be non zero
            alpha : hyperparameter to tune for similarity score transformation
        Output : scalar, reward value'''

        # anaele : threshold=0.3 and alpha =2 are hyperparameters to be tuned
        
        obs_np = np.array(obs)
        target = np.array(target)
        
        # calculate the error distance for left and right end_effector
        l_dis = np.linalg.norm(obs_np[:3] - target[0][0])
        r_dis = np.linalg.norm(obs_np[3:] - target[0][1])

        # calculate reward
        #-----------------
        # reward for distance between end effectors and targets
        l_reward = np.exp(- alpha * l_dis) # if l_dis <= threshold else 0
        r_reward = np.exp(- alpha * r_dis) # if r_dis <= threshold else 0

        # set penalty if head or bust observation are not = 0 (as they are fix, is !=0 then arm has hit head or body or table )
        joints_reward = - np.abs(joints[:5]).sum() if joints is not None else 0

        # set penalty if z position of end effectors is too low (as they should be above the table)
        # l_z_reward = -10 if obs_np[2] < 0.01 else 0
        # r_z_reward = -10 if obs_np[5] < 0.01 else 0

        reward = l_reward + r_reward + joints_reward # + l_z_reward + r_z_reward

        return reward
      
    def get_target(self):
        '''Get target from skeletons saved in ./resources/sample_poppy_skeletons/'''
        path = "./resources/sample_poppy_skeletons/"
        # files = os.listdir(path)
        file = 'anaele_bent_arms_0_poppy_skeletons.pt' # files[0]
              
        if file.endswith("poppy_skeletons.pt"):
            print("loading targets from : ", file)
            self.targets = torch.load(path+file)
        else:
            print("File must be in a poppy skeletons format")
    
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
        # we will use only end effectors and joints positions 
        self.poppy_goto(action)
        obs, joints_pos = self.get_observation()
        joints_str = ", ".join([f"{j:.1f}" for j in joints_pos]) # for printing in a compact way
                               
        # calculate the reward (based on similarity of end effectors and targets and on 
        # anomalies on bust and head joints (which should stay = 0 as these motors are neutralised in action space))
        reward = self.reward(obs, self.targets[self.current_step],joints = joints_pos)
        print(f"{self.episodes}-{self.current_step} Reward: {reward:.2f} Joints: [{joints_str}]")

        if reward < -5:
            self.poppy_goto(np.zeros(13))
            print("Anomaly detected, Poppy reset, training continues...")
        
        # Update current step, info and episode
        self.current_step += 1

        info = {'episode': self.episodes,
                'step': self.current_step,
                'reward': reward,
                'obs': obs,
                'joint_pos': joints_pos}
        self.infos.append(info)

        self.done = (self.current_step >= self.num_steps)

        if self.done:
            self.episodes += 1

        truncated = False
        info = {}

        return np.float32(obs), reward, self.done, truncated, info

    ###########################################################
    # TO CAPTURE VIDEO, EXTRAC TSKELETON AND GET TARGETS
    ###########################################################
    # Anaele : loads video, extract position with blazepose and preprocess human skeleton to poppy skeleton

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
    
    def get_target_from_video(self, video, path):
        '''Extract keypoints from video with blazepose pre-trained model, p
        reprocess huan skeleton
        and apply to poppy to get poppy skeleton'''

        # Extract keypoints from video with blazepose pre-trained model
        self.skeletons = blazepose_skeletons(path+video)

        # Preprocessing avec angle personalisé
        # angle = "auto" pour redressement automatique ou alpha = np.pi/4 pour angle personalisé (à implémenter)
        self.skeletons = preprocess_skeletons(self.skeletons, self.topology, ref_joint=None, angle=None,
                                               smoothing_n=4, interpolation_factor=1)  # Anaele : il faut faire le preprocessing
        self.skeletons = self.get_poppy_skeletons(self.skeletons)
        
        
        self.targets, all_positions = self.get_targets_from_skeleton(
                self.skeletons, self.topology, self.poppy_lengths)  
        self.targets = self.interpolate_targets(self.targets)
        self.targets = self.moving_average(self.targets, n=15)
