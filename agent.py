"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from lux.config import EnvConfig
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

from scipy.ndimage import distance_transform_cdt
from scipy.spatial import KDTree

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH = "./logs/exp_1/models/best_model"

class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        return dict(faction="AlphaStrike", bid=0)

    '''
    def manhattan_distance(self, binary_mask):
        # Get the distance map from every pixel to the nearest positive pixel
        distance_map = distance_transform_cdt(binary_mask, metric='taxicab')

        return distance_map

    def manhattan_dist_to_nth_closest(self, arr, n):
        if n == 1:
            distance_map = distance_transform_cdt(1-arr, metric='taxicab')
            return distance_map
        else:
            true_coords = np.transpose(np.nonzero(arr)) # get the coordinates of true values
            tree = KDTree(true_coords) # build a KDTree
            dist, _ = tree.query(np.transpose(np.nonzero(~arr)), k=n, p=1) # query the nearest to nth closest distances using p=1 for Manhattan distance
            return np.reshape(dist[:, n-1], arr.shape) # reshape the result to match the input shape and add an extra dimension for the different closest distances

    def count_region_cells(self, array, start, min_dist=2, max_dist=np.inf, exponent=1):
        def dfs(array, loc):
            distance_from_start = abs(loc[0]-start[0]) + abs(loc[1]-start[1])
            if not (0<=loc[0]<array.shape[0] and 0<=loc[1]<array.shape[1]):   # check to see if we're still inside the map
                return 0
            if (not array[loc]) or visited[loc]:     # we're only interested in low rubble, not visited yet cells
                return 0
            if not (min_dist <= distance_from_start <= max_dist):      
                return 0

            visited[loc] = True

            count = 1.0 * exponent**distance_from_start
            count += dfs(array, (loc[0]-1, loc[1]))
            count += dfs(array, (loc[0]+1, loc[1]))
            count += dfs(array, (loc[0], loc[1]-1))
            count += dfs(array, (loc[0], loc[1]+1))

            return count

        visited = np.zeros_like(array, dtype=bool)
        return dfs(array, start)
        
    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        
        if obs["teams"][self.player]["metal"] == 0:
            return dict()
        
        # potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        # potential_spawns_set = set(potential_spawns)

        ice = obs["board"]["ice"]
        dist_ice = self.manhattan_distance(1-ice)

        ice = obs["board"]["ice"]
        ore = obs["board"]["ore"]
        rubble = obs["board"]["rubble"]

        dist_ice = self.manhattan_distance(1 - ice)
        dist_ice = np.max(dist_ice) - dist_ice
        dist_ore = self.manhattan_distance(1 - ore)
        dist_ore = np.max(dist_ore) - dist_ore

        score = dist_ice + dist_ore
        valid_good_spawns = score * obs["board"]["valid_spawns_mask"]

        best_loc = np.argmax(valid_good_spawns)

        # this is the distance to the n-th closest ice, for each coordinate
        ice_distances = [self.manhattan_dist_to_nth_closest(ice, i) for i in range(1,5)]

        # this is the distance to the n-th closest ore, for each coordinate
        ore_distances = [self.manhattan_dist_to_nth_closest(ore, i) for i in range(1,5)]

        ICE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25]) 
        weigthed_ice_dist = np.sum(np.array(ice_distances) * ICE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

        ORE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25])
        weigthed_ore_dist = np.sum(np.array(ore_distances) * ORE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

        ICE_PREFERENCE = 3 # if you want to make ore more important, change to 0.3 for example

        combined_resource_score = (weigthed_ice_dist * ICE_PREFERENCE + weigthed_ore_dist)
        combined_resource_score = (np.max(combined_resource_score) - combined_resource_score) * obs["board"]["valid_spawns_mask"]


        best_loc = np.argmax(combined_resource_score)

        low_rubble = (rubble<25)

        low_rubble_scores = np.zeros_like(low_rubble, dtype=float)

        for i in range(low_rubble.shape[0]):
            for j in range(low_rubble.shape[1]):
                low_rubble_scores[i,j] = self.count_region_cells(low_rubble, (i,j), min_dist=0, max_dist=8, exponent=0.9)

        overall_score = (low_rubble_scores*2 + combined_resource_score ) * obs["board"]["valid_spawns_mask"]

        best_loc = np.argmax(overall_score)

        x, y = np.unravel_index(best_loc, (48, 48))
        
        best_loc = [x, y]
        
        # this will spawn a factory at pos and with all the starting metal and water
        metal = obs["teams"][self.player]["metal"]
        water = obs["teams"][self.player]["water"]
        
        return dict(spawn=best_loc, metal=metal, water=water)
    '''

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: 
        # https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        if obs["teams"][self.player]["metal"] == 0:
            return dict()
        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False

        ice_diff = np.diff(obs["board"]["ice"])
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]

            area = 3
            for x in range(area):
                for y in range(area):
                    check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        if not done_search:
            pos = spawn_loc

        metal = obs["teams"][self.player]["metal"]
        return dict(spawn=pos, metal=150, water=150)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()
        with th.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller 
            # used which will force the agent to generate actions that are valid only.
            action_mask = (
                th.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            
            # SB3 doesn't support invalid action masking. So we do it ourselves here
            features = self.policy.policy.features_extractor(obs.unsqueeze(0))
            x = self.policy.policy.mlp_extractor.shared_net(features.cuda()) # For local evaluation
            # x = self.policy.policy.mlp_extractor.shared_net(features)
            
            logits = self.policy.policy.action_net(x) # shape (1, N) where N=12 for the default controller

            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)
            print('ACTIONS: ', actions, file=sys.stderr)

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, actions[0]
        )

        # commented code below adds watering lichen which can easily improve your agent
        shared_obs = raw_obs[self.player]
        factories = shared_obs["factories"][self.player]
        for unit_id in factories.keys():
            factory = factories[unit_id]
            #print('Paani: ', factory["cargo"]["water"], file=sys.stderr)
            if factory["cargo"]["water"] >= 1000 or ((1000 - step) * 2) < factory["cargo"]["water"]:
                lux_action[unit_id] = 2 # water and grow lichen at the very end of the game
                #print('Putting lichen action', file=sys.stderr)

        return lux_action
