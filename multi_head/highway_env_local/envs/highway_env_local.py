import numpy as np
#from gym_local.envs.registration import register
from gym.envs.registration import register

from highway_env import utils

from ..envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle


class HighwayEnvLocal(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 30,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -10,
            ##############yaeli
            "right_lane_reward_1": 5,
            "high_speed_reward_1": 0,
            "lane_change_reward_1": 0,

            "right_lane_reward_2": 0,
            "high_speed_reward_2": 5,
            "lane_change_reward_2": 0,

            "right_lane_reward_3": 0,
            "high_speed_reward_3": 0,
            "lane_change_reward_3": 5,
            "real_time_rendering": True,
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,

        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=5,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                self.road.vehicles.append(
                    other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                )

    #def _reward(self, action: Action) -> float:

    def _reward(self, action: Action) -> float:
        lane_change = action == 0 or action == 2
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle,
                                                               ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"],
                                  [0, 1])
        """reward 1"""
        reward = \
            + self.config["lane_change_reward_1"] * lane_change \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward_1"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward_1"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["lane_change_reward_1"] + self.config[
                                 "high_speed_reward_1"] + self.config[
                                 "right_lane_reward_1"]],
                            [0, 1])
        reward_1 = 0 if not self.vehicle.on_road else reward

        """reward 2"""
        reward = \
            + self.config["lane_change_reward_2"] * lane_change \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward_2"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward_2"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["lane_change_reward_2"] + self.config[
                                 "high_speed_reward_2"] + self.config["right_lane_reward_2"]],
                            [0, 1])
        reward_2 = 0 if not self.vehicle.on_road else reward

        """reward 3"""
        reward = \
            + self.config["lane_change_reward_3"] * lane_change \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward_3"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward_3"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["lane_change_reward_3"] + self.config[
                                 "high_speed_reward_3"] + self.config[
                                 "right_lane_reward_3"]],
                            [0, 1])
        reward_3 = 0 if not self.vehicle.on_road else reward

        return np.array([reward_1, reward_2, reward_3])



    # def _reward_1(self, action: Action) -> float:
    #      """
    #       The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
    # #     :param action: the last action performed
    # #     :return: the corresponding reward
    # #     """
    #      lane_change = action == 0 or action == 2
    #      neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #      lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
    #          else self.vehicle.lane_index[2]
    #      scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
    #      reward = \
    #          + self.config["lane_change_reward_1"] * lane_change \
    #          + self.config["collision_reward"] * self.vehicle.crashed \
    #          + self.config["right_lane_reward_1"] * lane / max(len(neighbours) - 1, 1) \
    #          + self.config["high_speed_reward_1"] * np.clip(scaled_speed, 0, 1)
    #      reward = utils.lmap(reward,
    #                          [self.config["collision_reward"],
    #                           self.config["lane_change_reward_1"] + self.config["high_speed_reward_1"] + self.config[
    #                               "right_lane_reward_1"]],
    #                          [0, 1])
    #      reward = 0 if not self.vehicle.on_road else reward
    #      return reward
    # #
    # def _reward_2(self, action: Action) -> float:
    #     """
    #     The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
    #     :param action: the last action performed
    #     :return: the corresponding reward
    #     """
    #     lane_change = action == 0 or action == 2
    #     neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #     lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
    #         else self.vehicle.lane_index[2]
    #     scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
    #     reward = \
    #         + self.config["lane_change_reward_2"] * lane_change\
    #         + self.config["collision_reward"] * self.vehicle.crashed \
    #         + self.config["right_lane_reward_2"] * lane / max(len(neighbours) - 1, 1) \
    #         + self.config["high_speed_reward_2"] * np.clip(scaled_speed, 0, 1)
    #     reward = utils.lmap(reward,
    #                       [self.config["collision_reward"],
    #                        self.config["lane_change_reward_2"]+self.config["high_speed_reward_2"] + self.config["right_lane_reward_2"]],
    #                       [0, 1])
    #     reward = 0 if not self.vehicle.on_road else reward
    #     return reward
    #
    # def _reward_3(self, action: Action) -> float:
    #     """
    #      The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
    #      :param action: the last action performed
    #      :return: the corresponding reward
    #      """
    #     lane_change = action == 0 or action == 2
    #     neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #     lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
    #         else self.vehicle.lane_index[2]
    #     scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
    #     reward = \
    #         + self.config["lane_change_reward_3"] * lane_change \
    #         + self.config["collision_reward"] * self.vehicle.crashed \
    #         + self.config["right_lane_reward_3"] * lane / max(len(neighbours) - 1, 1) \
    #         + self.config["high_speed_reward_3"] * np.clip(scaled_speed, 0, 1)
    #     reward = utils.lmap(reward,
    #                         [self.config["collision_reward"],
    #                          self.config["lane_change_reward_3"] + self.config["high_speed_reward_3"] + self.config[
    #                              "right_lane_reward_3"]],
    #                         [0, 1])
    #     reward = 0 if not self.vehicle.on_road else reward
    #     return reward

    # def _reward_1(self, action: Action):
    #     neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #     lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
    #                  else self.vehicle.lane_index[2]
    #     reward= self.config["right_lane_reward_1"] * lane / max(len(neighbours) - 1, 1)
    #     if self.vehicle.crashed: reward = self.config["collision_reward"]
    #     if not self.vehicle.on_road: reward = self.config["collision_reward"]
    #     return reward
    #
    # def _reward_2(self, action: Action):
    #     scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
    #     self.config["high_speed_reward_2"] * np.clip(scaled_speed, 0, 1)
    #     if self.vehicle.crashed: reward = self.config["collision_reward"]
    #     if not self.vehicle.on_road: reward = self.config["collision_reward"]
    #     return reward
    #
    # def _reward_3(self, action: Action):
    #     lane_change = action == 0 or action == 2
    #     self.config["lane_change_reward_3"] * lane_change
    #     if self.vehicle.crashed: reward = self.config["collision_reward"]
    #     if not self.vehicle.on_road: reward = self.config["collision_reward"]
    #     return reward

    # def _reward_1(self, action: Action) -> float:
    #      neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #      lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
    #          else self.vehicle.lane_index[2]
    #      scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
    #      lane_change = action == 0 or action == 2
    #      reward = \
    #          + self.config["collision_reward"] * self.vehicle.crashed \
    #          + self.config["right_lane_reward_1"] * lane / max(len(neighbours) - 1, 1) \
    #          + self.config["high_speed_reward_1"] * np.clip(scaled_speed, 0, 1) \
    #          + self.config["lane_change_reward_1"] * lane_change
    #      reward = utils.lmap(reward,
    #                        [self.config["collision_reward"],
    #                         self.config["high_speed_reward_1"] + self.config["right_lane_reward_1"]+self.config["lane_change_reward_1"]],
    #                        [0, 1])
    #      reward = 0 if not self.vehicle.on_road else reward
    #      return reward
    #
    # def _reward_2(self, action: Action) -> float:
    #      neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #      lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
    #          else self.vehicle.lane_index[2]
    #      scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
    #      lane_change = action == 0 or action == 2
    #      reward = \
    #          + self.config["collision_reward"] * self.vehicle.crashed \
    #          + self.config["right_lane_reward_2"] * lane / max(len(neighbours) - 1, 1) \
    #          + self.config["high_speed_reward_2"] * np.clip(scaled_speed, 0, 1) \
    #          + self.config["lane_change_reward_2"] * lane_change
    #      reward = utils.lmap(reward,
    #                        [self.config["collision_reward"],
    #                         self.config["high_speed_reward_2"] + self.config["right_lane_reward_2"]]+self.config["lane_change_reward_2"],
    #                        [0, 1])
    #      reward = 0 if not self.vehicle.on_road else reward
    #      return reward
    #
    # def _reward_3(self, action: Action) -> float:
    #      neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #      lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
    #          else self.vehicle.lane_index[2]
    #      scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
    #      lane_change = action == 0 or action == 2
    #      reward = \
    #          + self.config["collision_reward"] * self.vehicle.crashed \
    #          + self.config["right_lane_reward_3"] * lane / max(len(neighbours) - 1, 1) \
    #          + self.config["high_speed_reward_3"] * np.clip(scaled_speed, 0, 1) \
    #          + self.config["lane_change_reward_3"] * lane_change
    #      reward = utils.lmap(reward,
    #                        [self.config["collision_reward"],
    #                         self.config["high_speed_reward_3"] + self.config["right_lane_reward_3"]]+self.config["lane_change_reward_3"],
    #                        [0, 1])
    #      reward = 0 if not self.vehicle.on_road else reward
    #      return reward




    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


register(
    id='highway_local-v0',
    entry_point='highway_env_local.envs.highway_env_local:HighwayEnvLocal',
)
