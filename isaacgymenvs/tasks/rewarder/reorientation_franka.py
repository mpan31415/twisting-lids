import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from isaacgymenvs.tasks.rewarder.base import BaseRewardFunction

from isaacgymenvs.tasks.allegro_franka_utils import tolerance_successes_objective


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def xyzw_to_wxyz(quat):
    new_quat = quat.clone()
    new_quat[:, :1] = quat[:, -1:]
    new_quat[:, 1:] = quat[:, :-1]
    return new_quat


class FrankaReorientationRewardFunction(BaseRewardFunction):
    def __init__(self, **kwargs) -> None:
        
        print("=" * 50)
        print("FrankaReorientationRewardFunction")
        print("=" * 50)

        super().__init__(**kwargs)

        # self.rot_score = torch.zeros((self.num_envs,)).to(self.device)
        # self.z_score = torch.zeros((self.num_envs,)).to(self.device)

        return
    
    def _lifting_reward(self, object_pos, object_init_state, reward_settings) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""

        z_lift = 0.05 + object_pos[:, 2] - object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)

        # this flag tells us if we lifted an object above a certain height compared to the initial position
        lifted_object = (z_lift > reward_settings["lifting_bonus_threshold"]) | self.lifted_object

        # Since we stop rewarding the agent for height after the object is lifted, we should give it large positive reward
        # to compensate for "lost" opportunity to get more lifting reward for sitting just below the threshold.
        # This bonus depends on the max lifting reward (lifting reward coeff * threshold) and the discount factor
        # (i.e. the effective future horizon for the agent)
        # For threshold 0.15, lifting reward coeff = 3 and gamma 0.995 (effective horizon ~500 steps)
        # a value of 300 for the bonus reward seems reasonable
        just_lifted_above_threshold = lifted_object & ~self.lifted_object
        lift_bonus_rew = reward_settings["lifting_bonus"] * just_lifted_above_threshold

        # stop giving lifting reward once we crossed the threshold - now the agent can focus entirely on the
        # keypoint reward
        lifting_rew *= ~lifted_object

        # update the flag that describes whether we lifted an object above the table or not
        self.lifted_object = lifted_object
        return lifting_rew, lift_bonus_rew, lifted_object
    
    def _distance_delta_rewards(self, curr_fingertip_distances, lifted_object: Tensor) -> Tensor:
        """Rewards for fingertips approaching the object or penalty for hand getting further away from the object."""
        # this is positive if we got closer, negative if we're further away than the closest we've gotten
        fingertip_deltas_closest = self.closest_fingertip_dist - curr_fingertip_distances
        # update the values if finger tips got closer to the object
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, curr_fingertip_distances)

        # clip between zero and +inf to turn deltas into rewards
        fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        fingertip_delta_rew = torch.sum(fingertip_delta_rew, dim=-1)  # sum over all arms

        # vvvv this is commented out for 2 arms: we want the 2nd arm to be relatively close at all times

        # add this reward only before the object is lifted off the table
        # after this, we should be guided only by keypoint and bonus rewards
        # fingertip_delta_rew *= ~lifted_object

        return fingertip_delta_rew

    def _keypoint_reward(self, keypoints_max_dist, lifted_object: Tensor) -> Tensor:
        # this is positive if we got closer, negative if we're further away
        max_keypoint_deltas = self.closest_keypoint_max_dist - keypoints_max_dist

        # update the values if we got closer to the target
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, keypoints_max_dist)

        # clip between zero and +inf to turn deltas into rewards
        max_keypoint_deltas = torch.clip(max_keypoint_deltas, 0, 100)

        # administer reward only when we already lifted an object from the table
        # to prevent the situation where the agent just rolls it around the table
        keypoint_rew = max_keypoint_deltas * lifted_object

        return keypoint_rew

    def _compute_resets(self, object_pos, reset_buf, max_consecutive_successes, max_episode_length, is_success):
        resets = torch.where(object_pos[:, 2] < 0.1, torch.ones_like(reset_buf), reset_buf)  # fall
        if max_consecutive_successes > 0:
            # Reset progress buffer if max_consecutive_successes > 0
            self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
            resets = torch.where(self.successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(self.progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)
        resets = self._extra_reset_rules(resets)
        return resets
    
    def _extra_reset_rules(self, resets):
        return resets

    def _true_objective(self, success_tolerance, initial_tolerance, target_tolerance, successes):
        true_objective = tolerance_successes_objective(
            success_tolerance, initial_tolerance, target_tolerance, successes
        )
        return true_objective

    def forward(
        self,
        reward_settings: dict,
        object_pos: Tensor,
        object_init_state: Tensor,
        curr_fingertip_distances: Tensor,
        keypoints_max_dist: Tensor,
        success_tolerance: float,
        keypoint_scale: float,
        success_steps: int,
        successes: Tensor,
        reset_goal_buf: Tensor,
        rewards_episode: dict,
        max_consecutive_successes: int,
        max_episode_length: int,
        prev_episode_successes: Tensor,

    ):
        info_dict = {}

        ################### REWARDS ###################
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward(object_pos, object_init_state, reward_settings)
        fingertip_delta_rew = self._distance_delta_rewards(curr_fingertip_distances, lifted_object)
        keypoint_rew = self._keypoint_reward(keypoints_max_dist, lifted_object)

        keypoint_success_tolerance = success_tolerance * keypoint_scale

        # noinspection PyTypeChecker
        near_goal: Tensor = keypoints_max_dist <= keypoint_success_tolerance
        self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= success_steps
        goal_resets = is_success
        successes += is_success

        reset_goal_buf[:] = goal_resets

        rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        rewards_episode["raw_lifting_rew"] += lifting_rew
        rewards_episode["raw_keypoint_rew"] += keypoint_rew

        fingertip_delta_rew *= reward_settings["distance_delta_rew_scale"]
        lifting_rew *= reward_settings["lifting_rew_scale"]
        keypoint_rew *= reward_settings["keypoint_rew_scale"]

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        # We spread out the reward over "success_steps"
        bonus_rew = near_goal * (reward_settings["reach_goal_bonus"] / success_steps)

        reward = fingertip_delta_rew + lifting_rew + lift_bonus_rew + keypoint_rew + bonus_rew

        ################### RESETS ###################
        reset_buf = self._compute_resets(object_pos, reset_buf, max_consecutive_successes, max_episode_length, is_success)

        info_dict["successes"] = prev_episode_successes.mean()
        self.true_objective = self._true_objective(success_tolerance, initial_tolerance, target_tolerance, successes)
        info_dict["true_objective"] = self.true_objective

        # scalars for logging
        info_dict["true_objective_mean"] = self.true_objective.mean()
        info_dict["true_objective_min"] = self.true_objective.min()
        info_dict["true_objective_max"] = self.true_objective.max()

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (bonus_rew, "bonus_rew"),
        ]

        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        info_dict["rewards_episode"] = self.rewards_episode
        info_dict["episode_cumulative"] = episode_cumulative

        

        return reward, reset_goal_buf, rewards_episode, reset_buf, info_dict
    

    def reset(self, env_ids):
        self.z_score[env_ids] = 0
        self.rot_score[env_ids] = 0
        return


def build(**kwargs):
    return FrankaReorientationRewardFunction(**kwargs)
