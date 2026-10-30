
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TumorRefinementEnv(gym.Env):
    """
    RL environment for refining three-channel U-Net predictions.

    Observation:
        4 MRI channels
        3 U-Net probability maps
        3 current binary prediction maps

    Actions:
        0: decrease channel 0 threshold
        1: increase channel 0 threshold
        2: decrease channel 1 threshold
        3: increase channel 1 threshold
        4: decrease channel 2 threshold
        5: increase channel 2 threshold
        6: stop refinement
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=10, threshold_step=0.05):
        super().__init__()

        self.max_steps = max_steps
        self.threshold_step = threshold_step

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10, 240, 240),
            dtype=np.float32,
        )

        self.image = None
        self.probabilities = None
        self.targets = None
        self.thresholds = None
        self.predictions = None
        self.current_step = 0

    def _to_chw(self, array, channels):
        array = np.asarray(array, dtype=np.float32)

        if array.shape == (240, 240, channels):
            array = np.transpose(array, (2, 0, 1))

        if array.shape != (channels, 240, 240):
            raise ValueError(
                f"Expected {(channels, 240, 240)}, "
                f"received {array.shape}"
            )

        return array

    def _get_dice(self, predictions):
        dice_scores = []

        for channel in range(3):
            pred = predictions[channel].astype(bool)
            target = self.targets[channel].astype(bool)

            intersection = np.logical_and(pred, target).sum()
            denominator = pred.sum() + target.sum()

            if denominator == 0:
                dice = 1.0
            else:
                dice = (2.0 * intersection) / denominator

            dice_scores.append(dice)

        return float(np.mean(dice_scores))

    def _get_observation(self):
        observation = np.concatenate(
            [
                self.image,
                self.probabilities,
                self.predictions.astype(np.float32),
            ],
            axis=0,
        )

        return np.clip(observation, 0.0, 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is None:
            options = {}

        required = ["image", "probabilities", "targets"]

        for key in required:
            if key not in options:
                raise ValueError(
                    f"Missing '{key}' in reset options"
                )

        self.image = self._to_chw(options["image"], 4)
        self.probabilities = self._to_chw(
            options["probabilities"], 3
        )
        self.targets = self._to_chw(options["targets"], 3)

        # Normalize MRI channels for the RL agent's observation.
        for channel in range(4):
            image_channel = self.image[channel]
            low, high = np.percentile(image_channel, [1, 99])
            if high > low:
                self.image[channel] = np.clip(
                    (image_channel - low) / (high - low),
                    0.0,
                    1.0,
                )
            else:
                self.image[channel] = 0.0

        self.probabilities = np.clip(
            self.probabilities, 0.0, 1.0
        )

        self.targets = (
            self.targets >= 0.5
        ).astype(np.float32)

        self.thresholds = np.full(3, 0.5, dtype=np.float32)

        self.predictions = (
            self.probabilities >= self.thresholds[:, None, None]
        )

        self.current_step = 0
        self.previous_dice = self._get_dice(self.predictions)

        observation = self._get_observation()

        info = {
            "dice": self.previous_dice,
            "thresholds": self.thresholds.copy(),
        }

        return observation, info

    def step(self, action):
        if self.image is None:
            raise RuntimeError("Call reset() before step().")

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError("Invalid action")

        if action == 6:
            observation = self._get_observation()

            return observation, 0.0, True, False, {
                "dice": self.previous_dice,
                "thresholds": self.thresholds.copy(),
            }

        channel = action // 2

        if action % 2 == 0:
            self.thresholds[channel] -= self.threshold_step
        else:
            self.thresholds[channel] += self.threshold_step

        self.thresholds = np.clip(
            self.thresholds, 0.05, 0.95
        )

        self.predictions = (
            self.probabilities
            >= self.thresholds[:, None, None]
        )

        current_dice = self._get_dice(self.predictions)

        reward = current_dice - self.previous_dice
        self.previous_dice = current_dice

        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()

        info = {
            "dice": current_dice,
            "thresholds": self.thresholds.copy(),
        }

        return observation, float(reward), terminated, truncated, info
