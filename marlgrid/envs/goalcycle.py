from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class ClutteredGoalCycleEnv(MultiGridEnv):
    mission = "Cycle between yellow goal tiles."
    metadata = {}

    def __init__(self, *args, reward=1, penalty=0.0, n_clutter=None, clutter_density=None, n_bonus_tiles=3,
                 initial_reward=True, cycle_reset=False, reset_on_mistake=False, reward_decay=False, **kwargs):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        # Overwrite the default reward_decay for goal cycle environments.
        super().__init__(*args, **{**kwargs, 'reward_decay': reward_decay})

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width - 2) * (self.height - 2))
        else:
            self.n_clutter = n_clutter

        self.reward = reward
        self.penalty = penalty

        self.initial_reward = initial_reward
        self.n_bonus_tiles = n_bonus_tiles
        self.reset_on_mistake = reset_on_mistake

        self.bonus_tiles = []
        self.bonus_tiles_pos = [[] for _ in range(self.n_bonus_tiles)]
        self.wall_pos = [[] for _ in range(self.n_clutter)]

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)

        for bonus_id in range(getattr(self, 'n_bonus_tiles', 0)):
            color = "yellow"
            if bonus_id == 0:
                color = "yellow"
            elif bonus_id == 1:
                color = "purple"
            else:
                color = "green"
            BonusTile_pos = self.place_obj(BonusTile(
                color=color,
                reward=self.reward,
                penalty=self.penalty,
                bonus_id=bonus_id,
                n_bonus=self.n_bonus_tiles,
                initial_reward=self.initial_reward,
                reset_on_mistake=self.reset_on_mistake,
            ), max_tries=100)
            self.bonus_tiles_pos[bonus_id] = BonusTile_pos

        for wall_id in range(getattr(self, 'n_clutter', 0)):
            wall_pos = self.place_obj(Wall(), max_tries=100)
            self.wall_pos[wall_id] = wall_pos

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)
