import gymnasium as gym

from . import agents
from .insertanything_env_cfg import (
    InsertAnythingCircleHole_I_Cfg,
    InsertAnythingHexagonHole_III_Cfg,
    InsertAnythingHexagonHole_III_Iiwa14Bench_Cfg,
    InsertAnythingHexagonHole_III_Iiwa14Pegboard_Cfg,
    InsertAnythingHexagonHole_IV_Cfg,
    InsertAnythingLHole_III_Cfg,
    InsertAnythingLHole_III_Iiwa14Bench_Cfg,
    InsertAnythingLHole_III_Iiwa14Pegboard_Cfg,
    InsertAnythingSquareHole_II_Cfg,
    InsertAnythingTriangleHole_IV_Cfg,
)


def _register(task_id: str, env_cfg_entry_point, rl_games_cfg: str):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.insertanything_env:InsertAnythingEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rl_games_cfg_entry_point": f"{agents.__name__}:{rl_games_cfg}",
        },
    )


_register(
    "InsertAnything-CircleHole-I-Direct-v0",
    InsertAnythingCircleHole_I_Cfg,
    "rl_games_ppo_circle_cfg.yaml",
)
_register(
    "InsertAnything-SquareHole-II-Direct-v0",
    InsertAnythingSquareHole_II_Cfg,
    "rl_games_ppo_square_cfg.yaml",
)
_register(
    "InsertAnything-LHole-III-Direct-v0",
    InsertAnythingLHole_III_Cfg,
    "rl_games_ppo_L_cfg.yaml",
)
_register(
    "InsertAnything-TriangleHole-IV-Direct-v0",
    InsertAnythingTriangleHole_IV_Cfg,
    "rl_games_ppo_triangle_cfg.yaml",
)
_register(
    "InsertAnything-HexagonHole-III-Direct-v0",
    InsertAnythingHexagonHole_III_Cfg,
    "rl_games_ppo_hexagon_cfg.yaml",
)
_register(
    "InsertAnything-HexagonHole-IV-Direct-v0",
    InsertAnythingHexagonHole_IV_Cfg,
    "rl_games_ppo_hexagon_cfg.yaml",
)

# The cable-assembly iiwa 14 + Robotiq 2F-85 cell (same agents as the Franka tasks, so the released
# checkpoints load unchanged).
_register(
    "InsertAnything-HexagonHole-III-Iiwa14Bench-Direct-v0",
    InsertAnythingHexagonHole_III_Iiwa14Bench_Cfg,
    "rl_games_ppo_hexagon_cfg.yaml",
)
_register(
    "InsertAnything-HexagonHole-III-Iiwa14Pegboard-Direct-v0",
    InsertAnythingHexagonHole_III_Iiwa14Pegboard_Cfg,
    "rl_games_ppo_hexagon_cfg.yaml",
)
_register(
    "InsertAnything-LHole-III-Iiwa14Bench-Direct-v0",
    InsertAnythingLHole_III_Iiwa14Bench_Cfg,
    "rl_games_ppo_L_cfg.yaml",
)
_register(
    "InsertAnything-LHole-III-Iiwa14Pegboard-Direct-v0",
    InsertAnythingLHole_III_Iiwa14Pegboard_Cfg,
    "rl_games_ppo_L_cfg.yaml",
)

# USB-A: no policy of its own yet; the hexagon-III agent config (same observation layout).
from .insertanything_env_cfg import USB_ENV_CFGS  # noqa: E402

for _suffix, _cfg in USB_ENV_CFGS.items():
    _register(f"InsertAnything-{_suffix}-Direct-v0", _cfg, "rl_games_ppo_hexagon_cfg.yaml")
