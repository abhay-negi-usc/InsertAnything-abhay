
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg, PhysxRigidBodyCfg

from .insertanything_iiwa_cell_cfg import (
    InsertAnythingHexagonHole_III_Iiwa14Bench,
    InsertAnythingHexagonHole_III_Iiwa14Pegboard,
    InsertAnythingLHole_III_Iiwa14Bench,
    InsertAnythingLHole_III_Iiwa14Pegboard,
)
from .insertanything_tasks_cfg import (
    InsertAnythingCircleHole_I,
    InsertAnythingHexagonHole_III,
    InsertAnythingHexagonHole_IV,
    InsertAnythingLHole_III,
    InsertAnythingSquareHole_II,
    InsertAnythingTask,
    InsertAnythingTriangleHole_IV,
)

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "fingertip_quat_rel_fixed": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "contact_force": 3,
}


STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "fingertip_quat_rel_fixed": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "contact_force": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "dr_friction": 1,
    "dr_dead_zone": 6,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos_static_error: list = [0.001, 0.001, 0.001]
    fingertip_quat_static_error: list = [0.08, 0.08, 0.01]

    use_all_noise: bool = True
    fixed_asset_pos: list = [0.0005, 0.0005, 0.0005]
    fingertip_pos: list = [0.0005, 0.0005, 0.0005]
    fingertip_quat: list = [0.04, 0.04, 0.005]
    ee_linvel: list = [0.01, 0.01, 0.01]
    ee_angvel: list = [0.01, 0.01, 0.01]
    contact_force: list = [0.1, 0.1, 0.1]


@configclass
class CtrlCfg:
    ema_factor: float = 0.2

    pos_action_bounds: list = [0.05, 0.05, 0.05]
    rot_action_bounds: list = [1.0, 1.0, 1.0]
    pos_action_threshold: list = [0.02, 0.02, 0.02]
    rot_action_threshold: list = [0.097, 0.097, 0.097]

    reset_joints: list = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains: list = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale: float = 10.0
    default_task_prop_gains: list = [100, 100, 100, 30, 30, 30]

    default_dof_pos_tensor: list = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null: float = 10.0
    kd_null: float = 6.3246

    inertia_shaping: str = "none"
    """"none" (the released controller), "nominal" or "nominal_fext": shape the arm's task-space
    inertia to ``desired_task_inertia`` (insertanything_control.compute_dof_torque)."""
    desired_task_inertia: list | None = None
    """6 (diagonal) or 36 (row-major) values, task frame: the reference arm's fingertip inertia."""
    payload_gravity_comp: bool = False
    """Feed-forward the held asset's weight so that, in the task frame, it acts along -z as in the
    released (hole-up) setup: F = m (g_ref - g_task). A no-op when the task frame's z is up; on the
    pegboard (task z horizontal) it removes the sideways weight the policy never saw, which the
    stiffness-free action interface (F = Kp * 0.02 * a - Kd * v) otherwise turns into a steady drift."""


FRANKA_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="",
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        collision_props=PhysxCollisionCfg(contact_offset=0.0005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.00871,
            "panda_joint2": -0.10368,
            "panda_joint3": -0.00794,
            "panda_joint4": -1.49139,
            "panda_joint5": -0.00083,
            "panda_joint6": 1.38774,
            "panda_joint7": 0.0,
            "panda_finger_joint2": 0.04,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        "panda_arm1": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            joint_effort_limit=87,
            joint_velocity_limit=124.6,
        ),
        "panda_arm2": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            joint_effort_limit=12,
            joint_velocity_limit=149.5,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint[1-2]"],
            joint_effort_limit=40.0,
            joint_velocity_limit=0.04,
            stiffness=7500.0,
            damping=173.0,
            friction=0.1,
            armature=0.0,
        ),
    },
)



@configclass
class InsertAnythingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.0, 0.0, 0.70711, 0.70711)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = FRANKA_CFG
    fixed_asset: ArticulationCfg | None = None
    held_asset: ArticulationCfg | None = None
    contact_sensor: ContactSensorCfg | None = None
    pegboard: AssetBaseCfg | None = None


@configclass
class InsertAnythingEnvCfg(DirectRLEnvCfg):
    decimation: int = 8
    action_space: int = 6
    observation_space: int = 21
    state_space: int = 72

    state_order: list = [
        "fingertip_pos",
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "fingertip_quat_rel_fixed",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    task_name: str = "insertanything_circle_I"
    task: InsertAnythingTask = InsertAnythingTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    evaluation_mode: bool = False
    episode_length_s: float = 10.0

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        render_interval=decimation,
        physics=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InsertAnythingSceneCfg = InsertAnythingSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=True)

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/.*",
        filter_prim_paths_expr=["/World/envs/env_.*/FixedAsset/.*"],
        update_period=0.0,
    )

    def __post_init__(self):
        robot_cfg = self.task.robot_cfg
        if robot_cfg.articulation is not None:
            self.scene.robot = replace(robot_cfg.articulation, prim_path="{ENV_REGEX_NS}/Robot")
        else:
            self.scene.robot = replace(
                FRANKA_CFG, prim_path="{ENV_REGEX_NS}/Robot", spawn=replace(FRANKA_CFG.spawn, usd_path=robot_cfg.robot_usd)
            )
        if robot_cfg.reset_joints is not None:
            self.ctrl.reset_joints = list(robot_cfg.reset_joints)
        if robot_cfg.default_dof_pos is not None:
            self.ctrl.default_dof_pos_tensor = list(robot_cfg.default_dof_pos)
        if getattr(robot_cfg, "inertia_shaping", None) is not None:
            self.ctrl.inertia_shaping = robot_cfg.inertia_shaping
            self.ctrl.desired_task_inertia = list(robot_cfg.desired_task_inertia)
        if getattr(robot_cfg, "payload_gravity_comp", None) is not None:
            self.ctrl.payload_gravity_comp = robot_cfg.payload_gravity_comp

        self.scene.fixed_asset = self.task.fixed_asset
        self.scene.held_asset = self.task.held_asset

        table = getattr(self.task, "table", "seattle")
        if table == "none":
            self.scene.table = None
        elif table == "iiwa_bench":
            # The iiwa bench: work surface 6.2 mm above the base frame (iiwa14_bench.yaml ground_plane),
            # clear of the robot's base, and the pegboard behind the socket (drawn only).
            self.scene.table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(0.8, 1.4, 0.04),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.45)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.1, 0.0062 - 0.02)),
            )
            self.scene.pegboard = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Pegboard",
                spawn=sim_utils.CuboidCfg(
                    size=(0.02, 1.0, 0.6),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.75)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.92, 0.1, 0.3)),
            )

        contact_cfg = getattr(self.task, "contact_force", {})
        contact_enabled = bool(contact_cfg.get("enabled", False))
        if contact_enabled:
            force_source = contact_cfg.get("force_source", "peg_hole")
            if force_source == "gripper_peg":
                subpaths = robot_cfg.finger_prim_subpaths or robot_cfg.finger_bodies
                self.contact_sensor.filter_prim_paths_expr = [f"/World/envs/env_.*/Robot/{p}" for p in subpaths]
            else:
                self.contact_sensor.filter_prim_paths_expr = ["/World/envs/env_.*/FixedAsset/.*"]
            self.scene.contact_sensor = self.contact_sensor

        self.scene.clone_in_fabric = not contact_enabled


@configclass
class InsertAnythingCircleHole_I_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingCircleHole_I()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingSquareHole_II_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingSquareHole_II()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingLHole_III_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingLHole_III()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingTriangleHole_IV_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingTriangleHole_IV()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingHexagonHole_III_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingHexagonHole_III()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingHexagonHole_IV_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingHexagonHole_IV()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


# --- The cable-assembly iiwa 14 cell (insertanything_iiwa_cell_cfg.py) ---------------------------------


@configclass
class InsertAnythingHexagonHole_III_Iiwa14Bench_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingHexagonHole_III_Iiwa14Bench()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingHexagonHole_III_Iiwa14Pegboard_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingHexagonHole_III_Iiwa14Pegboard()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingLHole_III_Iiwa14Bench_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingLHole_III_Iiwa14Bench()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingLHole_III_Iiwa14Pegboard_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingLHole_III_Iiwa14Pegboard()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


# --- USB-A (insertanything_usb_cfg.py) -----------------------------------------------------------------


def _usb_env_cfg(task_cls):
    @configclass
    class _Cfg(InsertAnythingEnvCfg):
        task: InsertAnythingTask = task_cls()
        task_name: str = task.name
        episode_length_s: float = task.duration_s

    _Cfg.__name__ = task_cls.__name__ + "_Cfg"
    return _Cfg


from .insertanything_usb_cfg import USB_TASKS  # noqa: E402

USB_ENV_CFGS = {suffix: _usb_env_cfg(cls) for suffix, cls in USB_TASKS.items()}
