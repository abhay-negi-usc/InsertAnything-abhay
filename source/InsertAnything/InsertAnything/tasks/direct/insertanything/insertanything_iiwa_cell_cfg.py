"""The iiwa 14 bench of the cable-assembly project, as an InsertAnything cell.

Robot: KUKA LBR iiwa 14 R820 + Robotiq 2F-85 on the media flange (cable-assembly
``configs/cells/iiwa14_bench.yaml``), from the USD that ``cable-assembly/sim/scripts/build_iiwa14.py``
builds. The arm is torque-driven here (stiffness 0), as InsertAnything's operational-space
controller expects, not position-driven as in cable-assembly's own BNC task.

Two placements of the single-hole tasks:

* ``Iiwa14Bench``: the hole faces up at the released Franka tasks' spot (0.6 m out). Changes only
  the robot, so it isolates the arm/gripper effect.
* ``Iiwa14Pegboard``: the hole mouth at the bench's USB receptacle pose, (815, 226, 272) mm in the
  iiwa base frame (``iiwa14_bench.frames.yaml`` ``targets.usb_connector``), facing the robot (-x)
  as on the vertical pegboard. The task frame is Ry(-90 deg): the policy still sees a hole facing +z.

Assets: ``assets/iiwa_cell`` holds the robot USDs, copied from cable-assembly's
``sim/src/cable_assembly_sim/assets/data``: ``iiwa14_2f85{,_bench}.usda``, ``iiwa14/usd``, ``robotiq_2f85``,
``robotiq_2f85_v3/v3_visuals.usdc`` and ``tool/tool_meshes.usdc``. Their layout is kept, because the .usda
files reference the others by relative path. Set ``CABLE_ASSEMBLY_SIM_ASSETS`` to use another copy, e.g.
cable-assembly's own after a rebuild there.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg, PhysxRigidBodyCfg

from .insertanything_tasks_cfg import InsertAnythingHexagonHole_III, InsertAnythingLHole_III, RobotCfg

CABLE_ASSEMBLY_SIM_ASSETS = os.environ.get(
    "CABLE_ASSEMBLY_SIM_ASSETS",
    os.path.join(os.path.dirname(__file__), "assets", "iiwa_cell"),
)
IIWA14_2F85_USD = os.path.join(CABLE_ASSEMBLY_SIM_ASSETS, "iiwa14_2f85.usda")

ARM_JOINTS = [f"A{i}" for i in range(1, 8)]
# The 2F-85 linkage, commanded as a whole (cable-assembly assets/iiwa14.py GRIPPER_JOINT_SIGNS).
GRIPPER_JOINT_SIGNS = {
    "finger_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": 1.0,
    "left_inner_finger_knuckle_joint": -1.0,
    "right_inner_finger_knuckle_joint": -1.0,
}
EFFORT_LIMIT_NM = {"A1": 320.0, "A2": 320.0, "A3": 176.0, "A4": 176.0, "A5": 110.0, "A6": 40.0, "A7": 40.0}
VELOCITY_LIMIT_DEG_S = {"A1": 85.0, "A2": 85.0, "A3": 100.0, "A4": 75.0, "A5": 130.0, "A6": 135.0, "A7": 135.0}

# Pad tips past tool0 with the fingers closed on a ~10 mm peg. The 2F-85 pads (Isaac's V4 model) span
# 110.4..148.4 mm past tool0 open; the four-bar linkage closed to ~41 deg moves them 13.8 mm further
# out (outer knuckle 57.2 mm, pivots at y 30.6 / 67.8 mm). cable-assembly's BNC task measured 162.1 mm
# at 39.5 deg. This is the counterpart of the Franka's panda_fingertip_centered.
PAD_TIP_Z = 0.1622


def robotiq_2f85_angle_for_width(width: float) -> float:
    """finger_joint angle [rad] at which the 2F-85 pads' inner faces are ``width`` apart.

    Parallel four-bar: the pad translates with the outer knuckle's tip, which swings about the
    finger_joint pivot on a 57.2 mm arm, (dy, dz) = (-37.16, 43.43) mm at 0 rad. The pads' inner faces
    are 43.5 mm off the axis when open (the USD's fingertip collider).
    """
    dy, dz, face0 = -0.03716, 0.04343, 0.0435

    def face(theta):
        return face0 - ((dy * math.cos(theta) + dz * math.sin(theta)) - dy)

    lo, hi = 0.0, 0.8
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if face(mid) > width / 2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _deg(q):
    return [math.radians(a) for a in q]


IIWA14_2F85_OSC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=IIWA14_2F85_USD,
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
            enabled_self_collisions=False, solver_position_iteration_count=192, solver_velocity_iteration_count=1
        ),
        collision_props=PhysxCollisionCfg(contact_offset=0.0005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={**dict(zip(ARM_JOINTS, _deg([0.0, 30.0, 0.0, -90.0, 0.0, 60.0, 0.0]))), ".*finger.*|.*knuckle.*": 0.0},
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        # Torque-driven: InsertAnything's OSC writes joint efforts (as for the Franka arm).
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            joint_effort_limit=EFFORT_LIMIT_NM,
            joint_velocity_limit={j: math.radians(v) for j, v in VELOCITY_LIMIT_DEG_S.items()},
        ),
        # cable-assembly's BNC task drive: firm enough that a part does not slide up the pads on contact.
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            joint_effort_limit=10.0,
            joint_velocity_limit=1.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            joint_effort_limit=10.0,
            joint_velocity_limit=10.0,
            stiffness=10.0,
            damping=0.05,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
            joint_effort_limit=1.0,
            joint_velocity_limit=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)


def iiwa_robot_cfg(joints_deg: list, held_width: float) -> RobotCfg:
    """The iiwa + 2F-85 with ``joints_deg`` as the reset, null-space and IK-retry posture, opening to
    1.25 x ``held_width`` before the grasp (the Franka tasks' rule)."""
    q = _deg(joints_deg)
    return RobotCfg(
        robot_usd=IIWA14_2F85_USD,
        articulation=IIWA14_2F85_OSC_CFG,
        # Same peg protrusion below the pad tips as on the Franka (17.6 mm of the peg in the pads).
        franka_fingerpad_length=0.017608,
        friction=0.75,
        arm_joint_names=ARM_JOINTS,
        gripper_joint_names=list(GRIPPER_JOINT_SIGNS),
        gripper_type="robotiq_2f85",
        gripper_joint_signs=list(GRIPPER_JOINT_SIGNS.values()),
        gripper_close_rad=0.8,  # past contact on a 10-15 mm peg (~0.68-0.72 rad), so the drive squeezes
        gripper_pregrasp_rad=robotiq_2f85_angle_for_width(1.25 * held_width),
        canonicalize_fingertip_quat=True,
        fingertip_body="tool0",
        fingertip_offset=[0.0, 0.0, PAD_TIP_Z],
        jacobian_mode="point",
        finger_bodies=["left_inner_finger", "right_inner_finger"],
        finger_prim_subpaths=[
            "gripper/Robotiq_2F_85/left_inner_finger",
            "gripper/Robotiq_2F_85/right_inner_finger",
        ],
        reset_joints=q,
        default_dof_pos=q,
        ik_fallback_joints=q,
    )


# IK seeds, from cable-assembly's analytic solver (robot/kinematics/iiwa14.py ik_steered, HOME seed) for
# the fingertip 47 mm out of the hole mouth on the hole axis, the canonical yaw 0.
# Pegboard: q = [25.2, 52.0, 0.0, -98.1, -151.5, 63.2, -13.8] deg, singularity margins 52/57 deg,
# +-30 deg twist and every corner of the start box reachable.
PEGBOARD_JOINTS_DEG = [25.22, 51.98, 0.0, -98.11, -151.48, 63.18, -13.78]
# Bench (hole up, task frame turned 180 deg about z so A7 sits near +5 deg instead of -175).
BENCH_JOINTS_DEG = [-5.8, 43.19, 8.74, -85.35, -7.58, 52.04, 5.28]

# Peg width between the jaws (the jaws close along the fingertip y): hexagon III across flats
# (hexagon_peg_III.usd, y extent 10.2 mm), the L III peg's square envelope (14.8 mm).
HEX_GRASP_WIDTH = 0.0102
L_GRASP_WIDTH = 0.0148

# The USB receptacle's mouth on the pegboard, iiwa base frame [m] (iiwa14_bench.frames.yaml).
USB_SOCKET_MOUTH = (0.815, 0.226, 0.272)
# Task frame -> base: Ry(-90 deg). Task +z (out of the hole) = base -x (toward the robot).
RY_MINUS_90_WXYZ = [math.cos(math.radians(-45.0)), 0.0, math.sin(math.radians(-45.0)), 0.0]
RZ_180_WXYZ = [0.0, 0.0, 0.0, 1.0]


def _pegboard_origin(hole_height: float) -> tuple:
    """Board origin such that its hole mouth (task z = height) sits at the socket's mouth."""
    x, y, z = USB_SOCKET_MOUTH
    return (x + hole_height, y, z)


@configclass
class InsertAnythingHexagonHole_III_Iiwa14Bench(InsertAnythingHexagonHole_III):
    name: str = "insertanything_hexagon_III_iiwa14_bench"
    robot_cfg: RobotCfg = iiwa_robot_cfg(BENCH_JOINTS_DEG, HEX_GRASP_WIDTH)
    cell_rotation_wxyz: list = RZ_180_WXYZ
    fixed_asset_pos: tuple = (0.6, 0.0, 0.06)
    table: str = "iiwa_bench"


@configclass
class InsertAnythingHexagonHole_III_Iiwa14Pegboard(InsertAnythingHexagonHole_III):
    name: str = "insertanything_hexagon_III_iiwa14_pegboard"
    robot_cfg: RobotCfg = iiwa_robot_cfg(PEGBOARD_JOINTS_DEG, HEX_GRASP_WIDTH)
    cell_rotation_wxyz: list = RY_MINUS_90_WXYZ
    fixed_asset_pos: tuple = _pegboard_origin(0.025)
    table: str = "iiwa_bench"


@configclass
class InsertAnythingLHole_III_Iiwa14Bench(InsertAnythingLHole_III):
    name: str = "insertanything_L_III_iiwa14_bench"
    robot_cfg: RobotCfg = iiwa_robot_cfg(BENCH_JOINTS_DEG, L_GRASP_WIDTH)
    cell_rotation_wxyz: list = RZ_180_WXYZ
    fixed_asset_pos: tuple = (0.6, 0.0, 0.06)
    table: str = "iiwa_bench"


@configclass
class InsertAnythingLHole_III_Iiwa14Pegboard(InsertAnythingLHole_III):
    name: str = "insertanything_L_III_iiwa14_pegboard"
    robot_cfg: RobotCfg = iiwa_robot_cfg(PEGBOARD_JOINTS_DEG, L_GRASP_WIDTH)
    cell_rotation_wxyz: list = RY_MINUS_90_WXYZ
    fixed_asset_pos: tuple = _pegboard_origin(0.025)
    table: str = "iiwa_bench"
