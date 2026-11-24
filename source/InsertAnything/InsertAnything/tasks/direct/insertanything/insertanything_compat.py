"""Simulator I/O for Isaac Lab 3.0.

Everything that reads or writes a simulator buffer goes through here. Isaac Lab 3.0 returns warp-backed
``ProxyArray`` buffers (``.torch`` gives the tensor), stores quaternions as (x, y, z, w), and renamed the
write/target methods to ``*_index``. The environment keeps (w, x, y, z), so conversion happens here and
nowhere else.
"""

from __future__ import annotations

import torch
import warp as wp

from .insertanything_math import wxyz_to_xyzw, xyzw_to_wxyz


def t(x) -> torch.Tensor:
    """Torch view of a simulator buffer."""
    return x.torch if hasattr(x, "torch") else x


# --- reads (quaternions returned as w, x, y, z) -----------------------------------------------------


def root_pose(asset) -> tuple[torch.Tensor, torch.Tensor]:
    return t(asset.data.root_pos_w), xyzw_to_wxyz(t(asset.data.root_quat_w))


def body_pose(asset, body_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return t(asset.data.body_pos_w)[:, body_idx], xyzw_to_wxyz(t(asset.data.body_quat_w)[:, body_idx])


def body_vel(asset, body_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return t(asset.data.body_lin_vel_w)[:, body_idx], t(asset.data.body_ang_vel_w)[:, body_idx]


def default_root_pose(asset) -> torch.Tensor:
    """(N, 7) default root pose, position then (w, x, y, z)."""
    pose = t(asset.data.default_root_pose).clone()
    return torch.cat((pose[:, 0:3], xyzw_to_wxyz(pose[:, 3:7])), dim=-1)


def jacobians(robot) -> torch.Tensor:
    """World-frame body Jacobians, (N, num_bodies - fixed_base, 6, num_dofs). Angular part needs no
    quaternion conversion."""
    return t(robot.data.body_link_jacobian_w)


def mass_matrix(robot) -> torch.Tensor:
    return t(robot.data.mass_matrix)


def joint_pos(robot) -> torch.Tensor:
    return t(robot.data.joint_pos)


def joint_vel(robot) -> torch.Tensor:
    return t(robot.data.joint_vel)


def default_joint_pos(robot) -> torch.Tensor:
    return t(robot.data.default_joint_pos)


# --- writes -----------------------------------------------------------------------------------------


def write_root_pose(asset, pose_wxyz: torch.Tensor, env_ids=None):
    pose = torch.cat((pose_wxyz[:, 0:3], wxyz_to_xyzw(pose_wxyz[:, 3:7])), dim=-1).contiguous()
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)


def write_root_velocity(asset, vel: torch.Tensor, env_ids=None):
    asset.write_root_velocity_to_sim_index(root_velocity=vel.contiguous(), env_ids=env_ids)


def write_joint_state(robot, pos: torch.Tensor, vel: torch.Tensor, env_ids=None):
    robot.write_joint_position_to_sim_index(position=pos.contiguous(), env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=vel.contiguous(), env_ids=env_ids)


def set_joint_position_target(robot, target: torch.Tensor, env_ids=None):
    robot.set_joint_position_target_index(target=target.contiguous(), env_ids=env_ids)


def set_joint_effort_target(robot, target: torch.Tensor, env_ids=None):
    robot.set_joint_effort_target_index(target=target.contiguous(), env_ids=env_ids)


def set_gravity(sim, gravity):
    sim.physics_manager.set_gravity(tuple(float(g) for g in gravity))


def set_friction(asset, value, env_ids):
    """Static and dynamic friction of every shape of ``asset`` in ``env_ids`` (an int means all of the
    first ``env_ids`` environments). ``value`` is a float or a per-env tensor."""
    materials = wp.to_torch(asset.root_view.get_material_properties()).clone()
    if isinstance(env_ids, int):
        env_ids_cpu = torch.arange(env_ids, device="cpu")
    elif isinstance(env_ids, torch.Tensor):
        env_ids_cpu = env_ids.to("cpu")
    else:
        env_ids_cpu = torch.tensor(env_ids, device="cpu")
    env_ids_cpu = env_ids_cpu.long()
    if isinstance(value, torch.Tensor):
        val = value.to(materials.device)
        if val.dim() == 1:
            val = val.unsqueeze(-1)
        materials[env_ids_cpu, :, 0] = val
        materials[env_ids_cpu, :, 1] = val
    else:
        materials[env_ids_cpu, :, 0] = value
        materials[env_ids_cpu, :, 1] = value
    asset.root_view.set_material_properties(
        wp.from_torch(materials.contiguous()), wp.from_torch(env_ids_cpu.to(torch.int32))
    )


def add_body_inertia(robot, offset: float = 0.01):
    """Factory's armature stand-in: add ``offset`` to every body's diagonal inertia."""
    inertias = t(robot.data.body_inertia).clone()
    inertias[:, :, [0, 4, 8]] += offset
    robot.set_inertias_index(inertias=inertias)


def net_contact_forces(sensor) -> torch.Tensor:
    """Net NORMAL contact force per sensor body, world frame. Isaac Lab 2.2's ``net_forces_w`` (what the
    released policies were trained on) was normal-only; Lab 3.0's ``net_forces_w`` adds friction, which
    for a gripped peg cancels the socket's reaction against the jaws' grip friction and leaves ~0 N."""
    return t(sensor.data.net_normal_forces_w)
