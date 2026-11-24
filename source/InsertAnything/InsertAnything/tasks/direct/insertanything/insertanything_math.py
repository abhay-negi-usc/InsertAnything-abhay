"""Scalar-first (w, x, y, z) quaternion helpers.

InsertAnything was written against Isaac Lab 2.2 / Isaac Sim 5.0, where ``isaacsim.core.utils.torch``
and ``isaaclab.utils.math`` both used (w, x, y, z). Isaac Lab 3.0 moved ``isaaclab.utils.math`` and every
simulator buffer to (x, y, z, w) and deprecated ``isaacsim.core.utils.torch``.

The released checkpoints observe (w, x, y, z) quaternions, so the environment keeps that convention
internally and converts only where data crosses to or from the simulator (``insertanything_compat``).
These functions are copied from ``isaacsim.core.utils.torch`` (Isaac Sim 6.2, deprecated module) and
``isaaclab.utils.math.axis_angle_from_quat`` (Isaac Lab 2.2), unchanged in behaviour.
"""

from __future__ import annotations

import numpy as np
import torch


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def copysign(a: float, b: torch.Tensor) -> torch.Tensor:
    a_t = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a_t) * torch.sign(b)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


def quat_apply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def get_euler_xyz(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extrinsic XYZ Euler angles, each wrapped to [0, 2*pi) as in Isaac Sim."""
    qw, qx, qy, qz = 0, 1, 2, 3
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qw, qx, qy, qz], dim=-1)


def tf_inverse(q: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


def tf_combine(
    q1: torch.Tensor, t1: torch.Tensor, q2: torch.Tensor, t2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Rotation vector of a (w, x, y, z) quaternion (Isaac Lab 2.2 ``axis_angle_from_quat``)."""
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:4], q[..., 0:1]), dim=-1)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 3:4], q[..., 0:3]), dim=-1)
