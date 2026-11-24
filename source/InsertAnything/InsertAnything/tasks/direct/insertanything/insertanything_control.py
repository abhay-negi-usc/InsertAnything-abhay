import math

import torch

from . import insertanything_math as torch_utils
from .insertanything_math import axis_angle_from_quat


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    """Compute task-space pose error between the target and current fingertip pose."""
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    if jacobian_type != "geometric":
        raise ValueError(f"Unsupported jacobian_type: {jacobian_type}")

    # Keep the target quaternion on the same hemisphere to follow the shortest rotation.
    quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
    ctrl_target_fingertip_midpoint_quat = torch.where(
        quat_dot.expand(-1, 4) >= 0.0,
        ctrl_target_fingertip_midpoint_quat,
        -ctrl_target_fingertip_midpoint_quat,
    )

    fingertip_midpoint_quat_norm = torch_utils.quat_mul(
        fingertip_midpoint_quat,
        torch_utils.quat_conjugate(fingertip_midpoint_quat),
    )[:, 0]
    fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
        fingertip_midpoint_quat
    ) / fingertip_midpoint_quat_norm.unsqueeze(-1)
    quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)
    axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    if rot_error_type == "axis_angle":
        return pos_error, axis_angle_error
    raise ValueError(f"Unsupported rot_error_type: {rot_error_type}")


def _apply_task_space_gains(
    delta_fingertip_pose,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    task_prop_gains,
    task_deriv_gains,
):
    """Apply task-space PD gains to pose error."""
    task_wrench = torch.zeros_like(delta_fingertip_pose)

    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )
    return task_wrench


_LAMBDA_D_CACHE = {}


def _desired_task_inertia(cfg, device):
    """``cfg.ctrl.desired_task_inertia``: 6 numbers (diagonal) or 36 (row-major), task frame [kg, kg m^2]."""
    vals = tuple(cfg.ctrl.desired_task_inertia)
    key = (vals, str(device))
    if key not in _LAMBDA_D_CACHE:
        t = torch.tensor(vals, dtype=torch.float32, device=device)
        _LAMBDA_D_CACHE[key] = torch.diag(t) if t.numel() == 6 else t.reshape(6, 6)
    return _LAMBDA_D_CACHE[key]


def compute_dof_torque(
    cfg,
    dof_pos,
    dof_vel,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    jacobian,
    arm_mass_matrix,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device,
    dead_zone_thresholds=None,
    external_wrench=None,
    bias_wrench=None,
):
    """Compute arm joint torques using operational-space control (J^T F, null-space posture).

    ``external_wrench``: the environment's wrench on the fingertip, task frame (only for
    ``ctrl.inertia_shaping == "nominal_fext"``). ``bias_wrench``: added to the applied wrench as it is
    (feed-forward, e.g. payload gravity compensation), task frame."""
    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    task_wrench += _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )

    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    arm_mass_matrix_task = torch.inverse(jacobian @ arm_mass_matrix_inv @ jacobian_t)
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv

    # Inertia shaping: make this arm's fingertip respond to the task wrench as an arm with task-space
    # inertia ``desired_task_inertia`` would (the arm the policy was trained on). With Lambda the arm's own
    # task inertia and Lambda_d the desired one, Lambda * a = F_applied + F_ext, so
    #   "nominal":      F_applied = Lambda Lambda_d^-1 F                    exact in free space
    #   "nominal_fext": F_applied = Lambda Lambda_d^-1 (F + F_ext) - F_ext   exact in contact too
    #   "nominal_trans": "nominal" on the translational 3x3 block only
    # ("nominal" alone leaves contact Lambda Lambda_d^-1 times stiffer than the reference arm.)
    shaping = getattr(cfg.ctrl, "inertia_shaping", "none")
    applied_wrench = task_wrench
    if shaping != "none":
        lam_d = _desired_task_inertia(cfg, device)
        ratio = arm_mass_matrix_task @ torch.inverse(lam_d).unsqueeze(0) if shaping != "nominal_trans" else None
        if shaping == "nominal":
            applied_wrench = (ratio @ task_wrench.unsqueeze(-1)).squeeze(-1)
        elif shaping == "nominal_trans":
            # Translation only. The controller holds roll/pitch stiffly, so the fingertip's translational
            # response is governed by Lambda's 3x3 translational block (fit from trajectories:
            # scripts/diagnostics/fit_apparent_inertia.py). F' = Lambda_tt Lambda_d,tt^-1 F gives this arm
            # the reference arm's force-to-acceleration map, its x-y-z coupling included; rotation stays
            # the arm's own, and no torque leaks into force through translation-rotation coupling terms.
            ratio_t = arm_mass_matrix_task[:, 0:3, 0:3] @ torch.inverse(lam_d[0:3, 0:3]).unsqueeze(0)
            applied_wrench = task_wrench.clone()
            applied_wrench[:, 0:3] = (ratio_t @ task_wrench[:, 0:3].unsqueeze(-1)).squeeze(-1)
        elif shaping == "nominal_fext":
            f_ext = external_wrench if external_wrench is not None else torch.zeros_like(task_wrench)
            applied_wrench = (ratio @ (task_wrench + f_ext).unsqueeze(-1)).squeeze(-1) - f_ext
        else:
            raise ValueError(f"Unknown inertia_shaping: {shaping}")
    if bias_wrench is not None:
        applied_wrench = applied_wrench + bias_wrench
    dof_torque[:, 0:7] = (jacobian_t @ applied_wrench.unsqueeze(-1)).squeeze(-1)

    default_dof_pos = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))
    distance_to_default_dof_pos = default_dof_pos - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (2 * math.pi) - math.pi

    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)

    null_projector = torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv
    torque_null = null_projector @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench


def get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """Map a task-space pose delta to joint-position delta with the selected IK method."""
    if ik_method == "pinv":
        k_val = 1.0
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        return delta_dof_pos.squeeze(-1)

    if ik_method == "trans":
        k_val = 1.0
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_t @ delta_pose.unsqueeze(-1)
        return delta_dof_pos.squeeze(-1)

    if ik_method == "dls":
        lambda_val = 0.1
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_t @ torch.inverse(jacobian @ jacobian_t + lambda_matrix) @ delta_pose.unsqueeze(-1)
        return delta_dof_pos.squeeze(-1)

    if ik_method == "svd":
        k_val = 1.0
        u_mat, singular_values, vh_mat = torch.linalg.svd(jacobian)
        singular_values_inv = 1.0 / singular_values
        min_singular_value = 1.0e-5
        singular_values_inv = torch.where(
            singular_values > min_singular_value,
            singular_values_inv,
            torch.zeros_like(singular_values_inv),
        )
        jacobian_pinv = (
            torch.transpose(vh_mat, dim0=1, dim1=2)[:, :, :6]
            @ torch.diag_embed(singular_values_inv)
            @ torch.transpose(u_mat, dim0=1, dim1=2)
        )
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        return delta_dof_pos.squeeze(-1)

    raise ValueError(f"Unsupported IK method: {ik_method}")
