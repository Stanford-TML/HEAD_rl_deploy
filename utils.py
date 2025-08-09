from typing import Tuple
import math
import numpy as np
import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul


@torch.jit.script
def rotatepoint(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q_v = [v[0], v[1], v[2], 0]
    # return quatmultiply(quatmultiply(q, q_v), quatconj(q))[:-1]
    #
    # https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    q_r = q[...,3:4]
    q_xyz = q[...,:3]
    t = 2*torch.linalg.cross(q_xyz, v)
    return v + q_r * t + torch.linalg.cross(q_xyz, t)

@torch.jit.script
def heading_zup(q: torch.Tensor) -> torch.Tensor:
    ref_dir = torch.zeros_like(q[...,:3])
    ref_dir[..., 0] = 1
    ref_dir = rotatepoint(q, ref_dir)
    return torch.atan2(ref_dir[...,1], ref_dir[...,0])

@torch.jit.script
def heading_yup(q: torch.Tensor) -> torch.Tensor:
    ref_dir = torch.zeros_like(q[...,:3])
    ref_dir[..., 0] = 1
    ref_dir = rotatepoint(q, ref_dir)
    return torch.atan2(-ref_dir[...,2], ref_dir[...,0])

@torch.jit.script
def quatnormalize(q: torch.Tensor) -> torch.Tensor:
    q = (1-2*(q[...,3:4]<0).to(q.dtype))*q
    return q / q.norm(p=2, dim=-1, keepdim=True)

@torch.jit.script
def quatmultiply(q0: torch.Tensor, q1: torch.Tensor):
    x0, y0, z0, w0 = torch.unbind(q0, -1)
    x1, y1, z1, w1 = torch.unbind(q1, -1)
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    return quatnormalize(torch.stack((x, y, z, w), -1))

@torch.jit.script
def quatconj(q: torch.Tensor):
    return torch.cat((-q[...,:3], q[...,-1:]), dim=-1)

@torch.jit.script
def axang2quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    # axis: n x 3
    # angle: n
    theta = (angle / 2).unsqueeze(-1)
    axis = axis / (axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9))
    xyz = axis * torch.sin(theta)
    w = torch.cos(theta)
    return quatnormalize(torch.cat((xyz, w), -1))
 
@torch.jit.script
def quatdiff_normalized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # quaternion representation of the rotation from unit vector a to b
    # need to check if a == -b
    # if a == -b: q = *a, 0         # 180 degree around any axis
    w = (a*b).sum(-1).add_(1)
    xyz = torch.linalg.cross(a, b)
    q = torch.cat((xyz, w.unsqueeze_(-1)), -1)
    return quatnormalize(q)

@torch.jit.script
def wrap2pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def quat2axang(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = q[..., 3]

    sin = torch.sqrt(1 - w * w)
    mask = sin > 1e-5

    angle = 2 * torch.acos(w)
    angle = wrap2pi(angle)
    axis = q[..., 0:3] / sin.unsqueeze_(-1)

    z_axis = torch.zeros_like(axis)
    z_axis[..., -1] = 1

    angle = torch.where(mask, angle, z_axis[...,0])
    axis = torch.where(mask.unsqueeze_(-1), axis, z_axis)
    return axis, angle

@torch.jit.script
def quat2expmap(q: torch.Tensor) -> torch.Tensor:
    ax, ang = quat2axang(q)
    return ang.unsqueeze(-1)*ax

@torch.jit.script
def slerp(q0, q1, frac):
    c = q0[..., 3]*q1[..., 3] + q0[..., 0]*q1[..., 0] + \
        q0[..., 1]*q1[..., 1] + q0[..., 2]*q1[..., 2]
    q1 = torch.where(c.unsqueeze_(-1) < 0, -q1, q1)

    c = c.abs_()
    s = torch.sqrt(1.0 - c*c)
    t = torch.acos(c)

    c1 = torch.sin((1-frac)*t) / s
    c2 = torch.sin(frac*t) / s
    
    x = c1*q0[..., 0:1] + c2*q1[..., 0:1]
    y = c1*q0[..., 1:2] + c2*q1[..., 1:2]
    z = c1*q0[..., 2:3] + c2*q1[..., 2:3]
    w = c1*q0[..., 3:4] + c2*q1[..., 3:4]

    q = torch.cat((x, y, z, w), dim=-1)
    q = torch.where(s < 0.001, 0.5*q0+0.5*q1, q)
    q = torch.where(c >= 1, q0, q)
    return q


@torch.jit.script
def expmap2quat(rotvec: torch.Tensor) -> torch.Tensor:
    angle = rotvec.norm(p=2, dim=-1, keepdim=True)
    axis = rotvec / angle
    ref_angle = torch.zeros_like(angle)
    ref_axis = torch.zeros_like(rotvec)
    ref_axis[..., 0] = 1
    axis = torch.where(angle < 1e-5, ref_axis, axis)
    angle = torch.where(angle < 1e-5, ref_angle, angle)

    theta = angle / 2
    xyz = axis * torch.sin(theta)
    w = torch.cos(theta)
    return quatnormalize(torch.cat((xyz, w), -1))

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def mirror_quaternion(q: torch.Tensor) -> torch.Tensor:
    """
    Mirrors the input quaternion(s) with respect to the y-axis.
    
    Assumes that the last dimension of q is 4, in the order (x, y, z, w).
    The mirror transformation corresponds to:
        (x, y, z, w) -> (-x, y, -z, w)
    
    Args:
        q (torch.Tensor): A tensor of shape (..., 4) containing quaternion(s).

    Returns:
        torch.Tensor: A tensor of shape (..., 4) with the mirrored quaternion(s).
    """
    if q.size(-1) != 4:
        raise ValueError("The last dimension of the input tensor must be 4 (i.e. (x, y, z, w)).")
    
    # Create a tensor for the sign flips: flip x and z, leave y and w unchanged.
    q = q.clone()
    q[...,0] *= -1
    q[...,2] *= -1
    return q

def get_gravity_orientation(quaternion):
    qw = quaternion[3]
    qx = quaternion[0]
    qy = quaternion[1]
    qz = quaternion[2]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def dehead_quaternion_from_gravity(gravity):
    """
    Compute a 'deheaded' (tilt-only) quaternion from a gravity vector.
    
    Given that a full orientation can be seen as a composition of a heading (yaw) and 
    a tilt (roll & pitch) (q_full = q_yaw * q_tilt), the gravity vector (which is unaffected 
    by yaw) determines the tilt portion. This function computes the quaternion q_tilt that 
    rotates the reference down vector [0, 0, -1] to align with the measured gravity vector.
    
    Parameters:
        gravity (array-like): 3-element gravity vector (e.g. from an accelerometer).
                              It does not have to be normalized.
    
    Returns:
        np.ndarray: A quaternion in [w, x, y, z] format representing the tilt (deheaded orientation).
                    When applied to a full orientation, this effectively removes the yaw component.
    """
    # Convert to numpy array and normalize
    g = np.array(gravity, dtype=np.float64)
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-8:
        raise ValueError("Gravity vector is zero or near-zero.")
    g = g / g_norm

    # The reference 'down' vector (assumes world down is [0, 0, -1])
    down = np.array([0, 0, -1], dtype=np.float64)
    
    # Compute the angle between the measured gravity and the ideal down vector.
    # dot = cos(angle)
    dot = np.dot(down, g)
    # Clamp dot to avoid numerical issues
    dot = np.clip(dot, -1.0, 1.0)
    angle = math.acos(dot)
    
    # Compute rotation axis using the cross product
    axis = np.cross(g, down)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        # The vectors are parallel or anti-parallel.
        # When g == down, no tilt is needed.
        # When g == -down, the rotation is 180 degrees about any axis perpendicular to down.
        # Here we choose an arbitrary axis.
        if dot > 0:  # g is the same as down (i.e. [0,0,-1])
            return np.array([0.0, 0.0, 0.0, 1.0])
        else:
            # 180 degree rotation. Choose [1, 0, 0] as the rotation axis.
            return np.array([1.0, 0.0, 0.0, 0.0])
    
    axis = axis / axis_norm
    
    # Convert the angle to a quaternion (using half-angle formula)
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    
    # Return the quaternion in [x, y, z, w] format
    return np.concatenate((xyz, [w]))