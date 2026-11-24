"""USB-A plug into a USB-A receptacle (the cable-assembly bench's panel-mount adapter, McMaster 4574N11, and
cord, 4974T135).

Assets: scripts/assets/build_usb_assets.py. Frames: the socket origin is where the plug origin sits when
seated, so the held base pose and its target coincide at a full insertion (base_height 0). In the socket
frame the pocket floor is at z = -4.84 mm and the shell's lip at z = +5.54 mm; the plug tip is 4.84 mm
below its origin.

Two receptacles, for the question of what to do with the shell's seven spring prongs (detent tabs, which
reach 0.15-0.6 mm into the plug's path in the CAD):
    NoProngs    the prongs cut away (cable-assembly's collider): the pocket at the USB-A shell's inside,
                0.30-0.33 mm of shell clearance per side.
    Compliant   the same pocket plus the prongs as compliant-contact colliders (k per contact).
Both keep cable-assembly's 0.15 mm/face relief of the tongue (its contacts' give).
"""

from isaaclab.utils import configclass

from .insertanything_iiwa_cell_cfg import (
    BENCH_JOINTS_DEG,
    PEGBOARD_JOINTS_DEG,
    RY_MINUS_90_WXYZ,
    RZ_180_WXYZ,
    USB_SOCKET_MOUTH,
    iiwa_robot_cfg,
)
from .insertanything_tasks_cfg import (
    INSERTANYTHING_ASSETS_DIR,
    FixedAssetCfg,
    HeldAssetCfg,
    InsertAnythingHexagonHole_III,
    RobotCfg,
)

USB_DIR = f"{INSERTANYTHING_ASSETS_DIR}/usb"
USB_LIP_Z = 0.00554
"""The receptacle shell's lip above the socket origin [m]: the 'hole height' (the task's mouth)."""
USB_GRASP_WIDTH = 0.0080
"""The overmold's thickness between the jaws (plug y)."""


@configclass
class UsbSocketNoProngs(FixedAssetCfg):
    usd_path: str = f"{USB_DIR}/usb_socket_noprongs.usd"
    diameter: float = 0.0125
    height: float = USB_LIP_Z
    base_height: float = 0.0
    friction: float = 0.3
    mass: float = 0.02


@configclass
class UsbSocketCompliant(UsbSocketNoProngs):
    usd_path: str = f"{USB_DIR}/usb_socket_compliant.usd"


@configclass
class UsbPlug(HeldAssetCfg):
    usd_path: str = f"{USB_DIR}/usb_plug.usd"
    diameter: float = USB_GRASP_WIDTH
    height: float = 0.043
    friction: float = 0.5
    mass: float = 0.0083
    # Fingertip on the overmold's centre line, 25 mm up from the plug origin (30 mm above the tip).
    grasp_pos: list = [0.0, 0.000456, 0.025]


@configclass
class InsertAnythingUsb(InsertAnythingHexagonHole_III):
    """The hexagon-III task (decoupled gated reward, gripper-peg force) on the USB geometry."""

    name: str = "insertanything_usb_noprongs"
    fixed_asset_cfg: FixedAssetCfg = UsbSocketNoProngs()
    held_asset_cfg: HeldAssetCfg = UsbPlug()
    asset_size: float = USB_GRASP_WIDTH
    duration_s: float = 20.0

    # Plug tip ~10 mm above the lip at the nominal start.
    hand_init_pos: list = [0.0, 0.0, 0.040]
    hand_init_pos_range: list = [0.02, 0.02, 0.005]
    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    # The hand-down grasp puts the plug at socket yaw 180 deg (as the L and hexagon tasks' holes).
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 10.0
    held_asset_pos_range: list = [0.0, 0.0, 0.0]

    symmetry_angles_deg: list = [0.0]  # USB-A is not 180 deg symmetric
    yaw_success_threshold: float = 0.05
    # Seated to within 10% of the lip height (0.55 mm).
    success_threshold: float = 0.1


@configclass
class InsertAnythingUsbCompliant(InsertAnythingUsb):
    name: str = "insertanything_usb_compliant"
    fixed_asset_cfg: FixedAssetCfg = UsbSocketCompliant()


def _iiwa_bench(base, base_name):
    @configclass
    class _T(base):
        name: str = base_name + "_iiwa14_bench"
        robot_cfg: RobotCfg = iiwa_robot_cfg(BENCH_JOINTS_DEG, USB_GRASP_WIDTH)
        cell_rotation_wxyz: list = RZ_180_WXYZ
        fixed_asset_pos: tuple = (0.6, 0.0, 0.06)
        table: str = "iiwa_bench"

    _T.__name__ = base.__name__ + "_Iiwa14Bench"
    return _T


def _iiwa_pegboard(base, base_name):
    x, y, z = USB_SOCKET_MOUTH

    @configclass
    class _T(base):
        name: str = base_name + "_iiwa14_pegboard"
        robot_cfg: RobotCfg = iiwa_robot_cfg(PEGBOARD_JOINTS_DEG, USB_GRASP_WIDTH)
        cell_rotation_wxyz: list = RY_MINUS_90_WXYZ
        fixed_asset_pos: tuple = (x + USB_LIP_Z, y, z)  # the lip at the real receptacle's mouth
        table: str = "iiwa_bench"

    _T.__name__ = base.__name__ + "_Iiwa14Pegboard"
    return _T


InsertAnythingUsb_Iiwa14Bench = _iiwa_bench(InsertAnythingUsb, "insertanything_usb_noprongs")
InsertAnythingUsb_Iiwa14Pegboard = _iiwa_pegboard(InsertAnythingUsb, "insertanything_usb_noprongs")
InsertAnythingUsbCompliant_Iiwa14Bench = _iiwa_bench(InsertAnythingUsbCompliant, "insertanything_usb_compliant")
InsertAnythingUsbCompliant_Iiwa14Pegboard = _iiwa_pegboard(InsertAnythingUsbCompliant, "insertanything_usb_compliant")

# (gym id suffix, task class)
USB_TASKS = {
    "Usb-NoProngs": InsertAnythingUsb,
    "Usb-NoProngs-Iiwa14Bench": InsertAnythingUsb_Iiwa14Bench,
    "Usb-NoProngs-Iiwa14Pegboard": InsertAnythingUsb_Iiwa14Pegboard,
    "Usb-Compliant": InsertAnythingUsbCompliant,
    "Usb-Compliant-Iiwa14Bench": InsertAnythingUsbCompliant_Iiwa14Bench,
    "Usb-Compliant-Iiwa14Pegboard": InsertAnythingUsbCompliant_Iiwa14Pegboard,
}
