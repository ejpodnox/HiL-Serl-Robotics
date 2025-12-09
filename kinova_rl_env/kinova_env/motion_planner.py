"""
Minimal IK/FK helpers for Kinova control.

Two backends are provided:
    - MotionPlanner   : KDL-based (requires system PyKDL; mostly kept for legacy)
    - PinMotionPlanner: Pinocchio-based (pure Python wheel; preferred in conda)
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.spatial.transform import Rotation

try:
    from kdl_parser_py.urdf import treeFromUrdfModel
    from urdf_parser_py.urdf import URDF
    import PyKDL as kdl

    KDL_AVAILABLE = True
except ImportError:
    KDL_AVAILABLE = False

# Optional Pinocchio backend (pure Python wheel works in conda)
try:
    import pinocchio as pin

    PIN_AVAILABLE = True
    PIN_LOCAL_WORLD = getattr(
        getattr(pin, "ReferenceFrame", pin), "LOCAL_WORLD_ALIGNED", None
    )
except ImportError:
    PIN_AVAILABLE = False
    PIN_LOCAL_WORLD = None


class MotionPlanner:
    def __init__(
        self,
        urdf_path: Optional[str],
        base_link: str = "base_link",
        end_effector_link: str = "tool_frame",
        joint_names: Optional[List[str]] = None,
        max_ik_iterations: int = 20,
        position_tolerance: float = 0.001,
    ):
        self.base_link = base_link
        self.end_effector_link = end_effector_link
        self.max_ik_iterations = max_ik_iterations
        self.position_tolerance = position_tolerance

        self.joint_names = joint_names or [
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'joint_7'
        ]
        self.num_joints = len(self.joint_names)

        self.chain: Optional['kdl.Chain'] = None
        self.fk_solver: Optional['kdl.ChainFkSolverPos_recursive'] = None
        self.ik_vel_solver: Optional['kdl.ChainIkSolverVel_pinv'] = None
        self.ik_solver: Optional['kdl.ChainIkSolverPos_NR'] = None

        if urdf_path and KDL_AVAILABLE:
            self._initialize_kdl(urdf_path)
        elif not KDL_AVAILABLE:
            print("[MotionPlanner] KDL not available. Install: pip install kdl_parser_py urdf_parser_py PyKDL")
        else:
            print("[MotionPlanner] URDF path not provided. IK disabled.")

    def _initialize_kdl(self, urdf_path: str) -> None:
        try:
            robot = URDF.from_xml_file(urdf_path)
            ok, tree = treeFromUrdfModel(robot)
            if not ok:
                print("[MotionPlanner] Failed to build KDL tree from URDF")
                return

            self.chain = tree.getChain(self.base_link, self.end_effector_link)
            self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
            self.ik_vel_solver = kdl.ChainIkSolverVel_pinv(self.chain)
            self.ik_solver = kdl.ChainIkSolverPos_NR(
                self.chain,
                self.fk_solver,
                self.ik_vel_solver,
                self.max_ik_iterations,
                self.position_tolerance
            )
            print("[MotionPlanner] KDL initialized for IK/FK.")
        except Exception as e:
            print(f"[MotionPlanner] KDL init error: {e}")
            self.chain = None

    def forward_kinematics(self, joint_angles: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.fk_solver is None:
            return None
        try:
            jnt_array = kdl.JntArray(self.num_joints)
            for i in range(self.num_joints):
                jnt_array[i] = joint_angles[i]
            end_frame = kdl.Frame()
            ret = self.fk_solver.JntToCart(jnt_array, end_frame)
            if ret < 0:
                return None
            pos = np.array([end_frame.p.x(), end_frame.p.y(), end_frame.p.z()])
            rot = np.array([
                [end_frame.M[0, 0], end_frame.M[0, 1], end_frame.M[0, 2]],
                [end_frame.M[1, 0], end_frame.M[1, 1], end_frame.M[1, 2]],
                [end_frame.M[2, 0], end_frame.M[2, 1], end_frame.M[2, 2]]
            ])
            return pos, rot
        except Exception:
            return None

    def solve_ik(self, target_pos: np.ndarray, target_rot: np.ndarray, seed_joints: np.ndarray) -> Optional[np.ndarray]:
        if self.ik_solver is None:
            return None
        try:
            kdl_rot = kdl.Rotation(
                target_rot[0, 0], target_rot[0, 1], target_rot[0, 2],
                target_rot[1, 0], target_rot[1, 1], target_rot[1, 2],
                target_rot[2, 0], target_rot[2, 1], target_rot[2, 2]
            )
            kdl_pos = kdl.Vector(target_pos[0], target_pos[1], target_pos[2])
            target_frame = kdl.Frame(kdl_rot, kdl_pos)

            seed = kdl.JntArray(self.num_joints)
            for i in range(self.num_joints):
                seed[i] = seed_joints[i]

            result = kdl.JntArray(self.num_joints)
            ret = self.ik_solver.CartToJnt(seed, target_frame, result)
            if ret < 0:
                return None
            return np.array([result[i] for i in range(self.num_joints)])
        except Exception:
            return None

    @staticmethod
    def apply_twist(pos: np.ndarray, rot: np.ndarray, twist: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate a small twist to get a target pose."""
        target_pos = pos + twist[:3] * dt
        delta_rot = Rotation.from_rotvec(twist[3:] * dt).as_matrix()
        target_rot = rot @ delta_rot
        return target_pos, target_rot


class PinMotionPlanner:
    """
    Pinocchio-based IK/FK helper.

    Notes:
        - Uses kinematic model only (no collision geometry required)
        - Supports optional DOF reduction via `active_joints`
    """

    def __init__(
        self,
        urdf_path: Optional[str],
        base_link: str = "base_link",
        end_effector_link: str = "tool_frame",
        damping: float = 1e-3,
        max_iterations: int = 20,
        position_tolerance: float = 1e-3,
        ee_candidates: Optional[List[str]] = None,
        active_joints: Optional[List[str]] = None,
    ):
        self.base_link = base_link
        self.end_effector_link = end_effector_link
        self.damping = damping
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.ee_candidates = ee_candidates or [end_effector_link]
        self.active_joints = active_joints

        self.model = None
        self.data = None
        self.ee_frame_id = None

        if not urdf_path:
            print("[PinMotionPlanner] URDF path not provided. IK disabled.")
            return
        if not PIN_AVAILABLE:
            print("[PinMotionPlanner] Pinocchio not available. Install: pip install pin")
            return

        try:
            # Kinematic model only; no collision/visuals needed for IK
            model = pin.buildModelFromUrdf(urdf_path)

            resolved_active = self._resolve_active_joints(model)
            model = self._lock_inactive_joints(model, resolved_active)
            self.active_joints = resolved_active
            self.model = model
            self.data = model.createData()

            self._resolve_ee_frame()
            print("[PinMotionPlanner] Pinocchio initialized for IK/FK.")
        except Exception as exc:
            print(f"[PinMotionPlanner] Init error: {exc}")
            self.model = None

    def _check_ready(self) -> bool:
        return self.model is not None and self.data is not None and self.ee_frame_id is not None

    def _resolve_active_joints(self, model) -> List[str]:
        model_joint_names = model.names[1:]  # skip universe
        resolved: List[str] = []

        if self.active_joints:
            for desired in self.active_joints:
                if desired in model_joint_names:
                    resolved.append(desired)
                    continue
                # try suffix or substring match (handles prefixes like gen3_)
                matches = [jn for jn in model_joint_names if jn.endswith(desired) or desired in jn]
                if matches:
                    resolved.append(matches[0])
            if resolved:
                return resolved
            print(f"[PinMotionPlanner] Active joints not found in model, falling back to auto selection: {self.active_joints}")

        # Auto-pick non-gripper joints (filter out robotiq/finger/gripper/camera)
        resolved = [
            jn for jn in model_joint_names
            if not any(skip in jn for skip in ("robotiq", "finger", "gripper", "camera", "base"))
        ]
        return resolved[:7]  # keep first 7 arm joints

    def _lock_inactive_joints(self, model, active: List[str]):
        model_joint_names = model.names[1:]
        lock_ids = []
        for jn in model_joint_names:
            if jn not in active:
                try:
                    lock_ids.append(model.getJointId(jn))
                except Exception:
                    continue
        if not lock_ids:
            return model
        qref = pin.neutral(model)
        return pin.buildReducedModel(model, lock_ids, qref)

    def _resolve_ee_frame(self) -> None:
        candidates = []
        for name in self.ee_candidates:
            if name not in candidates:
                candidates.append(name)
        for ee_name in candidates:
            frame_id = self.model.getFrameId(ee_name)
            if frame_id < len(self.model.frames):
                self.ee_frame_id = frame_id
                self.end_effector_link = ee_name
                return
        raise ValueError(f"End-effector frame not found. Tried: {candidates}")

    def forward_kinematics(self, joint_angles: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._check_ready():
            return None
        try:
            q = np.asarray(joint_angles, dtype=float)
            if q.shape[0] != self.model.nq:
                raise ValueError(f"Expected {self.model.nq} joints, got {q.shape[0]}")
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.ee_frame_id]
            return oMf.translation.copy(), oMf.rotation.copy()
        except Exception:
            return None

    def solve_ik(self, target_pos: np.ndarray, target_rot: np.ndarray, seed_joints: np.ndarray) -> Optional[np.ndarray]:
        if not self._check_ready():
            return None
        try:
            q = np.asarray(seed_joints, dtype=float).copy()
            if q.shape[0] != self.model.nq:
                raise ValueError(f"Expected {self.model.nq} joints, got {q.shape[0]}")

            target_pos = np.asarray(target_pos, dtype=float)
            target_rot = np.asarray(target_rot, dtype=float)

            for _ in range(self.max_iterations):
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                oMf = self.data.oMf[self.ee_frame_id]

                err_pos = target_pos - oMf.translation
                err_rot = 0.5 * pin.log3(target_rot @ oMf.rotation.T)
                err = np.concatenate([err_pos, err_rot])

                if np.linalg.norm(err_pos) < self.position_tolerance and np.linalg.norm(err_rot) < 1e-3:
                    break

                ref_frame = PIN_LOCAL_WORLD or getattr(getattr(pin, "ReferenceFrame", pin), "LOCAL", 0)
                J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, ref_frame)

                JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
                dq = J.T @ np.linalg.solve(JJt, err)
                q = q + dq

            return q
        except Exception:
            return None

    @staticmethod
    def apply_twist(pos: np.ndarray, rot: np.ndarray, twist: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        return MotionPlanner.apply_twist(pos, rot, twist, dt)
