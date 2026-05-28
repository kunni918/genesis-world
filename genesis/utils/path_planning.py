import dataclasses
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING

import quadrants as qd
import torch
import torch.nn.functional as F

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class
from genesis.utils.misc import qd_to_torch

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


_QD_STATE_TYPES = (qd.Tensor, qd.Field)


class PathPlanner(ABC):
    def __init__(self, entity):
        self._entity = entity
        self._solver: "RigidSolver" = entity._solver

        self.PENETRATION_EPS = 1e-5 if gs.qd_float == qd.f32 else 0.0

        for joint in entity.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                gs.raise_exception("planning for the gs.JOINT_TYPE.FREE is not supported (yet)")
            elif joint.type == gs.JOINT_TYPE.SPHERICAL:
                gs.raise_exception("planning for the gs.JOINT_TYPE.SPHERICAL is not supported (yet)")

    @abstractmethod
    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        resolution=0.05,
        timeout=None,
        max_nodes=2000,
        smooth_path=True,
        num_waypoints=100,
        ignore_collision=False,
        ee_link_idx=None,
        obj_entity=None,
        envs_idx=None,
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal` for `self._entity`. Each call snapshots and
        restores live solver state when `obj_entity` is provided, so the live scene is unchanged
        whether the planner succeeds, fails, or raises.
        """
        ...

    def get_link_pose(self, robot_g_link_idx, obj_g_link_idx, envs_idx):
        """
        Get the relative pose of a given robot link wrt some object link.

        Parameters
        ----------
        robot_g_link_idx: int
            Global link idx of the link of the robot.
        obj_g_link_idx: int
            Global link idx of the base link of the object.
        """
        if self._solver.n_envs > 0:
            robot_trans = self._solver.get_links_pos(links_idx=robot_g_link_idx, envs_idx=envs_idx)
            robot_quat = self._solver.get_links_quat(links_idx=robot_g_link_idx, envs_idx=envs_idx)
            obj_trans = self._solver.get_links_pos(links_idx=obj_g_link_idx, envs_idx=envs_idx)
            obj_quat = self._solver.get_links_quat(links_idx=obj_g_link_idx, envs_idx=envs_idx)
        else:
            robot_trans = self._solver.get_links_pos(links_idx=robot_g_link_idx)
            robot_quat = self._solver.get_links_quat(links_idx=robot_g_link_idx)
            obj_trans = self._solver.get_links_pos(links_idx=obj_g_link_idx)
            obj_quat = self._solver.get_links_quat(links_idx=obj_g_link_idx)

        trans = gu.inv_transform_by_trans_quat(obj_trans, robot_trans, robot_quat)
        quat = gu.transform_quat_by_quat(obj_quat, gu.inv_quat(robot_quat))

        return trans, quat

    def update_object(self, ee_link_idx, obj_link_idx, _pos, _quat, envs_idx):
        if self._solver.n_envs > 0:
            robot_trans = self._solver.get_links_pos(ee_link_idx, envs_idx=envs_idx)
            robot_quat = self._solver.get_links_quat(ee_link_idx, envs_idx=envs_idx)
        else:
            robot_trans = self._solver.get_links_pos(ee_link_idx)
            robot_quat = self._solver.get_links_quat(ee_link_idx)

        trans, quat = gu.transform_pos_quat_by_trans_quat(_pos, _quat, robot_trans, robot_quat)

        if self._solver.n_envs > 0:
            self._solver.set_base_links_pos(trans, obj_link_idx, envs_idx=envs_idx)
            self._solver.set_base_links_quat(quat, obj_link_idx, envs_idx=envs_idx)
        else:
            self._solver.set_base_links_pos(trans, obj_link_idx)
            self._solver.set_base_links_quat(quat, obj_link_idx)

    def snapshot_entity_state(self, entity, envs_idx):
        if self._solver.n_envs > 0:
            pos = entity.get_pos(envs_idx=envs_idx).clone()
            quat = entity.get_quat(envs_idx=envs_idx).clone()
            dofs_velocity = entity.get_dofs_velocity(envs_idx=envs_idx).clone()
        else:
            pos = entity.get_pos().clone()
            quat = entity.get_quat().clone()
            dofs_velocity = entity.get_dofs_velocity().clone()

        return pos, quat, dofs_velocity

    def _assert_can_restore_base_pose(self, entity, pos, quat, envs_idx):
        # Fixed entities with non-batched vertices share one pose across envs and cannot be restored
        # for a strict subset; allow the restore only when the current pose already matches.
        if not (
            self._solver.n_envs > 0
            and entity.base_link.is_fixed
            and not entity._batch_fixed_verts
            and len(envs_idx) != self._solver.n_envs
        ):
            return True

        current_pos = entity.get_pos(envs_idx=envs_idx)
        current_quat = entity.get_quat(envs_idx=envs_idx)
        if not (
            torch.allclose(current_pos, pos, rtol=0.0, atol=gs.EPS)
            and torch.allclose(current_quat, quat, rtol=0.0, atol=gs.EPS)
        ):
            gs.raise_exception(
                "Cannot restore env-specific pose for a fixed object with non-batched fixed vertices. "
                "Set morph option `batch_fixed_verts=True` or plan across all environments."
            )
        return False

    def restore_entity_state(self, entity, state, envs_idx):
        pos, quat, dofs_velocity = state
        obj_link_idx = entity.base_link_idx
        can_restore_base_pose = self._assert_can_restore_base_pose(entity, pos, quat, envs_idx)

        if self._solver.n_envs > 0:
            if can_restore_base_pose:
                self._solver.set_base_links_pos(pos, obj_link_idx, envs_idx=envs_idx)
                self._solver.set_base_links_quat(quat, obj_link_idx, envs_idx=envs_idx)
            if entity.n_dofs > 0:
                dofs_idx = entity._get_global_idx(None, entity.n_dofs, entity._dof_start, unsafe=True)
                self._solver.set_dofs_velocity(dofs_velocity, dofs_idx, envs_idx=envs_idx)
        else:
            if can_restore_base_pose:
                self._solver.set_base_links_pos(pos, obj_link_idx)
                self._solver.set_base_links_quat(quat, obj_link_idx)
            if entity.n_dofs > 0:
                dofs_idx = entity._get_global_idx(None, entity.n_dofs, entity._dof_start, unsafe=True)
                self._solver.set_dofs_velocity(dofs_velocity, dofs_idx)

    def iter_state_tensors(self, state):
        if isinstance(state, _QD_STATE_TYPES):
            yield state
            return

        if dataclasses.is_dataclass(state):
            for field in dataclasses.fields(state):
                yield from self.iter_state_tensors(getattr(state, field.name))

    def snapshot_tensor_state(self, tensors):
        seen = set()
        state = []
        for tensor in tensors:
            tensor_id = id(tensor)
            if tensor_id in seen:
                continue
            seen.add(tensor_id)
            state.append((tensor, qd_to_torch(tensor, copy=True).detach()))
        return state

    def restore_tensor_state(self, state):
        for tensor, value in state:
            tensor.from_torch(value)

    def normalize_envs_idx(self, envs_idx):
        if self._solver.n_envs == 0:
            return envs_idx
        # `_sanitize_envs_idx` already maps bool masks to integer indices.
        return self._solver._scene._sanitize_envs_idx(envs_idx)

    def iter_constraint_state_tensors(self, constraint_solver):
        constraint_state = getattr(constraint_solver, "constraint_state", None)
        if self._solver._use_contact_island:
            # `ConstraintSolverIsland` rebuilds the heavy constraint matrices on every step from the
            # contact island state, but it keeps `qacc_ws`/`is_warmstart` and a few counters across
            # steps. Restore only those — the Jacobian/Cholesky scratch is regenerated.
            for name in (
                "qacc_ws",
                "is_warmstart",
                "n_constraints",
                "n_constraints_equality",
                "n_constraints_frictionloss",
                "qd_n_equalities",
            ):
                tensor = getattr(constraint_solver, name, None)
                if tensor is None and constraint_state is not None:
                    tensor = getattr(constraint_state, name, None)
                if tensor is not None:
                    yield tensor
            return

        # `ConstraintSolver` (non-island) only keeps `qacc_ws`/`is_warmstart` between steps; the rest
        # is rebuilt from `_eq_const_info_cache` on demand.
        yield constraint_state.is_warmstart
        yield constraint_state.qacc_ws

    def snapshot_constraint_state(self):
        constraint_solver = self._solver.constraint_solver
        if constraint_solver is None:
            return None

        cache = getattr(constraint_solver, "_eq_const_info_cache", None)
        cache_state = None if cache is None else cache.copy()
        tensor_state = self.snapshot_tensor_state(self.iter_constraint_state_tensors(constraint_solver))
        return tensor_state, cache_state

    def restore_constraint_state(self, state):
        constraint_solver = self._solver.constraint_solver
        tensor_state, cache_state = state
        self.restore_tensor_state(tensor_state)

        cache = getattr(constraint_solver, "_eq_const_info_cache", None)
        if cache is not None and cache_state is not None:
            cache.clear()
            cache.update(cache_state)

    def iter_collider_state_tensors(self):
        collider_state = self._solver.collider._collider_state
        yield collider_state.n_contacts
        yield collider_state.n_contacts_hibernated
        yield collider_state.first_time
        if getattr(self._solver, "_requires_grad", False) or getattr(
            self._solver._static_rigid_sim_config, "requires_grad", False
        ):
            yield from self.iter_state_tensors(collider_state.diff_contact_input)
        yield from self.iter_state_tensors(collider_state.contact_data)
        yield from self.iter_state_tensors(collider_state.contact_cache)

    def snapshot_collider_state(self):
        return self.snapshot_tensor_state(self.iter_collider_state_tensors())

    def restore_collider_state(self, state):
        self.restore_tensor_state(state)
        self._solver.collider._contact_data_cache.clear()

    def snapshot_hibernation_state(self):
        # Honor the user request even if `_use_hibernation` was downgraded to False because
        # `use_contact_island=False`; the underlying buffers still exist and may be mutated.
        if not (self._solver._use_hibernation or self._solver._options.use_hibernation):
            return None

        rigid_global_info = self._solver._rigid_global_info
        tensors = [
            self._solver.entities_state.hibernated,
            self._solver.links_state.hibernated,
            self._solver.dofs_state.hibernated,
            self._solver.geoms_state.hibernated,
            self._solver.geoms_state.min_buffer_idx,
            self._solver.geoms_state.max_buffer_idx,
            rigid_global_info.n_awake_dofs,
            rigid_global_info.awake_dofs,
            rigid_global_info.n_awake_entities,
            rigid_global_info.awake_entities,
            rigid_global_info.n_awake_links,
            rigid_global_info.awake_links,
        ]
        contact_island = getattr(self._solver.constraint_solver, "contact_island", None)
        if contact_island is not None:
            tensors.extend(self.iter_state_tensors(contact_island.contact_island_state))
        return self.snapshot_tensor_state(tensors)

    @contextmanager
    def _planning_transaction(self, qpos_cur, envs_idx, obj_entity):
        """
        Context manager that snapshots all solver state the planner can mutate and restores it on
        normal return AND on any exception, including KeyboardInterrupt/SystemExit. Per-section
        restore failures are aggregated so one broken restore does not skip the others. On normal
        return, the first restore failure is raised; on an in-flight planner exception, restore
        failures are logged so they do not mask the planner exception.
        """
        obj_state = None
        collider_state = None
        hibernation_state = None
        constraint_state = None
        errno_state = None
        if obj_entity is not None:
            obj_state = self.snapshot_entity_state(obj_entity, envs_idx)
            hibernation_state = self.snapshot_hibernation_state()
            collider_state = self.snapshot_collider_state()
            constraint_state = self.snapshot_constraint_state()
            errno_state = self.snapshot_tensor_state([self._solver._errno])

        # Track an in-flight planner exception via `try/except BaseException; raise` so that the
        # finally clause can distinguish a normal exit from an exception unwind. Using
        # `sys.exc_info()` inside the finally is unsafe: if the caller invokes `plan_path` from
        # inside its own `except` block, `sys.exc_info()` would report the OUTER exception and
        # silently mask restore failures here.
        #
        # The outer `try:` is purely stylistic — a flat `try/except BaseException as exc: ...;
        # raise; finally: ...` is functionally identical. The nested form is chosen to visually
        # separate the planner-error capture from the restore steps; collapse if you prefer.
        planner_error: BaseException | None = None
        try:
            try:
                yield
            except BaseException as exc:
                planner_error = exc
                raise
        finally:
            errors: list[tuple[str, BaseException]] = []

            def attempt(scope, fn):
                # Catch `BaseException` (not just `Exception`) so a signal raised inside a restore
                # step does not skip the remaining steps and silently replace `planner_error` with
                # itself. If a `BaseException`-non-`Exception` is caught here it is still re-raised
                # at the end, after every restore has had a chance to run.
                try:
                    fn()
                except BaseException as exc:
                    errors.append((scope, exc))

            # Order matters: hibernation rebuilds awake buffers that the collider/constraint
            # snapshots are keyed against.
            if self._solver.n_envs > 0:
                attempt(
                    "robot qpos",
                    lambda: self._entity.set_qpos(qpos_cur, envs_idx=envs_idx, zero_velocity=False),
                )
            else:
                attempt("robot qpos", lambda: self._entity.set_qpos(qpos_cur, zero_velocity=False))
            if obj_entity is not None and obj_state is not None:
                attempt("object state", lambda: self.restore_entity_state(obj_entity, obj_state, envs_idx))
            if hibernation_state is not None:
                attempt("hibernation state", lambda: self.restore_tensor_state(hibernation_state))
            if collider_state is not None:
                attempt("collider state", lambda: self.restore_collider_state(collider_state))
            if constraint_state is not None:
                attempt("constraint state", lambda: self.restore_constraint_state(constraint_state))
            if errno_state is not None:
                attempt("errno state", lambda: self.restore_tensor_state(errno_state))

            if errors:
                # A `BaseException`-non-`Exception` (e.g. KeyboardInterrupt) must always win over
                # a regular Exception, both with and without an in-flight planner exception.
                base_errors = [e for _, e in errors if not isinstance(e, Exception)]
                if planner_error is not None:
                    for scope, exc in errors:
                        gs.logger.warning(f"Planner state restore failed ({scope}): {exc}")
                    if base_errors:
                        raise base_errors[0]
                elif base_errors:
                    for scope, exc in errors:
                        if exc is base_errors[0]:
                            continue
                        gs.logger.warning(f"Additional planner restore error ({scope}): {exc}")
                    raise base_errors[0]
                else:
                    for scope, exc in errors[1:]:
                        gs.logger.warning(f"Additional planner restore error ({scope}): {exc}")
                    raise errors[0][1]

    # ------------------------------------------------------------------------------------
    # ------------------------------ util funcs ------------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_qposs(self, qpos_goal, qpos_start, envs_idx):
        envs_idx = self.normalize_envs_idx(envs_idx)
        qpos_cur = self._entity.get_qpos(envs_idx=envs_idx).clone()

        assert qpos_goal is not None
        qpos_goal, *_ = self._solver._sanitize_io_variables(qpos_goal, None, self._entity.n_qs, "qpos_idx", envs_idx)
        if qpos_start is None:
            qpos_start = qpos_cur
        qpos_start, _, envs_idx = self._solver._sanitize_io_variables(
            qpos_start, None, self._entity.n_qs, "qpos_idx", envs_idx
        )
        if self._solver.n_envs == 0:
            qpos_goal = qpos_goal[None]
            qpos_start = qpos_start[None]

        return qpos_cur, qpos_goal, qpos_start, envs_idx

    def _validate_plan_object_args(self, ee_link_idx, obj_entity):
        if (ee_link_idx is None) != (obj_entity is None):
            gs.raise_exception("`ee_link_idx` and `obj_entity` must be specified together.")
        if obj_entity is not None:
            if obj_entity is self._entity:
                gs.raise_exception("`obj_entity` cannot be the planning entity itself.")
            if obj_entity._solver is not self._solver:
                gs.raise_exception("`obj_entity` must belong to the same scene as the planning entity.")
            if not any(link.idx == ee_link_idx for link in self._entity.links):
                gs.raise_exception("`ee_link_idx` must belong to the planning entity.")
            if len(obj_entity.links) != 1:
                gs.raise_exception("Only non-articulated objects are supported for now.")

    def _empty_plan_result(self, num_waypoints):
        path = torch.empty((num_waypoints, 0, self._entity.n_qs), dtype=gs.tc_float, device=gs.device)
        is_invalid = torch.empty((0,), dtype=torch.bool, device=gs.device)
        return path, is_invalid

    def get_exclude_geom_pairs(self, qposs, envs_idx):
        """
        Parameters
        ----------
        qposs : list of torch.Tensor
            List of qpos tensors to ignore the collision check.
        envs_idx : torch.Tensor
            Environment indices.

        Returns
        -------
        unique_pairs : torch.Tensor
            Unique pairs of geom indices to ignore the collision check.
        """
        if self._solver.n_envs > 0:
            envs_idx = self.normalize_envs_idx(envs_idx)

        collision_pairs = []
        for qpos in qposs:
            if self._solver.n_envs > 0:
                self._entity.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=False)
            else:
                self._entity.set_qpos(qpos[0], zero_velocity=False)
            self._solver._kernel_detect_collision()
            scene_contact_info = self._entity.get_contacts()
            geom_a = scene_contact_info["geom_a"]
            geom_b = scene_contact_info["geom_b"]
            if self._solver.n_envs > 0:
                valid_mask = scene_contact_info["valid_mask"]
                geom_a = geom_a[envs_idx]
                geom_b = geom_b[envs_idx]
                valid_mask = valid_mask[envs_idx]
                geom_a = geom_a[valid_mask]
                geom_b = geom_b[valid_mask]
            collision_pairs.append(torch.stack((geom_a, geom_b), dim=1))
        collision_pairs = torch.cat(collision_pairs, dim=0)  # N, 2

        if gs.backend != gs.metal:
            unique_pairs = torch.unique(collision_pairs, dim=0)
        else:
            # Apple Metal GPU backend does not support `torch.unique([...], dim=[...])`
            unique_pairs = torch.unique(collision_pairs.cpu(), dim=0).to(device=gs.device)
        return unique_pairs

    @qd.kernel
    def interpolate_path(
        self,
        path: qd.types.ndarray(),  # [N, B, Dof]
        sample_ind: qd.types.ndarray(),  # [B, 2]
        mask: qd.types.ndarray(),  # [B]
        tensor: qd.types.ndarray(),  # [N, B, Dof]
    ):
        qd.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(path.shape[1]):
            for i_q in range(self._entity.n_qs):
                for i_s in range(path.shape[0]):
                    tensor[i_s, i_b, i_q] = path[i_s, i_b, i_q]

        qd.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(path.shape[1]):
            if mask[i_b]:
                num_samples = sample_ind[i_b, 1] - sample_ind[i_b, 0]
                for i_q in range(self._entity.n_qs):
                    start = path[sample_ind[i_b, 0], i_b, i_q]
                    end = path[sample_ind[i_b, 1], i_b, i_q]
                    step = (end - start) / num_samples
                    for i_s in range(num_samples):
                        tensor[sample_ind[i_b, 0] + i_s, i_b, i_q] = start + step * i_s

    def check_collision(
        self,
        path,
        ignore_geom_pairs,
        envs_idx,
        *,
        is_plan_with_obj=False,
        obj_geom_start=-1,
        obj_geom_end=-1,
        # ee_link_idx/obj_link_idx/_pos/_quat are only consumed when is_plan_with_obj=True; they
        # default to None so callers don't need a sentinel int when no object is attached.
        ee_link_idx=None,
        obj_link_idx=None,
        _pos=None,
        _quat=None,
    ):
        out = torch.zeros((path.shape[1],), dtype=gs.tc_bool, device=gs.device)
        for qpos in path:
            if self._solver.n_envs > 0:
                self._entity.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=False)
            else:
                self._entity.set_qpos(qpos[0], zero_velocity=False)

            if is_plan_with_obj:
                self.update_object(ee_link_idx, obj_link_idx, _pos, _quat, envs_idx)
            self._solver._kernel_detect_collision()
            self._kernel_check_collision(
                ignore_geom_pairs,
                envs_idx,
                is_plan_with_obj=is_plan_with_obj,
                obj_geom_start=obj_geom_start,
                obj_geom_end=obj_geom_end,
                out=out,
                collider_state=self._solver.collider._collider_state,
            )
        return out

    @qd.kernel
    def _kernel_check_collision(
        self,
        ignore_geom_pairs: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        is_plan_with_obj: qd.i32,
        obj_geom_start: qd.i32,
        obj_geom_end: qd.i32,
        out: qd.types.ndarray(),
        collider_state: array_class.ColliderState,
    ):
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            collision_detected = self._func_check_collision(
                collider_state,
                ignore_geom_pairs,
                i_b,
                is_plan_with_obj=is_plan_with_obj,
                obj_geom_start=obj_geom_start,
                obj_geom_end=obj_geom_end,
            )
            out[i_b_] = out[i_b_] or qd.cast(collision_detected, gs.qd_bool)

    @qd.func
    def _func_check_collision(
        self,
        collider_state: array_class.ColliderState,
        ignore_geom_pairs: qd.types.ndarray(),
        i_b: qd.i32,
        is_plan_with_obj: qd.i32 = False,
        obj_geom_start: qd.i32 = -1,
        obj_geom_end: qd.i32 = -1,
    ) -> qd.i32:
        is_collision_detected = qd.cast(False, gs.qd_int)
        for i_c in range(collider_state.n_contacts[i_b]):
            if not is_collision_detected:
                i_ga = collider_state.contact_data.geom_a[i_c, i_b]
                i_gb = collider_state.contact_data.geom_b[i_c, i_b]

                is_ignored = False
                if collider_state.contact_data.penetration[i_c, i_b] < self.PENETRATION_EPS:
                    is_ignored = True
                for i_p in range(ignore_geom_pairs.shape[0]):
                    if not is_ignored:
                        if (ignore_geom_pairs[i_p, 0] == i_ga and ignore_geom_pairs[i_p, 1] == i_gb) or (
                            ignore_geom_pairs[i_p, 0] == i_gb and ignore_geom_pairs[i_p, 1] == i_ga
                        ):
                            is_ignored = True
                if not is_ignored:
                    if (self._entity.geom_start <= i_ga < self._entity.geom_end) or (
                        self._entity.geom_start <= i_gb < self._entity.geom_end
                    ):
                        is_collision_detected = True
                    if is_plan_with_obj:
                        if (obj_geom_start <= i_ga < obj_geom_end) or (obj_geom_start <= i_gb < obj_geom_end):
                            is_collision_detected = True
        return is_collision_detected

    def shortcut_path(
        self,
        path_mask,
        path,
        iterations=50,
        ignore_geom_pairs=None,
        envs_idx=None,
        is_plan_with_obj=False,
        obj_geom_start=-1,
        obj_geom_end=-1,
        # ee_link_idx/obj_link_idx are only consumed via `check_collision` when is_plan_with_obj=True;
        # default to None so callers don't need a sentinel int when no object is attached.
        ee_link_idx=None,
        obj_link_idx=None,
        _pos=None,
        _quat=None,
    ):
        """
        path_mask: torch.Tensor
            valid waypoint mask [N,B] for the obtained path
        path: torch.Tensor
            the [N,B,Dof] tensor containing batched waypoints
        iterations: int
            the number of refine iterations
        """
        # Need at least 3 waypoints to shortcut (multinomial samples 2 indices,
        # and the shortcut only applies when their gap > 1).
        if path.shape[0] < 3:
            return path

        for i in range(iterations):
            ind = torch.multinomial(path_mask.T, 2).sort().values.to(gs.tc_int)  # B, 2
            ind_mask = (ind[:, 1] - ind[:, 0]) > 1
            result_path = torch.empty_like(path)
            self.interpolate_path(path.contiguous(), ind, ind_mask, result_path)
            collision_mask = self.check_collision(
                result_path,
                ignore_geom_pairs,
                envs_idx,
                is_plan_with_obj=is_plan_with_obj,
                obj_geom_start=obj_geom_start,
                obj_geom_end=obj_geom_end,
                ee_link_idx=ee_link_idx,
                obj_link_idx=obj_link_idx,
                _pos=_pos,
                _quat=_quat,
            )  # B
            path[:, ~collision_mask] = result_path[:, ~collision_mask]
        return path


@qd.data_oriented
class RRT(PathPlanner):
    def __init__(self, entity):
        super().__init__(entity)
        self._is_rrt_init = False

    def _init_rrt_fields(self, goal_bias=0.05, max_nodes=2000, pos_tol=5e-3, max_step_size=0.1):
        if not self._is_rrt_init:
            self._rrt_goal_bias = goal_bias
            self._rrt_max_nodes = max_nodes
            self._rrt_pos_tol = pos_tol
            self._rrt_max_step_size = max_step_size
            self._rrt_start_configuration = qd.field(dtype=gs.qd_float, shape=(self._entity.n_qs, self._solver._B))
            self._rrt_goal_configuration = qd.field(dtype=gs.qd_float, shape=(self._entity.n_qs, self._solver._B))
            self.struct_rrt_node_info = qd.types.struct(
                configuration=qd.types.vector(self._entity.n_qs, gs.qd_float),
                parent_idx=gs.qd_int,
            )
            # FIXME: AOS, which does not match other Genesis structs. Old, untested code. We prefer not to touch for now.
            self._rrt_node_info = self.struct_rrt_node_info.field(shape=(self._rrt_max_nodes, self._solver._B))
            self._rrt_tree_size = qd.field(dtype=gs.qd_int, shape=(self._solver._B,))
            self._rrt_is_active = qd.field(dtype=gs.qd_bool, shape=(self._solver._B,))
            self._rrt_goal_reached_node_idx = qd.field(dtype=gs.qd_int, shape=(self._solver._B,))
            self._is_rrt_init = True

    def _reset_rrt_fields(self):
        self._rrt_start_configuration.fill(0.0)
        self._rrt_goal_configuration.fill(0.0)
        self._rrt_node_info.parent_idx.fill(-1)
        self._rrt_node_info.configuration.fill(0.0)
        self._rrt_tree_size.fill(0)
        self._rrt_is_active.fill(False)
        self._rrt_goal_reached_node_idx.fill(-1)

    @qd.kernel
    def _kernel_rrt_init(
        self, qpos_start: qd.types.ndarray(), qpos_goal: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        qd.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_q in range(self._entity.n_qs):
                # save original qpos
                self._rrt_start_configuration[i_q, i_b] = qpos_start[i_b_, i_q]
                self._rrt_goal_configuration[i_q, i_b] = qpos_goal[i_b_, i_q]
                self._rrt_node_info[0, i_b].configuration[i_q] = qpos_start[i_b_, i_q]
            self._rrt_node_info[0, i_b].parent_idx = 0
            self._rrt_tree_size[i_b] = 1
            self._rrt_is_active[i_b] = True

    @qd.kernel
    def _kernel_rrt_step1(
        self,
        qpos: qd.Tensor,
        q_limit_lower: qd.types.ndarray(),
        q_limit_upper: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        joints_state: array_class.JointsState,
        joints_info: array_class.JointsInfo,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        dofs_state: array_class.DofsState,
        dofs_info: array_class.DofsInfo,
        entities_info: array_class.EntitiesInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Step 1 includes:
        - generate random sample
        - find nearest neighbor
        - steer from nearest neighbor to random sample
        - add new node
        - set the steer result (to prepare for collision checking)
        """
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            if self._rrt_is_active[i_b]:
                random_sample = qd.Vector(
                    [
                        q_limit_lower[i_q] + qd.random(dtype=gs.qd_float) * (q_limit_upper[i_q] - q_limit_lower[i_q])
                        for i_q in range(self._entity.n_qs)
                    ]
                )
                if qd.random() < self._rrt_goal_bias:
                    random_sample = qd.Vector(
                        [self._rrt_goal_configuration[i_q, i_b] for i_q in range(self._entity.n_qs)]
                    )

                # find nearest neighbor
                nearest_neighbor_idx = -1
                nearest_neighbor_dist = gs.qd_float(1e30)
                for i_n in range(self._rrt_tree_size[i_b]):
                    dist = (self._rrt_node_info.configuration[i_n, i_b] - random_sample).norm_sqr()
                    if dist < nearest_neighbor_dist:
                        nearest_neighbor_dist = dist
                        nearest_neighbor_idx = i_n

                # steer from nearest neighbor to random sample
                nearest_config = self._rrt_node_info.configuration[nearest_neighbor_idx, i_b]
                direction = random_sample - nearest_config
                steer_result = qd.Vector.zero(gs.qd_float, self._entity.n_qs)
                for i_q in range(self._entity.n_qs):
                    # If the step size exceeds max_step_size, clip it
                    if qd.abs(direction[i_q]) > self._rrt_max_step_size:
                        direction[i_q] = (-1.0 if direction[i_q] < 0.0 else 1.0) * self._rrt_max_step_size
                    steer_result[i_q] = nearest_config[i_q] + direction[i_q]

                if self._rrt_tree_size[i_b] < self._rrt_max_nodes - 1:
                    # add new node
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = steer_result
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = nearest_neighbor_idx
                    self._rrt_tree_size[i_b] += 1

                    # set the steer result and collision check for i_b
                    for i_q in range(self._entity.n_qs):
                        qpos[i_q + self._entity._q_start, i_b] = steer_result[i_q]
                    gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                        self._entity._idx_in_solver,
                        i_b,
                        links_state,
                        links_info,
                        joints_state,
                        joints_info,
                        dofs_state,
                        dofs_info,
                        entities_info,
                        rigid_global_info,
                        self._solver._static_rigid_sim_config,
                        is_backward=False,
                    )
                    gs.engine.solvers.rigid.rigid_solver.func_update_geoms_batch(
                        i_b,
                        entities_info,
                        geoms_state,
                        geoms_info,
                        links_state,
                        rigid_global_info,
                        self._solver._static_rigid_sim_config,
                        force_update_fixed_geoms=False,
                        is_backward=False,
                    )

    @qd.kernel
    def _kernel_rrt_step2(
        self,
        qpos: qd.Tensor,
        ignore_geom_pairs: qd.types.ndarray(),
        ignore_collision: qd.i32,
        envs_idx: qd.types.ndarray(),
        is_plan_with_obj: qd.i32,
        obj_geom_start: qd.i32,
        obj_geom_end: qd.i32,
        collider_state: array_class.ColliderState,
    ):
        """
        Step 2 includes:
        - check collision
        - if collision is detected, remove the new node
        - if collision is not detected, check if the new node is within goal configuration
        """
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            if self._rrt_is_active[i_b]:
                is_collision_detected = qd.cast(False, gs.qd_int)
                if not ignore_collision:
                    is_collision_detected = self._func_check_collision(
                        collider_state, ignore_geom_pairs, i_b, is_plan_with_obj, obj_geom_start, obj_geom_end
                    )
                if is_collision_detected:
                    self._rrt_tree_size[i_b] -= 1
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = 0.0
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = -1
                else:
                    # check the obtained steer result is within goal configuration only if no collision
                    is_goal = True
                    for i_q in range(self._entity.n_qs):
                        if (
                            qd.abs(qpos[i_q + self._entity._q_start, i_b] - self._rrt_goal_configuration[i_q, i_b])
                            > self._rrt_pos_tol
                        ):
                            is_goal = False
                            break
                    if is_goal:
                        self._rrt_goal_reached_node_idx[i_b] = self._rrt_tree_size[i_b] - 1
                        self._rrt_is_active[i_b] = False

    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        resolution=0.05,
        timeout=None,
        max_nodes=2000,
        smooth_path=True,
        num_waypoints=100,
        ignore_collision=False,
        ee_link_idx=None,
        obj_entity=None,
        envs_idx=None,
    ):
        if qpos_goal is None:
            gs.raise_exception("`qpos_goal` must be specified.")
        if self._solver.n_envs == 0 and envs_idx is not None:
            gs.raise_exception("`envs_idx` is only supported for batched scenes.")

        self._validate_plan_object_args(ee_link_idx, obj_entity)
        envs_idx = self.normalize_envs_idx(envs_idx)
        if self._solver.n_envs > 0 and len(envs_idx) == 0:
            return self._empty_plan_result(num_waypoints)
        qpos_cur, qpos_goal, qpos_start, envs_idx = self._sanitize_qposs(qpos_goal, qpos_start, envs_idx)

        is_plan_with_obj = ee_link_idx is not None and obj_entity is not None
        if is_plan_with_obj:
            obj_geom_start = obj_entity.geom_start
            obj_geom_end = obj_entity.geom_end
            obj_link_idx = obj_entity.base_link_idx
            _pos, _quat = self.get_link_pose(ee_link_idx, obj_link_idx, envs_idx)
        else:
            obj_geom_start, obj_geom_end = -1, -1
            obj_link_idx = None
            _pos, _quat = None, None

        with self._planning_transaction(qpos_cur, envs_idx, obj_entity if is_plan_with_obj else None):
            ignore_geom_pairs = self.get_exclude_geom_pairs((qpos_goal, qpos_start), envs_idx)

            self._init_rrt_fields(max_nodes=max_nodes, max_step_size=resolution)
            self._reset_rrt_fields()
            self._kernel_rrt_init(qpos_start, qpos_goal, envs_idx)

            gs.logger.debug("Start RRT planning...")
            time_start = time.time()
            for _ in range(self._rrt_max_nodes):
                if self._rrt_is_active.to_torch().any():
                    self._kernel_rrt_step1(
                        qpos=self._solver.qpos,
                        q_limit_lower=self._entity.q_limit[0],
                        q_limit_upper=self._entity.q_limit[1],
                        envs_idx=envs_idx,
                        links_state=self._solver.links_state,
                        links_info=self._solver.links_info,
                        joints_state=self._solver.joints_state,
                        joints_info=self._solver.joints_info,
                        geoms_state=self._solver.geoms_state,
                        geoms_info=self._solver.geoms_info,
                        dofs_state=self._solver.dofs_state,
                        dofs_info=self._solver.dofs_info,
                        entities_info=self._solver.entities_info,
                        rigid_global_info=self._solver._rigid_global_info,
                    )
                    if is_plan_with_obj:
                        self.update_object(ee_link_idx, obj_link_idx, _pos, _quat, envs_idx)
                    self._solver._kernel_detect_collision()
                    self._kernel_rrt_step2(
                        qpos=self._solver.qpos,
                        collider_state=self._solver.collider._collider_state,
                        ignore_geom_pairs=ignore_geom_pairs,
                        ignore_collision=ignore_collision,
                        envs_idx=envs_idx,
                        is_plan_with_obj=is_plan_with_obj,
                        obj_geom_start=obj_geom_start,
                        obj_geom_end=obj_geom_end,
                    )
                else:
                    break
                if timeout is not None:
                    if time.time() - time_start > timeout:
                        gs.logger.info("RRT planning timeout.")
                        break

            gs.logger.debug(f"RRT planning time: {time.time() - time_start}")

            is_invalid = self._rrt_is_active.to_torch(device=gs.device).bool()[envs_idx]
            ts = self._rrt_tree_size.to_torch(device=gs.device)
            g_n = self._rrt_goal_reached_node_idx.to_torch(device=gs.device)[envs_idx]  # B

            node_info = self._rrt_node_info.to_torch(device=gs.device)
            parents_idx = node_info["parent_idx"]
            configurations = node_info["configuration"]

            res = [g_n]
            for _ in range(ts.max()):
                g_n = parents_idx[g_n, envs_idx]
                res.append(g_n)
                if (g_n == 0).all():
                    break
            res_idx = torch.stack(res[::-1], dim=0)
            sol = configurations[res_idx, envs_idx]  # N, B, DoF

            if is_invalid.all():
                sol = torch.zeros((num_waypoints, len(envs_idx), sol.shape[-1]), dtype=gs.tc_float, device=gs.device)
                return sol, is_invalid

            mask = rrt_valid_mask(res_idx)
            if self._solver.n_envs > 1:
                sol = align_waypoints_length(sol, mask, mask.sum(dim=0).max())
            if smooth_path:
                sol = self.shortcut_path(
                    torch.ones_like(sol[..., 0]),
                    sol,
                    iterations=10,
                    ignore_geom_pairs=ignore_geom_pairs,
                    envs_idx=envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                    ee_link_idx=ee_link_idx,
                    obj_link_idx=obj_link_idx,
                    _pos=_pos,
                    _quat=_quat,
                )
            sol = align_waypoints_length(sol, torch.ones_like(sol[..., 0], dtype=torch.bool), num_waypoints)

            if not ignore_collision:
                is_invalid |= self.check_collision(
                    sol,
                    ignore_geom_pairs,
                    envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                    ee_link_idx=ee_link_idx,
                    obj_link_idx=obj_link_idx,
                    _pos=_pos,
                    _quat=_quat,
                ).bool()

            if is_invalid.any():
                gs.logger.info(f"RRT planning failed in {int(is_invalid.sum())} environments")
            return sol, is_invalid


@qd.data_oriented
class RRTConnect(PathPlanner):
    def __init__(self, entity):
        super().__init__(entity)
        self._is_rrt_connect_init = False

    def _init_rrt_connect_fields(self, goal_bias=0.1, max_nodes=4000, max_step_size=0.05):
        if not self._is_rrt_connect_init:
            self._rrt_goal_bias = goal_bias
            self._rrt_max_nodes = max_nodes
            self._rrt_max_step_size = max_step_size
            self._rrt_start_configuration = qd.field(dtype=gs.qd_float, shape=(self._entity.n_qs, self._solver._B))
            self._rrt_goal_configuration = qd.field(dtype=gs.qd_float, shape=(self._entity.n_qs, self._solver._B))
            self.struct_rrt_node_info = qd.types.struct(
                configuration=qd.types.vector(self._entity.n_qs, gs.qd_float),
                parent_idx=gs.qd_int,
                child_idx=gs.qd_int,
            )
            # FIXME: AOS, which does not match other Genesis structs. Old, untested code. We prefer not to touch for now.
            self._rrt_node_info = self.struct_rrt_node_info.field(shape=(self._rrt_max_nodes, self._solver._B))
            self._rrt_tree_size = qd.field(dtype=gs.qd_int, shape=(self._solver._B,))
            self._rrt_is_active = qd.field(dtype=gs.qd_bool, shape=(self._solver._B,))
            self._rrt_goal_reached_node_idx = qd.field(dtype=gs.qd_int, shape=(self._solver._B,))
            self._is_rrt_connect_init = True

    def _reset_rrt_connect_fields(self):
        self._rrt_start_configuration.fill(0.0)
        self._rrt_goal_configuration.fill(0.0)
        self._rrt_node_info.parent_idx.fill(-1)
        self._rrt_node_info.child_idx.fill(-1)
        self._rrt_node_info.configuration.fill(0.0)
        self._rrt_tree_size.fill(0)
        self._rrt_is_active.fill(False)
        self._rrt_goal_reached_node_idx.fill(-1)

    @qd.kernel
    def _kernel_rrt_connect_init(
        self, qpos_start: qd.types.ndarray(), qpos_goal: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        # NOTE: run IK before this
        qd.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_q in range(self._entity.n_qs):
                # save original qpos
                self._rrt_start_configuration[i_q, i_b] = qpos_start[i_b_, i_q]
                self._rrt_goal_configuration[i_q, i_b] = qpos_goal[i_b_, i_q]
                self._rrt_node_info[0, i_b].configuration[i_q] = qpos_start[i_b_, i_q]
                self._rrt_node_info[1, i_b].configuration[i_q] = qpos_goal[i_b_, i_q]
            self._rrt_node_info[0, i_b].parent_idx = 0
            self._rrt_node_info[1, i_b].child_idx = 1
            self._rrt_tree_size[i_b] = 2
            self._rrt_is_active[i_b] = True

    @qd.kernel
    def _kernel_rrt_connect_step1(
        self,
        qpos: qd.Tensor,
        forward_pass: qd.i32,
        q_limit_lower: qd.types.ndarray(),
        q_limit_upper: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        joints_state: array_class.JointsState,
        joints_info: array_class.JointsInfo,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        dofs_state: array_class.DofsState,
        dofs_info: array_class.DofsInfo,
        entities_info: array_class.EntitiesInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Step 1 includes:
        - generate random sample
        - find nearest neighbor
        - steer from nearest neighbor to random sample
        - add new node
        - set the steer result (to prepare for collision checking)
        """
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            if self._rrt_is_active[i_b]:
                random_sample = qd.Vector(
                    [
                        q_limit_lower[i_q] + qd.random(dtype=gs.qd_float) * (q_limit_upper[i_q] - q_limit_lower[i_q])
                        for i_q in range(self._entity.n_qs)
                    ]
                )
                if qd.random() < self._rrt_goal_bias:
                    if forward_pass:
                        random_sample = qd.Vector(
                            [self._rrt_goal_configuration[i_q, i_b] for i_q in range(self._entity.n_qs)]
                        )
                    else:
                        random_sample = qd.Vector(
                            [self._rrt_start_configuration[i_q, i_b] for i_q in range(self._entity.n_qs)]
                        )

                # find nearest neighbor
                nearest_neighbor_idx = -1
                nearest_neighbor_dist = gs.qd_float(1e30)
                for i_n in range(self._rrt_tree_size[i_b]):
                    if forward_pass:
                        # NOTE: in forward pass, we only consider the previous forward pass nodes (which has parent_idx != -1)
                        if self._rrt_node_info[i_n, i_b].parent_idx == -1:
                            continue
                    else:
                        # NOTE: in backward pass, we only consider the previous backward pass nodes (which has child_idx != -1)
                        if self._rrt_node_info[i_n, i_b].child_idx == -1:
                            continue
                    dist = (self._rrt_node_info.configuration[i_n, i_b] - random_sample).norm_sqr()
                    if dist < nearest_neighbor_dist:
                        nearest_neighbor_dist = dist
                        nearest_neighbor_idx = i_n

                # steer from nearest neighbor to random sample
                nearest_config = self._rrt_node_info.configuration[nearest_neighbor_idx, i_b]
                direction = random_sample - nearest_config
                steer_result = qd.Vector.zero(gs.qd_float, self._entity.n_qs)
                for i_q in range(self._entity.n_qs):
                    # If the step size exceeds max_step_size, clip it
                    if qd.abs(direction[i_q]) > self._rrt_max_step_size:
                        direction[i_q] = (-1.0 if direction[i_q] < 0.0 else 1.0) * self._rrt_max_step_size
                    steer_result[i_q] = nearest_config[i_q] + direction[i_q]

                if self._rrt_tree_size[i_b] < self._rrt_max_nodes - 1:
                    # add new node
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = steer_result
                    if forward_pass:
                        self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = nearest_neighbor_idx
                    else:
                        self._rrt_node_info[self._rrt_tree_size[i_b], i_b].child_idx = nearest_neighbor_idx
                    self._rrt_tree_size[i_b] += 1

                    # set the steer result and collision check for i_b
                    for i_q in range(self._entity.n_qs):
                        qpos[i_q + self._entity._q_start, i_b] = steer_result[i_q]
                    gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                        self._entity._idx_in_solver,
                        i_b,
                        links_state,
                        links_info,
                        joints_state,
                        joints_info,
                        dofs_state,
                        dofs_info,
                        entities_info,
                        rigid_global_info,
                        self._solver._static_rigid_sim_config,
                        is_backward=False,
                    )
                    gs.engine.solvers.rigid.rigid_solver.func_update_geoms_batch(
                        i_b,
                        entities_info,
                        geoms_state,
                        geoms_info,
                        links_state,
                        rigid_global_info,
                        self._solver._static_rigid_sim_config,
                        force_update_fixed_geoms=False,
                        is_backward=False,
                    )

    @qd.kernel
    def _kernel_rrt_connect_step2(
        self,
        forward_pass: qd.i32,
        ignore_geom_pairs: qd.types.ndarray(),
        ignore_collision: qd.i32,
        envs_idx: qd.types.ndarray(),
        is_plan_with_obj: qd.i32,
        obj_geom_start: qd.i32,
        obj_geom_end: qd.i32,
        collider_state: array_class.ColliderState,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Step 2 includes:
        - check collision
        - if collision is detected, remove the new node
        - if collision is not detected, check if the new node is within goal configuration
        """
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            if self._rrt_is_active[i_b]:
                is_collision_detected = qd.cast(False, gs.qd_int)
                if not ignore_collision:
                    is_collision_detected = self._func_check_collision(
                        collider_state, ignore_geom_pairs, i_b, is_plan_with_obj, obj_geom_start, obj_geom_end
                    )
                if is_collision_detected:
                    self._rrt_tree_size[i_b] -= 1
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = 0.0
                    if forward_pass:
                        self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = -1
                    else:
                        self._rrt_node_info[self._rrt_tree_size[i_b], i_b].child_idx = -1
                else:
                    # check the obtained steer result is within goal configuration only if no collision
                    for i_n in range(self._rrt_tree_size[i_b]):
                        if forward_pass:
                            if self._rrt_node_info[i_n, i_b].child_idx == -1:
                                continue
                        else:
                            if self._rrt_node_info[i_n, i_b].parent_idx == -1:
                                continue
                        is_connected = True
                        for i_q in range(self._entity.n_qs):
                            if (
                                qd.abs(
                                    rigid_global_info.qpos[i_q + self._entity._q_start, i_b]
                                    - self._rrt_node_info.configuration[i_n, i_b][i_q]
                                )
                                > self._rrt_max_step_size
                            ):
                                is_connected = False
                                break
                        if is_connected:
                            self._rrt_goal_reached_node_idx[i_b] = self._rrt_tree_size[i_b] - 1
                            if forward_pass:
                                self._rrt_node_info[self._rrt_tree_size[i_b] - 1, i_b].child_idx = i_n
                            else:
                                self._rrt_node_info[self._rrt_tree_size[i_b] - 1, i_b].parent_idx = i_n
                            self._rrt_is_active[i_b] = False
                            break

    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        resolution=0.05,
        timeout=None,
        max_nodes=4000,
        smooth_path=True,
        num_waypoints=300,
        ignore_collision=False,
        ee_link_idx=None,
        obj_entity=None,
        envs_idx=None,
    ):
        if qpos_goal is None:
            gs.raise_exception("`qpos_goal` must be specified.")
        if self._solver.n_envs == 0 and envs_idx is not None:
            gs.raise_exception("`envs_idx` is only supported for batched scenes.")

        self._validate_plan_object_args(ee_link_idx, obj_entity)
        envs_idx = self.normalize_envs_idx(envs_idx)
        if self._solver.n_envs > 0 and len(envs_idx) == 0:
            return self._empty_plan_result(num_waypoints)
        qpos_cur, qpos_goal, qpos_start, envs_idx = self._sanitize_qposs(qpos_goal, qpos_start, envs_idx)

        is_plan_with_obj = ee_link_idx is not None and obj_entity is not None
        if is_plan_with_obj:
            obj_geom_start = obj_entity.geom_start
            obj_geom_end = obj_entity.geom_end
            obj_link_idx = obj_entity.base_link_idx
            _pos, _quat = self.get_link_pose(ee_link_idx, obj_link_idx, envs_idx)
        else:
            obj_geom_start, obj_geom_end = -1, -1
            obj_link_idx = None
            _pos, _quat = None, None

        with self._planning_transaction(qpos_cur, envs_idx, obj_entity if is_plan_with_obj else None):
            ignore_geom_pairs = self.get_exclude_geom_pairs([qpos_goal, qpos_start], envs_idx)

            self._init_rrt_connect_fields(max_nodes=max_nodes, max_step_size=resolution)
            self._reset_rrt_connect_fields()
            self._kernel_rrt_connect_init(qpos_start, qpos_goal, envs_idx)

            gs.logger.debug("Start RRTConnect planning...")
            time_start = time.time()
            forward_pass = True
            for _ in range(self._rrt_max_nodes):
                self._kernel_rrt_connect_step1(
                    qpos=self._solver.qpos,
                    forward_pass=forward_pass,
                    q_limit_lower=self._entity.q_limit[0],
                    q_limit_upper=self._entity.q_limit[1],
                    envs_idx=envs_idx,
                    links_state=self._solver.links_state,
                    links_info=self._solver.links_info,
                    joints_state=self._solver.joints_state,
                    joints_info=self._solver.joints_info,
                    geoms_state=self._solver.geoms_state,
                    geoms_info=self._solver.geoms_info,
                    dofs_state=self._solver.dofs_state,
                    dofs_info=self._solver.dofs_info,
                    entities_info=self._solver.entities_info,
                    rigid_global_info=self._solver._rigid_global_info,
                )
                if is_plan_with_obj:
                    self.update_object(ee_link_idx, obj_link_idx, _pos, _quat, envs_idx)
                self._solver._kernel_detect_collision()
                self._kernel_rrt_connect_step2(
                    forward_pass=forward_pass,
                    ignore_geom_pairs=ignore_geom_pairs,
                    ignore_collision=ignore_collision,
                    envs_idx=envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                    collider_state=self._solver.collider._collider_state,
                    rigid_global_info=self._solver._rigid_global_info,
                )
                forward_pass = not forward_pass

                if not self._rrt_is_active.to_torch().any():
                    break
                if timeout is not None:
                    if time.time() - time_start > timeout:
                        gs.logger.info("RRTConnect planning timeout.")
                        break
            else:
                gs.logger.info(f"RRTConnect planning exceeded maximum number of nodes ({self._rrt_max_nodes}).")

            gs.logger.debug(f"RRTConnect planning time: {time.time() - time_start}")
            is_invalid = self._rrt_is_active.to_torch(device=gs.device).bool()[envs_idx]
            ts = self._rrt_tree_size.to_torch(device=gs.device)
            g_n = self._rrt_goal_reached_node_idx.to_torch(device=gs.device)[envs_idx]  # B

            node_info = self._rrt_node_info.to_torch(device=gs.device)
            parents_idx = node_info["parent_idx"]
            children_idx = node_info["child_idx"]
            configurations = node_info["configuration"]

            res = [g_n]
            for _ in range(ts.max() // 2):
                g_n = parents_idx[g_n, envs_idx]
                res.append(g_n)
                if torch.all(g_n == 0):
                    break
            res_idx = torch.stack(res[::-1], dim=0)

            c_n = self._rrt_goal_reached_node_idx.to_torch(device=gs.device)[envs_idx]  # B
            res = []
            for _ in range(ts.max() // 2):
                c_n = children_idx[c_n, envs_idx]
                res.append(c_n)
                if torch.all(c_n == 1):
                    break
            res_idx = torch.cat([res_idx, torch.stack(res, dim=0)], dim=0)
            sol = configurations[res_idx, envs_idx]  # N, B, DoF

            if is_invalid.all():
                return torch.zeros(num_waypoints, len(envs_idx), sol.shape[-1], device=gs.device), is_invalid

            mask = rrt_connect_valid_mask(res_idx)
            if self._solver.n_envs > 1:
                sol = align_waypoints_length(sol, mask, mask.sum(dim=0).max())
            if smooth_path:
                sol = self.shortcut_path(
                    torch.ones_like(sol[..., 0]),
                    sol,
                    iterations=10,
                    ignore_geom_pairs=ignore_geom_pairs,
                    envs_idx=envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                    ee_link_idx=ee_link_idx,
                    obj_link_idx=obj_link_idx,
                    _pos=_pos,
                    _quat=_quat,
                )
            sol = align_waypoints_length(sol, torch.ones_like(sol[..., 0], dtype=torch.bool), num_waypoints)

            if not ignore_collision:
                is_invalid |= self.check_collision(
                    sol,
                    ignore_geom_pairs,
                    envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                    ee_link_idx=ee_link_idx,
                    obj_link_idx=obj_link_idx,
                    _pos=_pos,
                    _quat=_quat,
                ).bool()

            if is_invalid.any():
                gs.logger.info(f"RRTConnect planning failed in {int(is_invalid.sum())} environments")

            return sol, is_invalid


# ------------------------------------------------------------------------------------
# ------------------------------------ utils -----------------------------------------
# ------------------------------------------------------------------------------------


def align_waypoints_length(path: torch.Tensor, mask: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    Aligns each waypoints length to the given num_points.

    Parameters
    ----------
    path: torch.Tensor
        path tensor in [N, B, Dof]
    mask: torch.Tensor
        the masking of path, indicating active waypoints [N, B]
    num_points: int
        the number of the desired waypoints

    Returns
    -------
        A new 2D PyTorch tensor [num_points, B, Dof]
    """
    t_path = path.permute(1, 2, 0)  # [B, Dof, N]
    res = torch.zeros(
        (num_points, t_path.shape[0], t_path.shape[1]), dtype=gs.tc_float, device=gs.device
    )  # [num_points, B, Dof]
    for i_b in range(t_path.shape[0]):
        if not mask[:, i_b].any():
            continue
        interpolated_path = torch.nn.functional.interpolate(
            t_path[i_b : i_b + 1, :, mask[:, i_b]], size=num_points, mode="linear", align_corners=True
        )[0]
        res[:, i_b] = interpolated_path.T
    return res


def rrt_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the RRTConnect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    mask = (tensor > 0.0).to(gs.tc_float)  # N, B
    mask_float = mask.T[:, None]  # B 1, N
    kernel = torch.ones((1, 1, 3), device=tensor.device, dtype=gs.tc_float)
    dilated_mask_float = F.conv1d(mask_float, kernel.to(mask_float.dtype), padding="same")
    dilated_mask = (dilated_mask_float > 0.0).squeeze(1).T
    return dilated_mask


def rrt_connect_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the RRTConnect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    mask = (tensor > 0.0).to(gs.tc_float)  # N, B
    mask_float = mask.T[:, None]  # B 1, N
    kernel = torch.ones(1, 1, 3, device=tensor.device, dtype=gs.tc_float)
    dilated_mask_float = F.conv1d(mask_float, kernel.to(mask_float.dtype), padding="same")
    dilated_mask = (dilated_mask_float > 0).squeeze(1).T
    return dilated_mask
