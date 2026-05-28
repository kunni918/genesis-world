import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import qd_to_torch

from .utils import assert_allclose


START_QPOS = np.array(
    [
        1.6877899169921875,
        -1.7629612684249878,
        -1.1454275846481323,
        -2.7444674968719482,
        -1.2441562414169312,
        1.2233363389968872,
        0.5208253860473633,
        0.03999991714954376,
        0.040000006556510925,
    ],
    dtype=np.float32,
)


def _add_restore_test_entities(scene, *, n_envs=0):
    anchor = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.0, 0.05),
            fixed=True,
        )
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.30, 0.10, 0.35),
        )
    )
    scene.build(n_envs=n_envs)
    return anchor, cube


def _assert_qd_state_matches(snapshot):
    for tensor, expected in snapshot:
        assert torch.equal(qd_to_torch(tensor, copy=True), expected)


def _assert_plan_valid(valid):
    if isinstance(valid, torch.Tensor):
        assert bool(valid.all())
    else:
        assert bool(valid)


def _base_collider_snapshot_ids(planner, collider_state):
    expected_ids = {
        id(collider_state.n_contacts),
        id(collider_state.n_contacts_hibernated),
        id(collider_state.first_time),
    }
    expected_ids.update(id(tensor) for tensor in planner.iter_state_tensors(collider_state.contact_data))
    expected_ids.update(id(tensor) for tensor in planner.iter_state_tensors(collider_state.contact_cache))
    return expected_ids


def _force_rrt_goal_sampling(monkeypatch):
    """Pin RRT's goal-bias to 1.0 so tests that assert path validity under tiny `max_nodes` are
    deterministic. The default 0.05 goal-bias is goal-bias-flaky for short-budget plans."""
    from genesis.utils.path_planning import RRT

    original_init_rrt_fields = RRT._init_rrt_fields

    def patched(self, goal_bias=0.05, max_nodes=2000, pos_tol=5e-3, max_step_size=0.1):
        original_init_rrt_fields(
            self, goal_bias=1.0, max_nodes=max_nodes, pos_tol=pos_tol, max_step_size=max_step_size
        )

    monkeypatch.setattr(RRT, "_init_rrt_fields", patched)


def _make_planner_scene(*, requires_grad: bool = False, **rigid_overrides):
    """Common scene factory for planner state tests: disables collision/constraint pipelines and
    hibernation/contact-island so the test scene is fully deterministic and cheap to build."""
    rigid_options = dict(
        enable_collision=False,
        enable_self_collision=False,
        enable_joint_limit=False,
        disable_constraint=True,
        use_contact_island=False,
        use_hibernation=False,
    )
    rigid_options.update(rigid_overrides)
    kwargs = dict(rigid_options=gs.options.RigidOptions(**rigid_options), show_viewer=False)
    if requires_grad:
        kwargs["sim_options"] = gs.options.SimOptions(requires_grad=True)
    return gs.Scene(**kwargs)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_internal_object_restore_does_not_update_tracked_targets(backend):
    from genesis.utils.path_planning import RRT

    scene = _make_planner_scene(requires_grad=True)
    anchor, cube = _add_restore_test_entities(scene)

    planner = RRT(anchor)
    state = planner.snapshot_entity_state(cube, envs_idx=None)
    cube._tgt.clear()

    planner.restore_entity_state(cube, state, envs_idx=None)

    assert cube._tgt == {}


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_internal_object_restore_respects_partial_envs_idx(backend, tol):
    from genesis.utils.path_planning import RRT

    scene = gs.Scene(show_viewer=False)
    anchor, cube = _add_restore_test_entities(scene, n_envs=2)
    planner = RRT(anchor)

    envs_idx = torch.tensor([1], dtype=gs.tc_int, device=gs.device)
    state = planner.snapshot_entity_state(cube, envs_idx=envs_idx)
    pos_before = cube.get_pos().clone()
    quat_before = cube.get_quat().clone()
    velocity_before = cube.get_dofs_velocity().clone()

    cube.set_pos(np.array([[0.1, -0.2, 0.4]], dtype=np.float32), envs_idx=envs_idx)
    cube.set_quat(np.array([[0.9238795, 0.0, 0.3826834, 0.0]], dtype=np.float32), envs_idx=envs_idx)
    cube.set_dofs_velocity(np.ones((1, cube.n_dofs), dtype=np.float32), envs_idx=envs_idx)

    planner.restore_entity_state(cube, state, envs_idx=envs_idx)

    assert_allclose(cube.get_pos(), pos_before, tol=tol)
    assert_allclose(cube.get_quat(), quat_before, tol=tol)
    assert_allclose(cube.get_dofs_velocity(), velocity_before, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_with_entity_restores_state_from_explicit_start_qpos(backend, planner, tol, monkeypatch):
    if planner == "RRT":
        _force_rrt_goal_sampling(monkeypatch)

    scene = _make_planner_scene()
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.30, 0.10, 0.35),
        )
    )
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    robot_qpos_before = franka.get_qpos().clone()
    cube_pos_before = cube.get_pos().clone()
    cube_quat_before = cube.get_quat().clone()
    cube_vel_before = cube.get_dofs_velocity().clone()

    path, valid = franka.plan_path(
        qpos_goal=START_QPOS,
        qpos_start=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        max_retry=0,
        smooth_path=False,
        ignore_collision=True,
        planner=planner,
        ee_link_name="hand",
        with_entity=cube,
        return_valid_mask=True,
    )

    _assert_plan_valid(valid)
    assert path.shape == (8, franka.n_qs)
    assert_allclose(franka.get_qpos(), robot_qpos_before, tol=tol)
    assert_allclose(cube.get_pos(), cube_pos_before, tol=tol)
    assert_allclose(cube.get_quat(), cube_quat_before, tol=tol)
    assert_allclose(cube.get_dofs_velocity(), cube_vel_before, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_with_entity_restores_existing_collider_state(backend):
    from genesis.utils.path_planning import RRTConnect

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.30, 0.10, 0.02),
        )
    )
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    scene.rigid_solver.collider.clear()
    scene.rigid_solver.collider.detection()
    collider_state = scene.rigid_solver.collider._collider_state
    planner = RRTConnect(franka)
    collider_snapshot = planner.snapshot_collider_state()
    n_contacts_before = qd_to_torch(collider_state.n_contacts, copy=True).clone()
    assert int(n_contacts_before[0]) > 0

    path, valid = franka.plan_path(
        qpos_goal=START_QPOS,
        qpos_start=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        max_retry=0,
        smooth_path=False,
        ignore_collision=True,
        planner="RRTConnect",
        ee_link_name="hand",
        with_entity=cube,
        return_valid_mask=True,
    )

    _assert_plan_valid(valid)
    assert path.shape == (8, franka.n_qs)
    _assert_qd_state_matches(collider_snapshot)


@pytest.mark.required
@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_with_entity_retries_when_planner_reports_invalid(backend, monkeypatch):
    """Wrapper-level: when the inner planner reports `is_invalid=True`, `plan_path` retries up to
    `max_retry+1` times and merges results. State restoration is the inner planner's contract and is
    covered by the direct-planner tests; the wrapper is a trivial delegator after the refactor."""
    from genesis.utils.path_planning import RRTConnect

    scene = gs.Scene(show_viewer=False)
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.30, 0.10, 0.35),
        )
    )
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    calls = []

    def fake_plan(self, *args, **kwargs):
        calls.append(kwargs.get("envs_idx"))
        is_invalid = torch.tensor([len(calls) == 1], dtype=torch.bool, device=gs.device)
        path = torch.zeros((kwargs["num_waypoints"], 1, self._entity.n_qs), dtype=gs.tc_float, device=gs.device)
        return path, is_invalid

    monkeypatch.setattr(RRTConnect, "plan", fake_plan)

    path, valid = franka.plan_path(
        qpos_goal=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        max_retry=1,
        smooth_path=False,
        ignore_collision=True,
        planner="RRTConnect",
        ee_link_name="hand",
        with_entity=cube,
        return_valid_mask=True,
    )

    _assert_plan_valid(valid)
    assert path.shape == (8, franka.n_qs)
    assert len(calls) == 2


@pytest.mark.required
@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("envs_idx_kind", ["tensor_bool", "list_bool", "numpy_bool"])
def test_plan_path_normalizes_bool_mask_envs_to_int_indices(backend, envs_idx_kind, monkeypatch):
    """Wrapper-level: bool mask `envs_idx` is normalized to int indices before reaching the
    planner. Restoration semantics for partial envs_idx are covered by the direct-planner tests."""
    from genesis.utils.path_planning import RRTConnect

    scene = gs.Scene(show_viewer=False)
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35))
    )
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)

    envs_mask = {
        "tensor_bool": torch.tensor([False, True], dtype=torch.bool, device=gs.device),
        "list_bool": [False, True],
        "numpy_bool": np.array([False, True], dtype=np.bool_),
    }[envs_idx_kind]
    expected_envs_idx = torch.tensor([1], dtype=gs.tc_int, device=gs.device)
    seen_envs_idx = []

    def fake_plan(self, *args, **kwargs):
        seen_envs_idx.append(kwargs["envs_idx"])
        path = torch.zeros((kwargs["num_waypoints"], 1, self._entity.n_qs), dtype=gs.tc_float, device=gs.device)
        return path, torch.tensor([False], dtype=torch.bool, device=gs.device)

    monkeypatch.setattr(RRTConnect, "plan", fake_plan)

    path, valid = franka.plan_path(
        qpos_goal=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        max_retry=0,
        smooth_path=False,
        ignore_collision=True,
        envs_idx=envs_mask,
        planner="RRTConnect",
        ee_link_name="hand",
        with_entity=cube,
        return_valid_mask=True,
    )

    _assert_plan_valid(valid)
    assert path.shape == (8, 1, franka.n_qs)
    assert torch.equal(seen_envs_idx[0], expected_envs_idx)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_snapshot_constraint_state_excludes_rebuilt_contact_island_when_not_hibernating(backend):
    from genesis.utils.path_planning import RRTConnect

    scene = gs.Scene(show_viewer=False)
    anchor, _ = _add_restore_test_entities(scene)
    planner = RRTConnect(anchor)
    planner._solver._use_contact_island = True
    constraint_solver = scene.rigid_solver.constraint_solver
    del constraint_solver._eq_const_info_cache
    contact_island_state = constraint_solver.contact_island.contact_island_state

    tensor_state, _ = planner.snapshot_constraint_state()
    snapshot_ids = {id(tensor) for tensor, _ in tensor_state}
    contact_island_ids = {id(tensor) for tensor in planner.iter_state_tensors(contact_island_state)}
    expected_ids = {
        id(constraint_solver.qacc_ws),
        id(constraint_solver.constraint_state.is_warmstart),
        id(constraint_solver.constraint_state.n_constraints),
        id(constraint_solver.constraint_state.n_constraints_equality),
        id(constraint_solver.constraint_state.n_constraints_frictionloss),
        id(constraint_solver.constraint_state.qd_n_equalities),
    }

    assert snapshot_ids == expected_ids
    assert snapshot_ids.isdisjoint(contact_island_ids)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_snapshot_constraint_state_restores_contact_island_runtime_values(backend):
    from genesis.utils.path_planning import RRTConnect

    scene = gs.Scene(show_viewer=False)
    anchor, _ = _add_restore_test_entities(scene)
    planner = RRTConnect(anchor)
    planner._solver._use_contact_island = True
    constraint_solver = scene.rigid_solver.constraint_solver

    tensors = [
        constraint_solver.qacc_ws,
        constraint_solver.constraint_state.is_warmstart,
        constraint_solver.constraint_state.n_constraints,
        constraint_solver.constraint_state.n_constraints_equality,
        constraint_solver.constraint_state.n_constraints_frictionloss,
        constraint_solver.constraint_state.qd_n_equalities,
    ]
    for tensor in tensors:
        qd_to_torch(tensor, copy=False).fill_(1)

    constraint_snapshot = planner.snapshot_constraint_state()
    for tensor, _ in constraint_snapshot[0]:
        qd_to_torch(tensor, copy=False).fill_(0)

    planner.restore_constraint_state(constraint_snapshot)

    _assert_qd_state_matches(constraint_snapshot[0])


@pytest.mark.required
@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_without_entity_uses_qpos_restore_without_runtime_snapshots(backend, planner, monkeypatch, tol):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = _make_planner_scene()
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    planner_cls = {"RRT": RRT, "RRTConnect": RRTConnect}[planner]
    if planner == "RRT":
        _force_rrt_goal_sampling(monkeypatch)

    qpos_before = franka.get_qpos().clone()

    def fail_runtime_snapshot(*args, **kwargs):
        raise AssertionError("no-object plan_path should not snapshot solver runtime state")

    def dirty_get_exclude_geom_pairs(self, *args, **kwargs):
        self._entity.set_qpos(qpos_before + 0.01, zero_velocity=False)
        return torch.empty((0, 2), dtype=gs.tc_int, device=gs.device)

    for helper_name in (
        "snapshot_hibernation_state",
        "snapshot_collider_state",
        "snapshot_constraint_state",
    ):
        monkeypatch.setattr(planner_cls, helper_name, fail_runtime_snapshot)
    monkeypatch.setattr(planner_cls, "get_exclude_geom_pairs", dirty_get_exclude_geom_pairs)

    path, valid = franka.plan_path(
        qpos_goal=START_QPOS,
        qpos_start=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        max_retry=0,
        smooth_path=False,
        ignore_collision=True,
        planner=planner,
        return_valid_mask=True,
    )

    _assert_plan_valid(valid)
    assert path.shape == (8, franka.n_qs)
    assert_allclose(franka.get_qpos(), qpos_before, tol=tol)


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_without_entity_uses_qpos_restore_without_runtime_snapshots(backend, planner, monkeypatch, tol):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = _make_planner_scene()
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    planner_cls = {"RRT": RRT, "RRTConnect": RRTConnect}[planner]
    if planner == "RRT":
        _force_rrt_goal_sampling(monkeypatch)

    planner_obj = planner_cls(franka)
    qpos_before = franka.get_qpos().clone()

    def fail_runtime_snapshot(*args, **kwargs):
        raise AssertionError("no-object direct planner should not snapshot solver runtime state")

    def dirty_get_exclude_geom_pairs(self, *args, **kwargs):
        self._entity.set_qpos(qpos_before + 0.01, zero_velocity=False)
        return torch.empty((0, 2), dtype=gs.tc_int, device=gs.device)

    for helper_name in (
        "snapshot_hibernation_state",
        "snapshot_collider_state",
        "snapshot_constraint_state",
    ):
        monkeypatch.setattr(planner_cls, helper_name, fail_runtime_snapshot)
    monkeypatch.setattr(planner_cls, "get_exclude_geom_pairs", dirty_get_exclude_geom_pairs)

    path, is_invalid = planner_obj.plan(
        qpos_goal=START_QPOS,
        qpos_start=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        smooth_path=False,
        ignore_collision=True,
    )

    _assert_plan_valid(~is_invalid)
    assert path.shape[-1] == franka.n_qs
    assert_allclose(franka.get_qpos(), qpos_before, tol=tol)


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_with_entity_restores_full_runtime_for_selected_env_query_with_grad(
    backend, planner, monkeypatch
):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = _make_planner_scene(requires_grad=True)
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)

    planner_cls = {"RRT": RRT, "RRTConnect": RRTConnect}[planner]
    if planner == "RRT":
        _force_rrt_goal_sampling(monkeypatch)

    planner_obj = planner_cls(franka)
    envs_idx = torch.tensor([1], dtype=gs.tc_int, device=gs.device)
    other_env_idx = torch.tensor([0], dtype=gs.tc_int, device=gs.device)

    collider_state = scene.rigid_solver.collider._collider_state
    contact_normal = qd_to_torch(collider_state.contact_data.normal, copy=False)
    cache_normal = qd_to_torch(collider_state.contact_cache.normal, copy=False)
    diff_input = collider_state.diff_contact_input
    diff_ref_id = qd_to_torch(diff_input.ref_id, copy=False)
    diff_valid = qd_to_torch(diff_input.valid, copy=False)
    diff_ref_penetration = qd_to_torch(diff_input.ref_penetration, copy=False)
    constraint_state = scene.rigid_solver.constraint_solver.constraint_state
    is_warmstart = qd_to_torch(constraint_state.is_warmstart, copy=False)
    qacc_ws = qd_to_torch(constraint_state.qacc_ws, copy=False)
    errno = qd_to_torch(scene.rigid_solver._errno, copy=False)

    contact_normal[:, other_env_idx] = 1.25
    contact_normal[:, envs_idx] = 3.25
    cache_normal[:, other_env_idx] = 1.25
    cache_normal[:, envs_idx] = 3.25
    diff_ref_id[other_env_idx] = 1
    diff_ref_id[envs_idx] = 2
    diff_valid[other_env_idx] = False
    diff_valid[envs_idx] = True
    diff_ref_penetration[other_env_idx] = 1.25
    diff_ref_penetration[envs_idx] = 3.25
    is_warmstart.copy_(torch.tensor([True, False], dtype=torch.bool, device=is_warmstart.device))
    qacc_ws[:, other_env_idx] = 1.25
    qacc_ws[:, envs_idx] = 3.25
    errno.copy_(torch.tensor([3, 4], dtype=errno.dtype, device=errno.device))

    expected = [
        (contact_normal, contact_normal.clone()),
        (cache_normal, cache_normal.clone()),
        (diff_ref_id, diff_ref_id.clone()),
        (diff_valid, diff_valid.clone()),
        (diff_ref_penetration, diff_ref_penetration.clone()),
        (is_warmstart, is_warmstart.clone()),
        (qacc_ws, qacc_ws.clone()),
        (errno, errno.clone()),
    ]

    def dirty_get_exclude_geom_pairs(self, qposs, actual_envs_idx):
        assert torch.equal(actual_envs_idx, envs_idx)
        contact_normal[:, other_env_idx] = 8.5
        contact_normal[:, envs_idx] = 9.5
        cache_normal[:, other_env_idx] = 8.5
        cache_normal[:, envs_idx] = 9.5
        diff_ref_id[other_env_idx] = 8
        diff_ref_id[envs_idx] = 9
        diff_valid[other_env_idx] = True
        diff_valid[envs_idx] = False
        diff_ref_penetration[other_env_idx] = 8.5
        diff_ref_penetration[envs_idx] = 9.5
        is_warmstart.copy_(torch.tensor([False, True], dtype=torch.bool, device=is_warmstart.device))
        qacc_ws[:, other_env_idx] = 8.5
        qacc_ws[:, envs_idx] = 9.5
        errno.copy_(torch.tensor([8, 9], dtype=errno.dtype, device=errno.device))
        return torch.empty((0, 2), dtype=gs.tc_int, device=gs.device)

    monkeypatch.setattr(planner_cls, "get_exclude_geom_pairs", dirty_get_exclude_geom_pairs)

    path, is_invalid = planner_obj.plan(
        qpos_goal=START_QPOS,
        qpos_start=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        smooth_path=False,
        ignore_collision=True,
        envs_idx=envs_idx,
        ee_link_idx=franka.get_link("hand").idx,
        obj_entity=cube,
    )

    _assert_plan_valid(~is_invalid)
    assert path.shape == (8, 1, franka.n_qs)
    for tensor, expected_value in expected:
        assert torch.equal(tensor, expected_value)


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_with_entity_restores_object_state(backend, planner, monkeypatch, tol):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = _make_planner_scene()
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    planner_cls = {"RRT": RRT, "RRTConnect": RRTConnect}[planner]
    if planner == "RRT":
        _force_rrt_goal_sampling(monkeypatch)

    cube_pos_before = cube.get_pos().clone()
    cube_quat_before = cube.get_quat().clone()
    cube_vel_before = cube.get_dofs_velocity().clone()

    def dirty_get_exclude_geom_pairs(self, *args, **kwargs):
        cube.set_pos(cube_pos_before + torch.tensor([0.01, 0.0, 0.0], device=gs.device))
        cube.set_quat(torch.tensor([0.9238795, 0.0, 0.3826834, 0.0], dtype=gs.tc_float, device=gs.device))
        cube.set_dofs_velocity(torch.ones_like(cube_vel_before))
        return torch.empty((0, 2), dtype=gs.tc_int, device=gs.device)

    monkeypatch.setattr(planner_cls, "get_exclude_geom_pairs", dirty_get_exclude_geom_pairs)

    planner_obj = planner_cls(franka)
    path, is_invalid = planner_obj.plan(
        qpos_goal=START_QPOS,
        qpos_start=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        smooth_path=False,
        ignore_collision=True,
        ee_link_idx=franka.get_link("hand").idx,
        obj_entity=cube,
    )

    _assert_plan_valid(~is_invalid)
    assert path.shape[-1] == franka.n_qs
    assert_allclose(cube.get_pos(), cube_pos_before, tol=tol)
    assert_allclose(cube.get_quat(), cube_quat_before, tol=tol)
    assert_allclose(cube.get_dofs_velocity(), cube_vel_before, tol=tol)


@pytest.mark.required
@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_restores_state_when_planner_raises(backend, planner, monkeypatch, tol):
    """When the planner raises mid-loop, `_planning_transaction` must restore live state and
    re-raise the original exception. Regression test for the bug where `sys.exc_info()` in the
    `finally` could mistake an outer exception for an in-flight planner exception and silently
    swallow restore failures."""
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = _make_planner_scene()
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    planner_cls = {"RRT": RRT, "RRTConnect": RRTConnect}[planner]
    cube_pos_before = cube.get_pos().clone()
    robot_qpos_before = franka.get_qpos().clone()

    class _Boom(RuntimeError):
        pass

    def explode_after_mutation(self, *args, **kwargs):
        cube.set_pos(cube_pos_before + torch.tensor([0.01, 0.0, 0.0], device=gs.device))
        self._entity.set_qpos(robot_qpos_before + 0.05, zero_velocity=False)
        raise _Boom("planner blew up mid-loop")

    monkeypatch.setattr(planner_cls, "get_exclude_geom_pairs", explode_after_mutation)

    planner_obj = planner_cls(franka)
    with pytest.raises(_Boom):
        planner_obj.plan(
            qpos_goal=START_QPOS,
            qpos_start=START_QPOS,
            max_nodes=8,
            num_waypoints=8,
            smooth_path=False,
            ignore_collision=True,
            ee_link_idx=franka.get_link("hand").idx,
            obj_entity=cube,
        )

    assert_allclose(franka.get_qpos(), robot_qpos_before, tol=tol)
    assert_allclose(cube.get_pos(), cube_pos_before, tol=tol)


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_does_not_swallow_restore_failure_inside_caller_except_block(
    backend, planner, monkeypatch, tol
):
    """Regression test for the `sys.exc_info()` bug: calling `plan()` from inside the caller's
    own `except` block must NOT make a restore failure look like an in-flight planner exception.
    A restore failure on normal return must propagate, even when an outer exception is active."""
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = _make_planner_scene()
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    planner_cls = {"RRT": RRT, "RRTConnect": RRTConnect}[planner]
    if planner == "RRT":
        _force_rrt_goal_sampling(monkeypatch)

    def fail_restore_entity_state(self, *args, **kwargs):
        raise RuntimeError("restore_entity_state failed")

    monkeypatch.setattr(planner_cls, "restore_entity_state", fail_restore_entity_state)
    planner_obj = planner_cls(franka)

    # Simulate a caller wrapping plan() in its own except block. The active outer exception must
    # not cause `_planning_transaction` to demote the restore failure to a warning.
    try:
        raise ValueError("outer error from caller")
    except ValueError:
        with pytest.raises(RuntimeError, match="restore_entity_state failed"):
            planner_obj.plan(
                qpos_goal=START_QPOS,
                qpos_start=START_QPOS,
                max_nodes=8,
                num_waypoints=8,
                smooth_path=False,
                ignore_collision=True,
                ee_link_idx=franka.get_link("hand").idx,
                obj_entity=cube,
            )


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_baseexception_in_restore_wins_over_exception(backend, monkeypatch):
    """When NO planner exception is in flight but two restore steps fail — a regular Exception
    AND a BaseException-non-Exception (e.g. KeyboardInterrupt) — the BaseException must win and
    propagate, regardless of which fired first."""
    from genesis.utils.path_planning import RRT

    scene = _make_planner_scene()
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    def fail_object_restore(self, *args, **kwargs):
        raise RuntimeError("object restore failed first")

    def signal_during_collider_restore(self, *args, **kwargs):
        raise KeyboardInterrupt("simulated signal after object restore failure")

    monkeypatch.setattr(RRT, "restore_entity_state", fail_object_restore)
    monkeypatch.setattr(RRT, "restore_collider_state", signal_during_collider_restore)

    planner_obj = RRT(franka)
    with pytest.raises(KeyboardInterrupt):
        planner_obj.plan(
            qpos_goal=START_QPOS,
            qpos_start=START_QPOS,
            max_nodes=8,
            num_waypoints=8,
            smooth_path=False,
            ignore_collision=True,
            ee_link_idx=franka.get_link("hand").idx,
            obj_entity=cube,
        )


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_does_not_drop_planner_error_on_baseexception_in_restore(backend, monkeypatch):
    """If a restore step raises `BaseException`-non-`Exception` (e.g. KeyboardInterrupt) AFTER the
    planner already raised, the in-flight planner exception must not be silently replaced — the
    planner exception is the diagnostic info the user needs."""
    from genesis.utils.path_planning import RRT

    scene = _make_planner_scene()
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    class _PlannerBoom(RuntimeError):
        pass

    def explode_after_mutation(self, *args, **kwargs):
        raise _PlannerBoom("planner failed")

    def restore_via_signal(self, *args, **kwargs):
        raise KeyboardInterrupt("simulated signal during restore")

    monkeypatch.setattr(RRT, "get_exclude_geom_pairs", explode_after_mutation)
    monkeypatch.setattr(RRT, "restore_entity_state", restore_via_signal)

    planner_obj = RRT(franka)
    with pytest.raises(KeyboardInterrupt):
        planner_obj.plan(
            qpos_goal=START_QPOS,
            qpos_start=START_QPOS,
            max_nodes=8,
            num_waypoints=8,
            smooth_path=False,
            ignore_collision=True,
            ee_link_idx=franka.get_link("hand").idx,
            obj_entity=cube,
        )


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_restores_eq_const_info_cache_end_to_end(backend, monkeypatch, tol):
    """End-to-end CPU coverage of `_eq_const_info_cache` save/clear/restore through `plan()`."""
    from genesis.utils.path_planning import RRTConnect

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=False,
            use_hibernation=False,
        ),
        show_viewer=False,
    )
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    constraint_solver = scene.rigid_solver.constraint_solver
    cache_key = ("eq_const_info_cache_round_trip", False)
    cache_value = object()
    constraint_solver._eq_const_info_cache[cache_key] = cache_value

    def dirty_cache(self, *args, **kwargs):
        constraint_solver._eq_const_info_cache.clear()
        constraint_solver._eq_const_info_cache[("dirty",)] = "stale"
        return torch.empty((0, 2), dtype=gs.tc_int, device=gs.device)

    monkeypatch.setattr(RRTConnect, "get_exclude_geom_pairs", dirty_cache)

    planner_obj = RRTConnect(franka)
    planner_obj.plan(
        qpos_goal=START_QPOS,
        qpos_start=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        smooth_path=False,
        ignore_collision=True,
        ee_link_idx=franka.get_link("hand").idx,
        obj_entity=cube,
    )

    assert constraint_solver._eq_const_info_cache.get(cache_key) is cache_value
    assert ("dirty",) not in constraint_solver._eq_const_info_cache


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_get_exclude_geom_pairs_filters_contacts_to_selected_env(backend, monkeypatch):
    from genesis.utils.path_planning import RRTConnect

    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)
    planner = RRTConnect(franka)
    envs_idx = torch.tensor([1], dtype=gs.tc_int, device=gs.device)

    monkeypatch.setattr(franka, "set_qpos", lambda *args, **kwargs: None)
    monkeypatch.setattr(scene.rigid_solver, "_kernel_detect_collision", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        franka,
        "get_contacts",
        lambda: {
            "geom_a": torch.tensor([[100, 101], [200, 201]], dtype=gs.tc_int, device=gs.device),
            "geom_b": torch.tensor([[110, 111], [210, 211]], dtype=gs.tc_int, device=gs.device),
            "valid_mask": torch.tensor([[True, False], [False, True]], dtype=torch.bool, device=gs.device),
        },
    )

    pairs = planner.get_exclude_geom_pairs([torch.zeros((1, franka.n_qs), dtype=gs.tc_float, device=gs.device)], envs_idx)

    assert torch.equal(pairs, torch.tensor([[201, 211]], dtype=gs.tc_int, device=gs.device))


@pytest.mark.required
@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("envs_idx_kind", ["empty_int", "false_bool", "false_bool_list", "false_bool_numpy"])
def test_plan_path_empty_env_mask_returns_empty(backend, envs_idx_kind):
    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)
    envs_idx = {
        "empty_int": torch.empty((0,), dtype=gs.tc_int, device=gs.device),
        "false_bool": torch.tensor([False, False], dtype=torch.bool, device=gs.device),
        "false_bool_list": [False, False],
        "false_bool_numpy": np.array([False, False], dtype=np.bool_),
    }[envs_idx_kind]

    path, valid = franka.plan_path(
        qpos_goal=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        max_retry=0,
        smooth_path=False,
        ignore_collision=True,
        envs_idx=envs_idx,
        return_valid_mask=True,
    )

    assert path.shape == (8, 0, franka.n_qs)
    assert valid.shape == (0,)
    assert valid.dtype == torch.bool


@pytest.mark.required
@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_empty_env_mask_still_requires_qpos_goal(backend):
    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)

    with pytest.raises(gs.GenesisException, match="qpos_goal"):
        franka.plan_path(
            qpos_goal=None,
            max_nodes=8,
            num_waypoints=8,
            max_retry=0,
            smooth_path=False,
            ignore_collision=True,
            envs_idx=torch.empty((0,), dtype=gs.tc_int, device=gs.device),
        )


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
@pytest.mark.parametrize("envs_idx_kind", ["empty_int", "false_bool", "false_bool_list", "false_bool_numpy"])
def test_direct_planner_empty_envs_returns_empty(backend, planner, envs_idx_kind):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)
    planner_obj = {"RRT": RRT, "RRTConnect": RRTConnect}[planner](franka)
    envs_idx = {
        "empty_int": torch.empty((0,), dtype=gs.tc_int, device=gs.device),
        "false_bool": torch.tensor([False, False], dtype=torch.bool, device=gs.device),
        "false_bool_list": [False, False],
        "false_bool_numpy": np.array([False, False], dtype=np.bool_),
    }[envs_idx_kind]

    path, is_invalid = planner_obj.plan(
        qpos_goal=START_QPOS,
        max_nodes=8,
        num_waypoints=8,
        smooth_path=False,
        ignore_collision=True,
        envs_idx=envs_idx,
    )

    assert path.shape == (8, 0, franka.n_qs)
    assert is_invalid.shape == (0,)
    assert is_invalid.dtype == torch.bool


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_empty_envs_still_requires_qpos_goal(backend, planner):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)
    planner_obj = {"RRT": RRT, "RRTConnect": RRTConnect}[planner](franka)

    with pytest.raises(gs.GenesisException, match="qpos_goal"):
        planner_obj.plan(
            qpos_goal=None,
            max_nodes=8,
            num_waypoints=8,
            smooth_path=False,
            ignore_collision=True,
            envs_idx=torch.empty((0,), dtype=gs.tc_int, device=gs.device),
        )


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_rejects_partial_object_attachment_args(backend, planner):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = gs.Scene(show_viewer=False)
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()
    planner_obj = {"RRT": RRT, "RRTConnect": RRTConnect}[planner](franka)

    with pytest.raises(gs.GenesisException, match="must be specified together"):
        planner_obj.plan(qpos_goal=START_QPOS, obj_entity=cube)

    with pytest.raises(gs.GenesisException, match="must be specified together"):
        planner_obj.plan(qpos_goal=START_QPOS, ee_link_idx=franka.get_link("hand").idx)


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_rejects_articulated_object_attachment(backend, planner):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = gs.Scene(show_viewer=False)
    articulated_object = scene.add_entity(gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf"))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()
    planner_obj = {"RRT": RRT, "RRTConnect": RRTConnect}[planner](franka)

    with pytest.raises(gs.GenesisException, match="non-articulated"):
        planner_obj.plan(
            qpos_goal=START_QPOS,
            ee_link_idx=franka.get_link("hand").idx,
            obj_entity=articulated_object,
        )


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_rejects_object_from_different_scene(backend, planner, monkeypatch):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = gs.Scene(show_viewer=False)
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    other_scene = gs.Scene(show_viewer=False)
    other_cube = other_scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    other_scene.build()

    planner_cls = {"RRT": RRT, "RRTConnect": RRTConnect}[planner]
    monkeypatch.setattr(
        planner_cls,
        "get_exclude_geom_pairs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validation must run before planning")),
    )

    planner_obj = planner_cls(franka)
    with pytest.raises(gs.GenesisException, match="same scene"):
        planner_obj.plan(
            qpos_goal=START_QPOS,
            ee_link_idx=franka.get_link("hand").idx,
            obj_entity=other_cube,
            max_nodes=8,
            num_waypoints=8,
            smooth_path=False,
            ignore_collision=True,
        )

    assert cube._solver is scene.rigid_solver


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_rejects_ee_link_from_other_entity(backend, planner, monkeypatch):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = gs.Scene(show_viewer=False)
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    wrong_link_source = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.10, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    planner_cls = {"RRT": RRT, "RRTConnect": RRTConnect}[planner]
    monkeypatch.setattr(
        planner_cls,
        "get_exclude_geom_pairs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validation must run before planning")),
    )

    planner_obj = planner_cls(franka)
    with pytest.raises(gs.GenesisException, match="planning entity"):
        planner_obj.plan(
            qpos_goal=START_QPOS,
            ee_link_idx=wrong_link_source.base_link.idx,
            obj_entity=cube,
            max_nodes=8,
            num_waypoints=8,
            smooth_path=False,
            ignore_collision=True,
        )


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_rejects_partial_object_attachment_args(backend):
    scene = gs.Scene(show_viewer=False)
    cube = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    with pytest.raises(gs.GenesisException, match="must be specified together"):
        franka.plan_path(qpos_goal=START_QPOS, ee_link_name="hand")

    with pytest.raises(gs.GenesisException, match="must be specified together"):
        franka.plan_path(qpos_goal=START_QPOS, with_entity=cube)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_rejects_with_entity_self(backend):
    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    with pytest.raises(gs.GenesisException, match="cannot be the planning entity itself"):
        franka.plan_path(qpos_goal=START_QPOS, ee_link_name="hand", with_entity=franka)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_rejects_negative_max_retry(backend):
    """`max_retry < 0` would skip the planner loop entirely and return an uninitialized path."""
    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    with pytest.raises(gs.GenesisException, match="max_retry"):
        franka.plan_path(qpos_goal=START_QPOS, max_retry=-1)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_rejects_non_positive_num_waypoints(backend):
    """`num_waypoints < 1` produces a zero-size output and crashes inside `align_waypoints_length`
    when the planner succeeds; reject up front."""
    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    with pytest.raises(gs.GenesisException, match="num_waypoints"):
        franka.plan_path(qpos_goal=START_QPOS, num_waypoints=0)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_sanitize_envs_idx_rejects_mixed_bool_int_list(backend):
    """A list mixing bool and int is ambiguous: a strict-int call would coerce True→1, silently
    changing user intent. Reject explicitly instead."""
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)

    with pytest.raises(gs.GenesisException, match="Mixing bool and int"):
        scene._sanitize_envs_idx([True, 0])


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_rejects_obj_entity_is_self(backend, planner):
    """Direct-planner counterpart of the wrapper's `with_entity is self` check."""
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    planner_cls = {"RRT": RRT, "RRTConnect": RRTConnect}[planner]
    planner_obj = planner_cls(franka)
    with pytest.raises(gs.GenesisException, match="cannot be the planning entity itself"):
        planner_obj.plan(
            qpos_goal=START_QPOS,
            ee_link_idx=franka.get_link("hand").idx,
            obj_entity=franka,
        )


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_rejects_articulated_object_attachment_before_snapshot(backend, monkeypatch):
    from genesis.utils.path_planning import RRTConnect

    scene = gs.Scene(show_viewer=False)
    articulated_object = scene.add_entity(gs.morphs.URDF(file="urdf/simple/two_link_arm.urdf"))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    monkeypatch.setattr(
        RRTConnect,
        "snapshot_entity_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validation must run before snapshot")),
    )

    with pytest.raises(gs.GenesisException, match="non-articulated"):
        franka.plan_path(
            qpos_goal=START_QPOS,
            ee_link_name="hand",
            with_entity=articulated_object,
            max_nodes=8,
            num_waypoints=8,
            max_retry=0,
            smooth_path=False,
            ignore_collision=True,
            planner="RRTConnect",
        )


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_rejects_object_from_different_scene_before_snapshot(backend, monkeypatch):
    from genesis.utils.path_planning import RRTConnect

    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    other_scene = gs.Scene(show_viewer=False)
    other_cube = other_scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.30, 0.10, 0.35)))
    other_scene.build()

    monkeypatch.setattr(
        RRTConnect,
        "snapshot_entity_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validation must run before snapshot")),
    )

    with pytest.raises(gs.GenesisException, match="same scene"):
        franka.plan_path(
            qpos_goal=START_QPOS,
            ee_link_name="hand",
            with_entity=other_cube,
            max_nodes=8,
            num_waypoints=8,
            max_retry=0,
            smooth_path=False,
            ignore_collision=True,
            planner="RRTConnect",
        )


@pytest.mark.cache(False)
@pytest.mark.parametrize("envs_idx_kind", ["empty_int", "one_int"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_plan_path_rejects_envs_idx_for_non_batched_scene(backend, envs_idx_kind):
    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()
    envs_idx = (
        torch.empty((0,), dtype=gs.tc_int, device=gs.device)
        if envs_idx_kind == "empty_int"
        else torch.tensor([0], dtype=gs.tc_int, device=gs.device)
    )

    with pytest.raises(gs.GenesisException, match="envs_idx"):
        franka.plan_path(qpos_goal=START_QPOS, envs_idx=envs_idx)


@pytest.mark.cache(False)
@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
@pytest.mark.parametrize("envs_idx_kind", ["empty_int", "one_int"])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_direct_planner_rejects_envs_idx_for_non_batched_scene(backend, planner, envs_idx_kind):
    from genesis.utils.path_planning import RRT, RRTConnect

    scene = gs.Scene(show_viewer=False)
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()
    planner_obj = {"RRT": RRT, "RRTConnect": RRTConnect}[planner](franka)
    envs_idx = (
        torch.empty((0,), dtype=gs.tc_int, device=gs.device)
        if envs_idx_kind == "empty_int"
        else torch.tensor([0], dtype=gs.tc_int, device=gs.device)
    )

    with pytest.raises(gs.GenesisException, match="envs_idx"):
        planner_obj.plan(qpos_goal=START_QPOS, envs_idx=envs_idx)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_restore_collider_state_clears_contact_cache(backend):
    from genesis.utils.path_planning import RRT

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    anchor = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.0, 0.05),
            fixed=True,
        )
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.30, 0.10, 0.02),
        )
    )
    scene.build()

    scene.rigid_solver.collider.clear()
    scene.rigid_solver.collider.detection()
    scene.rigid_solver.collider.get_contacts()
    assert scene.rigid_solver.collider._contact_data_cache

    planner = RRT(anchor)
    collider_state = planner.snapshot_collider_state()

    planner.restore_collider_state(collider_state)

    assert scene.rigid_solver.collider._contact_data_cache == {}


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_snapshot_collider_state_excludes_rebuilt_collision_scratch_buffers(backend):
    from genesis.utils.path_planning import RRT

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            use_contact_island=False,
            use_hibernation=False,
        ),
        show_viewer=False,
    )
    anchor, _ = _add_restore_test_entities(scene)
    planner = RRT(anchor)
    collider_state = scene.rigid_solver.collider._collider_state
    snapshot = planner.snapshot_collider_state()
    snapshot_ids = {id(tensor) for tensor, _ in snapshot}
    excluded_ids = {
        id(collider_state.n_broad_pairs),
        id(collider_state.broad_collision_pairs),
        id(collider_state.active_buffer),
        id(collider_state.box_depth),
        id(collider_state.box_points),
        id(collider_state.box_pts),
        id(collider_state.box_lines),
        id(collider_state.box_linesu),
        id(collider_state.box_axi),
        id(collider_state.box_ppts2),
        id(collider_state.box_pu),
        id(collider_state.xyz_max_min),
        id(collider_state.prism),
    }
    excluded_ids.update(id(tensor) for tensor in planner.iter_state_tensors(collider_state.sort_buffer))

    assert snapshot_ids.isdisjoint(excluded_ids)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_snapshot_collider_state_excludes_reinitialized_collision_scratch(backend):
    from genesis.utils.path_planning import RRT

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            use_contact_island=False,
            use_hibernation=False,
        ),
        show_viewer=False,
    )
    anchor, _ = _add_restore_test_entities(scene)
    planner = RRT(anchor)
    collider_state = scene.rigid_solver.collider._collider_state
    snapshot = planner.snapshot_collider_state()
    snapshot_ids = {id(tensor) for tensor, _ in snapshot}
    excluded_ids = {
        id(collider_state.contact_sort_key),
        id(collider_state.contact_sort_idx),
    }
    excluded_ids.update(id(tensor) for tensor in planner.iter_state_tensors(collider_state.narrowphase_work_queues))

    assert snapshot_ids.isdisjoint(excluded_ids)


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_snapshot_collider_state_restores_grad_contact_input(backend):
    from genesis.utils.path_planning import RRT

    scene = _make_planner_scene(requires_grad=True)
    anchor, _ = _add_restore_test_entities(scene)
    planner = RRT(anchor)
    collider_state = scene.rigid_solver.collider._collider_state

    snapshot = planner.snapshot_collider_state()
    snapshot_ids = {id(tensor) for tensor, _ in snapshot}
    expected_ids = _base_collider_snapshot_ids(planner, collider_state)
    expected_ids.update(id(tensor) for tensor in planner.iter_state_tensors(collider_state.diff_contact_input))

    assert snapshot_ids == expected_ids


@pytest.mark.cache(False)
@pytest.mark.parametrize("backend", [gs.cpu])
def test_internal_object_restore_allows_partial_fixed_entity_noop(backend, tol):
    from genesis.utils.path_planning import RRT

    scene = gs.Scene(show_viewer=False)
    anchor = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.0, 0.05),
            fixed=True,
        )
    )
    fixed_cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.30, 0.10, 0.35),
            fixed=True,
        )
    )
    scene.build(n_envs=2)
    planner = RRT(anchor)

    envs_idx = [1]
    state = planner.snapshot_entity_state(fixed_cube, envs_idx=envs_idx)
    pos_before = fixed_cube.get_pos().clone()
    quat_before = fixed_cube.get_quat().clone()

    planner.restore_entity_state(fixed_cube, state, envs_idx=envs_idx)

    assert_allclose(fixed_cube.get_pos(), pos_before, tol=tol)
    assert_allclose(fixed_cube.get_quat(), quat_before, tol=tol)


def _setup_dynamic_grasp_state():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0.01),
            size=(0.5, 0.8, 0.02),
            fixed=True,
        )
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.30, 0.10, 0.35),
        )
    )
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    for _ in range(10):
        scene.step()

    franka.set_dofs_position(START_QPOS)
    franka.control_dofs_position(START_QPOS)
    for _ in range(20):
        scene.step()

    fingers_dof = np.arange(7, 9)
    franka.control_dofs_position(np.array([0.0, 0.0], dtype=np.float32), fingers_dof)
    for _ in range(100):
        scene.step()

    for _ in range(20):
        scene.step()

    return scene, cube, franka


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
def test_plan_path_with_entity_restores_dynamic_grasp_state_smoke(backend, tol):
    from genesis.utils.path_planning import RRTConnect

    # Issue #2715 depends on a dynamic grasp state; this compact smoke keeps that state in the required gate.
    scene, cube, franka = _setup_dynamic_grasp_state()
    hand = franka.get_link("hand")
    qpos_goal = franka.inverse_kinematics(
        link=hand,
        pos=np.asarray([0.2999962, 0.1000018, 0.34340014], dtype=np.float32),
        quat=np.asarray([1.32832755e-05, 7.07106803e-01, 7.07106759e-01, 4.74470964e-06], dtype=np.float64),
    )
    qpos_goal[-2:] = 0.0

    planner = RRTConnect(franka)
    before_rigid_state = scene.get_state().solvers_state[scene.solvers.index(scene.rigid_solver)]
    cube_pos_before = cube.get_pos().clone()
    cube_quat_before = cube.get_quat().clone()
    cube_vel_before = cube.get_dofs_velocity().clone()
    collider_snapshot = planner.snapshot_collider_state()
    constraint_snapshot = planner.snapshot_constraint_state()

    _, valid = franka.plan_path(
        qpos_goal=qpos_goal,
        num_waypoints=16,
        max_nodes=16,
        max_retry=0,
        smooth_path=False,
        ignore_collision=True,
        ee_link_name="hand",
        with_entity=cube,
        return_valid_mask=True,
    )

    assert isinstance(valid, torch.Tensor)
    assert valid.dtype == torch.bool
    assert valid.shape == torch.Size([])
    _assert_plan_valid(valid)
    assert_allclose(franka.get_qpos(), before_rigid_state.qpos[:, franka.q_start : franka.q_end], tol=tol)
    assert_allclose(cube.get_pos(), cube_pos_before, tol=tol)
    assert_allclose(cube.get_quat(), cube_quat_before, tol=tol)
    assert_allclose(cube.get_dofs_velocity(), cube_vel_before, tol=tol)
    _assert_qd_state_matches(collider_snapshot)
    _assert_qd_state_matches(constraint_snapshot[0])
    assert scene.rigid_solver.constraint_solver._eq_const_info_cache == constraint_snapshot[1]


@pytest.mark.slow  # ~90s
@pytest.mark.parametrize("backend", [gs.gpu])
def test_plan_path_with_entity_restores_scene_state(backend, tol):
    # Issue #2715 requires a dynamic grasp state; a static cube planning query does not reproduce the leak.
    scene, cube, franka = _setup_dynamic_grasp_state()
    hand = franka.get_link("hand")
    qpos_goal = franka.inverse_kinematics(
        link=hand,
        pos=np.asarray([0.2999962, 0.1000018, 0.34340014], dtype=np.float32),
        quat=np.asarray([1.32832755e-05, 7.07106803e-01, 7.07106759e-01, 4.74470964e-06], dtype=np.float64),
    )
    qpos_goal[-2:] = 0.0

    before_rigid_state = scene.get_state().solvers_state[scene.solvers.index(scene.rigid_solver)]
    cube_pos_before = cube.get_pos().clone()
    cube_quat_before = cube.get_quat().clone()
    cube_vel_before = cube.get_dofs_velocity().clone()

    _, valid = franka.plan_path(
        qpos_goal=qpos_goal,
        num_waypoints=120,
        max_retry=0,
        smooth_path=True,
        ignore_collision=True,
        ee_link_name="hand",
        with_entity=cube,
        return_valid_mask=True,
    )

    _assert_plan_valid(valid)
    assert_allclose(franka.get_qpos(), before_rigid_state.qpos[:, franka.q_start : franka.q_end], tol=tol)
    assert_allclose(cube.get_pos(), cube_pos_before, tol=tol)
    assert_allclose(cube.get_quat(), cube_quat_before, tol=tol)
    assert_allclose(cube.get_dofs_velocity(), cube_vel_before, tol=tol)
