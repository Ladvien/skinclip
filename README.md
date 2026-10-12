# skinclip

Does a skinned character's animation pass its surface through itself? skinclip answers for glTF skins with numpy
alone, and repairs what it finds.

- **Check.** Pose a skinned figure with its clips, frame by frame, and measure where its triangles cross. The relation
  is metamorphic: a surface free of intersections at rest must stay free of them in every pose, measured as the length
  of the intersection contour (Grigorev et al., *ContourCraft*, SIGGRAPH 2024).
- **Repair limbs.** Turn the arms and legs of a humanoid clip off the trunk and off each other, by the smallest
  rotation that clears their capsules.
- **Cloth chains.** A ring skirt of bone chains: layout, skin weights and a baked, collision-aware animation for it.

## The rule: numpy only

skinclip imports numpy and nothing else outside the standard library: no scipy, no bpy. It runs in Blender 5.2's
Python 3.13 (numpy 2.3), in a Kimodo MCP venv (3.11, numpy 2.4) and in a plain `python3` (3.11, numpy 1.25). The code
uses no numpy-2-only API.

numpy is **not declared** as a dependency, on purpose: blended puts its venv's site-packages ahead of Blender's own,
so a declared numpy wheel would shadow Blender's bundled numpy. Install the extra where there is no numpy yet:

```
python3 -m pip install -e '.[numpy]'    # or '.[numpy,dev]' for pytest
python3 -m pytest -q
```

## API

| Module | What it holds |
|---|---|
| `skinclip.quat` | xyzw quaternions over leading axes: `multiply`, `conjugate`, `rotate`, `exp_map`, `arc`, `from_matrix`, `twist`, `slerp`, `quat_matrix`, `trs_matrix` |
| `skinclip.gltf` | `read_glb`, `accessor`, `read_clips`, `write_glb`, `write_rotations(source, out, animation, keys)` |
| `skinclip.model` | `Clip`, `Skeleton`, `Figure`, `read_figure(name, path, settings)`, `crease_zones`, `welded_corners` |
| `skinclip.pose` | `sample`, `node_worlds`, `joint_matrices`, `skin`, `clip_from_frames` |
| `skinclip.intersect` | `CheckSettings`, `pose_cut`, `check_clip`, `clip_cuts`, `ClipCheck`, `closest_points`, `segment_distance`, `contour_lengths`, `candidate_pairs` |
| `skinclip.humanoid` | `ROLES`, `UE5`, `MIXAMO`, `detect(names)` |
| `skinclip.capsules` | `Capsule`, `measure(figure, roles)`, `positions(figure, worlds, capsules)` |
| `skinclip.repair` | `repair(figure, roles, capsules, rot, root_translation, loop, half_spaces)`, `HalfSpace`, `RepairReport` |
| `skinclip.cloth` | `RingSkirtSpec`, `ring_skirt_layout`, `ring_skirt_weights`, `bake`, `chain_nodes` |
| `skinclip.testing` | `tube_humanoid(path, names, clip)`: a skinned tube humanoid with a hand-through-torso clip, for tests |

Every array is numpy; quaternions are glTF order (x, y, z, w); lengths are metres.

## License

MIT.
