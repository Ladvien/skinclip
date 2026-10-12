"""The humanoid rigs skinclip recognises, as a map from role to joint name.

A role is a place in the body (`upperarm_l`), not a name: the capsule, repair and cloth code speaks roles and a rig
convention supplies the names. The two conventions are the ones trellis2's retargeter maps (`UE5_TO_CANON`): the Unreal
mannequin's bones and Mixamo's `mixamorig:*`.
"""

from __future__ import annotations

from collections.abc import Iterable

CENTRAL = ("pelvis", "spine_01", "spine_02", "spine_03", "neck", "head")
SIDED = ("clavicle", "upperarm", "lowerarm", "hand", "thigh", "calf", "foot", "ball")
ROLES = CENTRAL + tuple(f"{role}_{side}" for role in SIDED for side in ("l", "r"))

UE5 = {
    "pelvis": "pelvis",
    "spine_01": "spine_01",
    "spine_02": "spine_02",
    "spine_03": "spine_03",
    "neck": "neck_01",
    "head": "Head",
    **{f"{role}_{side}": f"{role}_{side}" for role in SIDED for side in ("l", "r")},
}

_MIXAMO_SIDED = {
    "clavicle": "Shoulder",
    "upperarm": "Arm",
    "lowerarm": "ForeArm",
    "hand": "Hand",
    "thigh": "UpLeg",
    "calf": "Leg",
    "foot": "Foot",
    "ball": "ToeBase",
}
MIXAMO = {
    "pelvis": "mixamorig:Hips",
    "spine_01": "mixamorig:Spine",
    "spine_02": "mixamorig:Spine1",
    "spine_03": "mixamorig:Spine2",
    "neck": "mixamorig:Neck",
    "head": "mixamorig:Head",
    **{
        f"{role}_{side}": f"mixamorig:{'Left' if side == 'l' else 'Right'}{name}"
        for role, name in _MIXAMO_SIDED.items()
        for side in ("l", "r")
    },
}

CONVENTIONS = {"ue5": UE5, "mixamo": MIXAMO}


def detect(names: Iterable[str]) -> dict[str, str]:
    """The role -> joint name map of the first convention (UE5, then Mixamo) whose 22 joints are all in `names`."""
    present = set(names)
    closest: tuple[int, list[str]] | None = None
    for roles in (UE5, MIXAMO):
        missing = [name for name in roles.values() if name not in present]
        if not missing:
            return dict(roles)
        if closest is None or len(roles) - len(missing) > closest[0]:
            closest = (len(roles) - len(missing), missing)
    assert closest is not None
    raise ValueError(f"not a humanoid rig: missing {', '.join(closest[1][:6])}")


def convention(roles: dict[str, str]) -> str:
    """'ue5' or 'mixamo': which map `roles` is."""
    for name, known in CONVENTIONS.items():
        if roles == known:
            return name
    raise ValueError("roles match no known rig convention")
