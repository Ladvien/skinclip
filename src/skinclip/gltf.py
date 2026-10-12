"""Reading and writing the parts of a glTF 2 binary that skinning and animation need, with the stdlib and numpy only."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .model import Clip

_COMPONENT_BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NORMALISED = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
_WIDTH = _TYPE_COUNTS
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942


def read_glb(path: Path) -> tuple[dict, bytes]:
    """The JSON document and the BIN chunk of a glTF 2 binary."""
    data = path.read_bytes()
    magic, version, _total = struct.unpack("<4sII", data[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path} is not a glTF 2 binary")
    json_length, json_type = struct.unpack("<II", data[12:20])
    if json_type != _JSON_CHUNK:
        raise ValueError(f"{path}: first chunk is not JSON")
    document = json.loads(data[20 : 20 + json_length])
    offset = 20 + json_length
    bin_length, bin_type = struct.unpack("<II", data[offset : offset + 8])
    if bin_type != _BIN_CHUNK:
        raise ValueError(f"{path}: second chunk is not BIN")
    return document, data[offset + 8 : offset + 8 + bin_length]


def accessor_bytes(document: dict, blob_data: bytes, index: int) -> bytes:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    if "byteStride" in view or "sparse" in accessor:
        raise ValueError(f"accessor {index} is strided or sparse; this reader needs tightly packed data")
    size = accessor["count"] * _TYPE_COUNTS[accessor["type"]] * _COMPONENT_BYTES[accessor["componentType"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return blob_data[start : start + size]


def accessor(document: dict, data: bytes, index: int) -> np.ndarray:
    """An accessor as an array (count, components); normalised integers come back as floats in 0..1 (or -1..1)."""
    spec = document["accessors"][index]
    raw = np.frombuffer(accessor_bytes(document, data, index), dtype=_DTYPES[spec["componentType"]])
    values = raw.reshape(spec["count"], _WIDTH[spec["type"]])
    if spec.get("normalized"):
        return values.astype(float) / _NORMALISED[spec["componentType"]]
    return values


def read_clips(document: dict, data: bytes) -> dict[str, Clip]:
    """Every animation of the document as a `Clip` (node name -> path -> (times, values, interpolation))."""
    names = [node.get("name", "") for node in document["nodes"]]
    clips = {}
    for animation in document.get("animations", []):
        channels: dict[str, dict[str, tuple[np.ndarray, np.ndarray, str]]] = {}
        seconds = 0.0
        for channel in animation["channels"]:
            target = channel["target"]
            if "node" not in target or target["path"] not in ("translation", "rotation", "scale"):
                continue
            sampler = animation["samplers"][channel["sampler"]]
            times = accessor(document, data, sampler["input"])[:, 0].astype(float)
            values = accessor(document, data, sampler["output"]).astype(float)
            seconds = max(seconds, float(times[-1]))
            channels.setdefault(names[target["node"]], {})[target["path"]] = (
                times,
                values,
                sampler.get("interpolation", "LINEAR"),
            )
        clips[animation.get("name", f"animation_{len(clips)}")] = Clip(animation.get("name", ""), seconds, channels)
    return clips


def write_glb(document: dict, binary: bytes, path: Path) -> None:
    """Writes a glTF 2 binary: the JSON chunk padded with spaces and the BIN chunk with zeros, to 4 bytes."""
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), _JSON_CHUNK)
        + encoded
        + struct.pack("<II", len(binary), _BIN_CHUNK)
        + binary
    )


def write_rotations(source: Path, out: Path, animation: int, keys: dict[str, np.ndarray]) -> None:
    """Writes `source` to `out` with the rotation channel of each named node in animation `animation` replaced by
    `keys` (frames, 4), float32. A node without a rotation channel gets one, on the animation's first time accessor.

    The new keys are appended to the BIN chunk as new bufferViews and accessors (the replaced accessors stay as
    unreferenced bytes); every replaced channel gets its own LINEAR sampler."""
    document, data = read_glb(source)
    nodes = document["nodes"]
    index = {node.get("name", ""): i for i, node in enumerate(nodes)}
    target = document["animations"][animation]
    samplers, channels = target["samplers"], target["channels"]
    first_input = samplers[0]["input"]
    blob = bytearray(data)
    for name, values in keys.items():
        if name not in index:
            raise ValueError(f"{name}: no node of that name in {source.name}")
        existing = next(
            (c for c in channels if c["target"].get("node") == index[name] and c["target"]["path"] == "rotation"), None
        )
        time_accessor = samplers[existing["sampler"]]["input"] if existing is not None else first_input
        times = document["accessors"][time_accessor]["count"]
        values = np.asarray(values, "<f4").reshape(-1, 4)
        if len(values) != times:
            raise ValueError(f"{name}: {len(values)} keys but the animation has {times} times")
        blob += b"\0" * (-len(blob) % 4)
        offset = len(blob)
        blob += values.tobytes()
        document["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(values) * 16})
        document["accessors"].append(
            {"bufferView": len(document["bufferViews"]) - 1, "componentType": 5126, "count": len(values), "type": "VEC4"}
        )
        samplers.append({"input": time_accessor, "output": len(document["accessors"]) - 1, "interpolation": "LINEAR"})
        if existing is not None:
            existing["sampler"] = len(samplers) - 1
        else:
            channels.append({"sampler": len(samplers) - 1, "target": {"node": index[name], "path": "rotation"}})
    document["buffers"][0]["byteLength"] = len(blob)
    write_glb(document, bytes(blob), out)
