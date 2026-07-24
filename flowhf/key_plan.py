from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import struct
from statistics import NormalDist
from typing import Any, Iterator, Mapping

from flowhf.digest import stable_digest, stable_json_dumps


KEYED_PRG_VERSION = "sha256_counter_normal_icdf_table20_float32"
WRONG_KEY_COUNT = 32
WRONG_KEY_DOMAIN = "slm_wm_content_survival_observation_wrong_key_v1"
_INDEX_BITS = 20
_QUANTILE_COUNT = 1 << _INDEX_BITS
_COUNTER_BYTES = 16


@dataclass(frozen=True)
class KeyCandidate:
    role: str
    index: int | None
    material: str = field(repr=False)
    material_digest_random: str


@dataclass(frozen=True)
class KeyPlan:
    registered: KeyCandidate
    wrong: tuple[KeyCandidate, ...]
    plan_digest_random: str

    @property
    def ordered(self) -> tuple[KeyCandidate, ...]:
        return (self.registered, *self.wrong)

    def public_record(self) -> dict[str, Any]:
        return {
            "wrong_key_domain": WRONG_KEY_DOMAIN,
            "wrong_key_count": WRONG_KEY_COUNT,
            "registered_key_material_digest_random": (
                self.registered.material_digest_random
            ),
            "wrong_key_material_digests_random": [
                item.material_digest_random for item in self.wrong
            ],
            "key_plan_digest_random": self.plan_digest_random,
        }


def build_key_plan(registered_key_material: str) -> KeyPlan:
    registered = build_registered_key_candidate(registered_key_material)
    registered_digest = registered.material_digest_random
    wrong: list[KeyCandidate] = []
    for index in range(WRONG_KEY_COUNT):
        derivation = stable_digest(
            {
                "domain": WRONG_KEY_DOMAIN,
                "registered_key_material_digest_random": registered_digest,
                "wrong_key_index": index,
            }
        )
        material = f"slm-wm-observation-wrong-key:{derivation}"
        wrong.append(
            KeyCandidate(
                role="wrong",
                index=index,
                material=material,
                material_digest_random=stable_digest(
                    {"key_material": material}
                ),
            )
        )
    payload = {
        "wrong_key_domain": WRONG_KEY_DOMAIN,
        "registered_key_material_digest_random": registered_digest,
        "wrong_key_material_digests_random": [
            item.material_digest_random for item in wrong
        ],
    }
    return KeyPlan(
        registered=registered,
        wrong=tuple(wrong),
        plan_digest_random=stable_digest(payload),
    )


def build_registered_key_candidate(
    registered_key_material: str,
) -> KeyCandidate:
    if type(registered_key_material) is not str or not registered_key_material:
        raise ValueError("registered watermark key is required")
    registered_digest = stable_digest({"key_material": registered_key_material})
    return KeyCandidate(
        role="registered",
        index=None,
        material=registered_key_material,
        material_digest_random=registered_digest,
    )


def _domain_digest(
    shape: tuple[int, ...],
    key_material: str,
    domain_fields: Mapping[str, Any],
) -> bytes:
    payload = {
        "keyed_prg_version": KEYED_PRG_VERSION,
        "key_material": key_material,
        "domain_fields": dict(domain_fields),
        "shape": shape,
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).digest()


def _quantile_indices(
    element_count: int,
    domain: bytes,
) -> Iterator[int]:
    produced = 0
    counter = 0
    bit_buffer = 0
    available_bits = 0
    mask = (1 << _INDEX_BITS) - 1
    while produced < element_count:
        block = hashlib.sha256(
            domain + counter.to_bytes(_COUNTER_BYTES, "big")
        ).digest()
        counter += 1
        bit_buffer = (bit_buffer << 256) | int.from_bytes(block, "big")
        available_bits += 256
        while available_bits >= _INDEX_BITS and produced < element_count:
            available_bits -= _INDEX_BITS
            yield (bit_buffer >> available_bits) & mask
            bit_buffer &= (1 << available_bits) - 1 if available_bits else 0
            produced += 1


def _binary32_normal_midpoint(index: int) -> float:
    probability = (float(index) + 0.5) / float(_QUANTILE_COUNT)
    value = NormalDist().inv_cdf(probability)
    return struct.unpack(">f", struct.pack(">f", value))[0]


def build_keyed_gaussian_tensor(
    shape: tuple[int, ...],
    key_material: str,
    domain_fields: Mapping[str, Any],
) -> Any:
    import torch

    normalized_shape = tuple(int(value) for value in shape)
    if not normalized_shape or any(value <= 0 for value in normalized_shape):
        raise ValueError("keyed tensor shape must contain positive dimensions")
    if not key_material or not domain_fields:
        raise ValueError("key and domain fields are required")
    domain = _domain_digest(
        normalized_shape,
        key_material,
        domain_fields,
    )
    values = [
        _binary32_normal_midpoint(index)
        for index in _quantile_indices(math.prod(normalized_shape), domain)
    ]
    return torch.tensor(values, dtype=torch.float32, device="cpu").reshape(
        normalized_shape
    )
