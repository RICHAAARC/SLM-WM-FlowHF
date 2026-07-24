# Source provenance

The implementation is a clean, narrow rewrite from the read-only SLM-WM
baseline commit:

`34825098553d22f68f188afcd938d0aa72132caf`

It does not import the original repository at runtime.

| Responsibility | Baseline path | Baseline file SHA-256 |
|---|---|---|
| SD3.5 load and latent conversion | `experiments/runtime/diffusion/sd3_pipeline_runtime.py` | `115b0440831b85757f9daf72f6b0807494ca1bd5c174fe1d0c63feb57469eb32` |
| HF-tail template | `main/methods/carrier/high_frequency_tail.py` | `0ae921a876ba0a133ec192b827947c9e3a10fe051bfbac8e55c2de1fd8750369` |
| keyed PRG | `main/core/keyed_prg.py` | `9fd5f24023862afef4743dc6aca1cf0b4401f1ffb8d848c4d52f86616945cea2` |
| frozen quantile table | `main/core/normal_quantile_table.py` | `e98c2a0d76080d5080b8d22eb20cb7559c8291a668cf810aa508d89bc7b8776e` |
| HF blind score | `main/methods/carrier/blind_content_score.py` | `f018903a5d9d42dab65c91597b073a0fc7d5e448ac74501edf76472e29d6e990` |
| final-image latent input | `main/methods/detection/image_only.py` | `fc9a50307a5740e83d152253c6a3c34e56eca2707b9dfb95128f3d17ff1161d1` |
| key identities | `experiments/protocol/detection_key_identity.py` | `eaf93b9a1ad9835dde96c62bef836f2a45497bb0f93765055c073691fafd3195` |
| 33-key observation | `experiments/runners/terminal_content_carrier_runtime.py` | `46a2ee8db287a6683ad2c9c526eb9a1b865f69d177f3e9891037ac685e8fadf8` |
| Colab delivery boundary | `scripts/run_content_survival_observation_colab.py` | `81d1b3fb15f65973d24808797ffc81174a631303a3ba35335e102fd3d0c3750e` |

The actual-dtype Euler operator is aligned to Diffusers tag `v0.38.0`,
`scheduling_flow_match_euler_discrete.py` SHA-256
`56a330bc5765578ac9738265c76a3213ab42ad61befe8c040777c538da52045d`.

The original keyed PRG indexes a frozen midpoint normal inverse-CDF table with
a SHA-256 counter bitstream. This repository computes the same midpoint values
with `statistics.NormalDist.inv_cdf` and rounds them to IEEE binary32. During
construction all 1,048,576 entries were compared to the baseline table with
zero mismatches. A compact golden vector is committed in
`tests/fixtures/hf_template_golden.json` and remains the executable boundary.

The successful baseline late-HF energy was
`0.0015 * 8 = 0.012` relative L2. This repository applies that fixed energy to
the full normalized HF-tail template without the former semantic-routing mask.
It is therefore a new, simpler scientific candidate, not a byte-equivalent
reproduction of the prior formal method.

HF template and scorer identities use only the fixed `model_id` and
`model_revision` digest. Python, package, CUDA, and GPU identities are retained
separately in the runtime environment record and cannot change template bytes.
