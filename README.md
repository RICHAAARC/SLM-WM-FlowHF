# SLM-WM FlowHF

This repository is a deliberately small scientific execution path for testing
high-frequency keyed carriers with Stable Diffusion 3.5 and its actual
Flow-Matching schedule.

The first experiment has exactly two diffusion chains for one prompt:

1. a clean chain;
2. a watermarked chain with one fixed-energy HF write in the step-end callback
   at index 18 of a 20-step schedule.

For each frozen output it evaluates the same registered key and 32
domain-separated wrong keys in three observation domains:

- the real generation-time index-18 latent (oracle);
- the final image after VAE re-encoding (direct);
- the same final image numerically inverted to the index-18 FlowMatch state
  using the actual scheduler timesteps, sigmas, shift configuration and prompt
  conditioning (inversion).

This is diagnostic software. Every result records
`diagnostic_only=true`, `supports_paper_claim=false`, and does not perform
qualification, attacks, LF/QK composition, prompt selection, or parameter
search.

## Local CPU checks

```bash
python -m pytest -q
```

The real model smoke is explicitly excluded from the default test path:

```bash
python -m pytest -q tests/integration/test_gpu_smoke.py -m gpu
```

## Real run

The request JSON never contains the watermark key. Supply the key only through
an inherited environment entry; the CLI removes it before model loading:

```bash
FLOWHF_WATERMARK_KEY='...' \
python scripts/run_flow_hf_experiment.py \
  --request /path/to/audited_run_request.json
```

All scientific outputs are written below the request's `run_root`. The Colab
entry keeps the run on the VM local disk and only copies a secret-scanned final
archive and detached checksum to Drive.

The audited Colab request uses `/content/flowhf/runs/<run_id>` and contains the
fields represented by `FlowHFRunSpec`: exact repository commit, prompt and
seed, fixed SD3.5 model/revision, actual scheduler settings, injection index,
HF energy, and inversion iteration count. The raw watermark key is never a
request field.

In Colab, the non-secret request is read first. Public `main`, clone `HEAD`, and
the requested commit must agree before the detached checkout is installed from
`pyproject.toml`; only the installed, verified code may then read the raw key.
