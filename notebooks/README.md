# Notebook boundary

`content_survival_flow_hf_colab.ipynb` is the one-prompt A100-80GB entry. It
contains only public clone/bootstrap, one local scientific call, and an
independent final packaging call. Its committed bytes are pinned by
`registry.json`.

The bootstrap reads only the non-secret request, verifies public `main` and the
detached commit, installs that commit's pinned `pyproject.toml` dependencies,
and only then imports project code that reads the raw key into memory.

After the first independent audit, experiment changes belong in `flowhf/` or
`scripts/`; the notebook is not a scientific implementation surface.
