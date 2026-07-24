# Scientific scope

The current code answers one narrow question: where is an HF keyed signal most
visible for the same generated image?

It compares generation-time index-18, direct final-image VAE re-encoding, and
FlowMatch numerical inversion back to index-18. The registered key and all 32
wrong keys see the same frozen latent in each observation domain. Wrong keys
are never available to generation, injection, or inversion.

The inverse is numerical: each forward replay interval follows Diffusers 0.38
by upcasting the sample to float32 for Euler addition and casting the result to
the model-output dtype. Quantized round-trip error is reported, not described
as exact inversion.

The implementation intentionally excludes:

- LF and dual weighting;
- Q/K geometry;
- seven-chain probes;
- Jacobian, JVP, VJP, PSD-CG, or null-space claims;
- resume, artifact-binding, formal schema, qualification, and attacks;
- multi-prompt or injection-position selection.

One prompt is only a GPU smoke. Even a successful result is not paper evidence.
