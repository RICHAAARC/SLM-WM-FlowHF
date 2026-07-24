# Scientific scope

## Current status

The current method is a **candidate primitive pending final hard-prompt
validation**. The lighthouse smoke and roster01/roster02 runs provide
small-scale support only. Roster03 and roster04 remain the final hard-prompt
gate; until both pass, the candidate is not a population-level, paper, or
qualification result.

## Candidate primitive

The candidate uses the fixed SD3.5 revision and a 20-step FlowMatch schedule.
It writes a registered-key-derived, uniform HF-tail carrier once inside
generation at callback index 18, after routing and actual-dtype materialization,
with relative L2 energy `0.012`. One scheduler interval remains after the
write. Semantic routing is disabled, so the current routing is uniform.

The primary detector scores the final image after VAE re-encoding. Only after
the generated image and observation latents are frozen does evaluation compare
the registered key with 32 wrong keys. The primary results are the watermarked
direct rank, registered-minus-maximum-wrong margin, and paired image quality.
Clean output is a negative control for systematic registered-key bias; an
occasional clean rank of one is not, by itself, an absolute per-prompt veto.
Across the frozen four-prompt roster, the arithmetic mean of the clean direct
registered-minus-maximum-wrong margins must be at most zero, and no more than
one clean direct result may have rank `1/33`. A single clean rank-one result
triggers recording and investigation but not rejection by itself; exceeding
either aggregate limit blocks freezing and migration.

All four paired quality metrics--PSNR, SSIM, MSE, and MAE--must be reported and
finite. Higher PSNR and SSIM and lower MSE and MAE are preferable. In this
roster gate they are primary descriptive results, not hard migration
thresholds. Any future threshold must be preregistered before a new,
independent experiment.

| Field | Frozen candidate value |
| --- | --- |
| Model and revision | Fixed SD3.5 identity owned by the FlowHF run spec |
| Scheduler and steps | FlowMatch Euler, 20 steps |
| Injection point | Callback index 18; one interval remains |
| Carrier domain | Generation latent, uniform HF tail |
| Routing | Uniform; semantic routing disabled |
| Template source | Registered-key-derived HF template |
| Relative L2 | `0.012`, after routing and actual-dtype materialization |
| Detector domain | Final image re-encoded by the VAE |
| Key evaluation | Frozen output, one registered key plus 32 wrong keys |
| Primary metrics | Direct rank and max-wrong margin; finite PSNR, SSIM, MSE, and MAE |
| Inversion role | Diagnostic paired control only |
| Excluded mechanisms | LF, dual weighting, Q/K, seven-chain probes, qualification |

## Inversion and non-primary mechanisms

FlowMatch inversion is retained as a diagnostic paired control. It is a
numerical fixed-point reconstruction, not exact inversion and not the primary
detector. Results so far show only a small margin increase and no change to the
direct detector's pass/fail decision.

LF and Q/K are not part of the current primary method. Semantic routing remains
disabled and may be studied later as an independent variable; it must not be
changed together with the first frozen primitive. The implementation also
excludes Jacobian, JVP, VJP, PSD-CG, null-space claims, attacks, and
injection-position selection.

## Conditional migration boundary

The primitive may be frozen and migrated only if all four frozen prompts show
watermarked direct rank `1/33` and an individual
registered-minus-maximum-wrong margin greater than zero; the clean direct
results satisfy both aggregate rules above; all four quality metrics are
complete, finite, and reported descriptively; and inversion does not change
the direct pass/fail decision.

Even if roster03 and roster04 pass and the fixed roster reaches `4/4`, this
permits freezing and migrating only the small-roster primitive. It does not
establish population-level, paper, qualification, generalization across
prompts, seeds, or keys, attack robustness, or fixed-FPR evidence.

The first migration must preserve the FlowHF injection position, energy, and
direct detector parameter for parameter. The former LF/HF dual and Q/K-coupled
route remains a historical non-passing route. Inversion remains an ablation
with no current decision benefit. Migration must not rewrite prior experiments
or failure conclusions.

This candidate phase does not add formal schemas, artifact binding, resume, or
qualification infrastructure.
