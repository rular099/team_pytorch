# FE01 V2 locked experiment protocol

Task `20261001-fe01-native-window-random-time-v2`, base
`9c95dbfaf92f36b2673d816026cd3f25b91eec66`, branch
`exp/fe01-feature-extractor-site-effects`. V2 supersedes V1. No V1 implementation
was present at this base. Training and formal evaluation require manual HPC submission.

## Main comparison and controls

| Run family | Frontend | Trainable boundary | Native length at 100 Hz | Cumulative post-P capacity |
|---|---|---|---:|---:|
| diting_pretrained_frozen | MAE 1200M backbone + existing attention-pool adapter | adapter and common downstream | 10000 | 94.99 s |
| team_original_scratch | original TEAM CNN/flatten/MLP port | complete frontend and common downstream | 10000 | 94.99 s |
| phasenet_pretrained_frozen | STEAD v2 bottleneck, before decoder | masked pool + single projection and common downstream | 3001 | 25.00 s |
| eqt_pretrained_frozen | STEAD v2 entire shared encoder/CNN/LSTM/transformer trunk | masked pool + single projection and common downstream | 6000 | 54.99 s |

These capacities describe the registered implementation and weights, with 5 s
pre-P history and inclusive current sample. They are not theoretical limits of
the model families. Actual DiTing weights/native forward remain pending HPC audit.
PhaseNet and EQT native forward were checked locally with the downloaded weights.

The common downstream is the resolved RT55 base: dimension 1000, absolute
coordinate embeddings, two multi-station transformer layers, independent
target cross-attention readout, PGA MDN plus magnitude/location auxiliary heads.
No VS30, station-ID embedding, DPK, RT57–61 residual/teacher, trained downstream,
or optimizer transfer. All common state_dict values are initialized from a
separate RNG stream using the same seed. Audit exports both structure and state
fingerprints. Different seeds must not share one fingerprint.

`amplitude_only` uses zero waveform features plus the common physical scale
statistics and coordinates. `coords_only` also zeros scale statistics. Both
still receive causal input availability/geometry; they are not information-free
controls. `diting_random_frozen` is optional, independent, and excluded from the
default primary matrix. Controls are separately trained; zeroing a trained
primary model at inference is not the control experiment.

## Causal inputs and declared differences

Stored Japan waves are station/time/NS,EW,UD acceleration in m/s²; the frontend
receives station/component/time NEZ. PhaseNet/EQT are explicitly reordered ZNE.
The waveform reader slices only up to the exclusive cutoff. The native tensor
contains all samples from floor(first P) minus 500 through current sample,
aligned at its right edge. Storage holes and negative-start history are padded
and masked. Physical zeros remain valid. Storage holes never reduce the required
protocol history. Unsupported history produces N/A rather than a prediction.

All frontends share masked demeaning and a three-component joint peak. Before
normalizing, the shared amplitude path computes 3 standard deviations, 3 RMS,
3 peaks, joint peak in log10 SI units, and log1p(valid duration). Padding never
contributes. Multiplication by ten preserves normalized shape and adds one to
the ten log10 amplitude fields. The scale path retains the RT55 width 11 and
projection/gates; its semantics differ explicitly from legacy ln/full-padded
statistics. This is a new retraining protocol, not a replacement for historical
RT55 scores. The upstream offline resample/detrend/filtering is unchanged and
not certified causal. Catalog P picks are retrospective proxies for trigger
availability; they do not establish true live latency.

Original TEAM is ported from `yetinam/TEAM/models.py`, rather than the defective
legacy torch NormalizedScaleEmbedding. The port uses Conv2D on time/component,
channels-last flatten order and dynamically computed flatten width. Original
internal raw-ln-scale concatenation is replaced by the common scale boundary.
DiTing reuses its existing attention-pool adapter, while FE01 preprocessing
differs from the legacy per-component standard deviation normalization.
PhaseNet/EQT common joint-peak preprocessing differs from their picking-task
native conventions; metadata and this difference are retained. Picking
probabilities, annotate/classify and their hidden preprocessing are never used.

Frozen frontends have requires_grad=False and stay eval after parent.train();
BN buffers and dropout remain fixed. The adapter/projection and every common
downstream trainable parameter enter Adam. All-empty events are rejected before
attention; the loader records no_input and never silently substitutes an event.

## Sampling, budget, cohorts and selection

At 0.01 s resolution sample with replacement three times per event/epoch:
[1,3), [3,5), [5,10), [10,20), [20,40), [40,90], probabilities
0.20/0.20/0.20/0.15/0.15/0.10. Condition this base density on each native
capability, retaining partial-bin mass proportional to retained sample ticks.
Stable SHA-based RNG uses data/event/epoch/draw, independent of workers, ranks
and model family. All optimization seeds use the paired sampling seed 42.
Audit draws 100000 samples per capability with expected counts/tolerances.
These simulations are not actual training exposure.

Training geometry is 50% normal and 50% causal random. Normal takes the earliest
arrived eligible stations up to 25. Random uses counts 1/3/5/8/12/16, fixed
hash-based priority, and reserves a noninput target. Validation geometry is
deterministic and paired across models/seeds. No amplitude threshold chooses
input availability. Target sampling aims at 30% input, 20% triggered noninput,
50% untriggered; random excludes inputs and uses 20:50. Exhausted categories
redistribute remaining slots without replacement. Final event PGA labels are
unchanged, with finite labels supplying supervision only.

Audit locks a family-independent cohort before any model scoring. Events lacking
causal input/query at 1 s, or having no permitted KNET station, are excluded with
explicit reasons. Other malformed metadata fails audit. Training statistics are
recomputed on finite labels of the retained train events/stations; validation and
test never fit normalization. The exact validation population includes input
IDs and target roles at each common cell and is verified every epoch. Audit
limits for pilots are metadata-selected first 16 train / 8 validation events.

Main formal matrix: four families × seeds 42/43/44 = 12 independent trainings.
Adam lr 0.001, weight decay 0, clip norm 5, effective batch 128, microbatch 8,
12 epochs, three draws/event/epoch, validation each epoch, no early stopping.
Cosine schedule is common and fixed by actual optimizer updates. This deliberately
uses fixed-budget cosine rather than legacy validation-dependent plateau.
Total updates are 12 × floor(3 × retained train events / 128). DDP drops the
identical trailing effective-batch remainder rather than padding event repeats.
Change microbatch only before audit and only if microbatch × world size divides
128. Pilot batch 16 / microbatch 1, one epoch and at most four updates; with
16 retained events it executes three updates, not four.

Loss is the original MDN with weights magnitude/location/PGA = 0.02/0.02/1 and
the original distribution-mean Huber auxiliary weight 0.1. Selection is the mean
of MAE at common 1/3/5/10/20 s for noninput targets, then equal mean of normal
and random. Smaller wins; strict improvement preserves earlier-epoch ties.
40/90 s results are capability diagnostics and excluded from primary ranking.
Random nonstandard validation manifests are frozen before predictions; the audit
also supplies three uniform, nonfixed times in the shared 1–20 s range per event.
The optional manifest tool supplies weighted times across the full 1–90 s range
for separate capability diagnostics. These populations must not be mixed.

## Evaluation and evidence limits

Export full Gaussian MDN, point mean in log10(m/s²), exact mixture CDF quantiles,
PIT, NLL, closed-form CRPS, Brier at -1.2, reliability, 68/95% interval coverage,
mean±sigma diagnostics and separate linear mixture mean/median. Report all,
input, triggered/untriggered noninput, counts and numerical failures. Rankings
reject missing common decisions, unmatched target/input populations, labels or
provenance. Paired bootstrap resamples event clusters 5000 times, keeping all
station/time rows together; equal-time/protocol CIs accompany the primary
endpoint, and pooled-target/event-macro deltas are separately named.

Spatial/site diagnostics use a train-only ridge attenuation reference with final
catalog magnitude/hypocenter/depth, explicitly retrospective/oracle. A separate
shrunken training station effect is retained. Held-out-event residuals and
event-centered spatial differences never enter the waveform model. Station
statistics retain sparse sites, flag <10 events, and export event bootstrap CIs,
cross-station recovery, direction/strength stability, equal-distance contrasts,
pairwise delta errors and spatial spread. Do not infer uniquely geological
causation from a repeatable station residual.

The independent spatial experiment assigns 1° geographic blocks by fixed hash:
10% test, next 10% validation, 80% train; 20 km geodesic interface buffers.
Angular blocks have unequal physical areas. Train removes held/buffer stations
before reading waves/labels/reference picks, from both input and query roles;
normalization and reference are refit. Validation/test query only their held core
and input only train core, retaining event-disjoint splits. Empty spatial cores
or unsupported common populations fail audit instead of adapting the rule.

Replay rebuilds raw allowed history for every second 1–90. Cache scope is one T
with exact waveform/mask equality checks; no across-T token reuse. Native prefix
late unsupported frames remain blank. Optional rolling inference is labeled OOD
unless matched rolling training was separately declared. Grid predictions are
independent actual coordinate queries, with no fabricated station identity.
The conservative fallback is one-query forward. Audit compares batching/order
with independent queries at actual configured model width; only after passing
may the locked effective config use up to 15-query chunks. Maps show observed truth at stations only, train reference,
residuals, errors, mixture interval width and a <=100 km input-distance heuristic
coverage mask. Fixed color limits permit model comparison. Every GIF frame is a
new prediction of the same final PGA, never an interpolation.

Whole-frame P50/P95 include raw reading, preprocessing, encoder, all observation
and grid queries, and preparation overhead. Export/rendering overhead after the
measured inference frame is separately outside that latency; communication is
unknown, so result availability assumes ideal zero delay. Cold load and peak
device memory are recorded. Only measured replay P95 <=1 s may label that case/
region as 1 Hz; it does not establish national real-time readiness.

Offline weights are version/hash-checked without runtime downloads. STEAD/Japan
overlap and DiTing corpus provenance remain unknown until corpus-level audit.
Test needs an explicit flag, matching frozen lock hash and a user-written prior
exposure ledger. Prior project test exposure may be unknown and must stay so.
No formal model result, site-effect claim or national accuracy is asserted here.
