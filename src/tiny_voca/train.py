"""From-scratch training on validated durations. Running this starts paid compute.

There is no learned aligner and no alignment loss: token durations come from
the annotation layer and must cover the acoustic frames exactly. The waveform
discriminators and the auxiliary mel head are training-only. No model or data
downloads and no infrastructure management happen here.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from .config import CHECKPOINT_FORMAT, Config
from .data import AnnotatedDataset, BucketSampler, collate
from .losses import (
    Discriminators,
    SpectralLoss,
    discriminator_loss,
    generator_gan_loss,
    masked_mean,
    pitch_shape_loss,
    scale_gradient,
)
from .model import TinyVoca
from .text import FRONTEND_RULES_VERSION, frontend_fingerprint, frontend_identity, frontend_rules_fingerprint

WEIGHTS = {"duration": 1.0, "pause": 1.0, "pitch": 2.0, "pitch_spread": 2.0, "voicing": 1.0, "energy": 1.0, "mel_head": 2.0,
           "stft": 2.0, "mel": 10.0}
OPTIONAL_WEIGHTS = {"pitch_shape": 0.0}
SEMITONES = 12 / math.log(2)


def masked_std(value, mask):
    """Standard deviation per row over the masked positions, [B]."""
    count = mask.sum(1).clamp_min(1)
    mean = (value * mask).sum(1) / count
    return (((value - mean[:, None]).square() * mask).sum(1) / count + 1e-8).sqrt()


class TrainingSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TinyVoca(config)
        self.mel_head = nn.Linear(config.channels, config.n_mels)   # auxiliary, not exported
        self.spectral = SpectralLoss(config.sample_rate)

    def objective(self, batch, teacher_probability, weights=WEIGHTS, random_phase=True, prosody_grad_scale=1.0):
        """Loss on full utterances. Returns ``(loss, terms, auxiliary)``.

        ``teacher_probability`` is the per-utterance chance that the acoustic
        decoder is conditioned on recorded pitch/voicing/energy; otherwise it
        gets the model's own (detached) predictions, exactly as at inference.
        ``prosody_grad_scale`` attenuates gradients flowing from ``decode`` back
        into ``prosody_hidden`` without changing the forward pass.
        The oscillator always follows the recorded pitch here, because the
        waveform target does.
        """
        m, c = self.model, self.model.config
        hidden, token_mask = m.encode(batch["tokens"], batch["stress"], batch["token_lengths"], batch["speakers"])
        output = m.duration(hidden, token_mask).float()
        # Length is learned only where a token has frames; whether a pause-capable
        # token has any is a separate yes/no target.
        present = batch["durations"] > 0
        duration = masked_mean((output[..., 0] - torch.log1p(batch["durations"].float())).square(), token_mask & present)
        pausable = token_mask & ~m.phone_tokens[batch["tokens"]]
        pause = masked_mean(F.binary_cross_entropy_with_logits(output[..., 1], present.float(), reduction="none"), pausable)
        frames, frame_mask, lengths = m.expand(hidden, batch["durations"], token_mask, batch["speakers"])
        if not torch.equal(lengths, batch["frame_lengths"]):
            raise ValueError("durations do not sum to the acoustic frame count")
        prosody_hidden, prediction = m.predict_prosody(frames, frame_mask)
        prediction, target = prediction.float(), batch["prosody"]
        trusted = batch["pitch_trusted"]
        # Recorded pitch counts fully; interpolated stretches only keep the curve continuous.
        pitch = masked_mean((prediction[..., 0] - target[..., 0]).abs(), frame_mask * (trusted + 0.1 * (1 - trusted)))
        # An absolute-error fit alone predicts a flatter contour than people speak
        # (about three quarters of the recorded spread on held-out text). Match each
        # utterance's pitch spread as well, the classic global-variance correction.
        sung = frame_mask * target[..., 1] * trusted
        enough = (sung.sum(1) >= 10).float()
        predicted_spread, recorded_spread = masked_std(prediction[..., 0], sung), masked_std(target[..., 0], sung)
        pitch_spread = masked_mean((predicted_spread - recorded_spread).abs(), enough)
        pitch_shape_weight = float(weights.get("pitch_shape", 0.0))
        if pitch_shape_weight > 0.0:
            pitch_shape = pitch_shape_loss(prediction[..., 0], target[..., 0], sung, energy=target[..., 2])
        else:
            with torch.no_grad():
                pitch_shape = pitch_shape_loss(prediction[..., 0], target[..., 0], sung, energy=target[..., 2])
        voicing = masked_mean(F.binary_cross_entropy_with_logits(prediction[..., 1], target[..., 1], reduction="none"), frame_mask)
        energy = masked_mean((prediction[..., 2] - target[..., 2]).abs(), frame_mask)
        teacher = torch.rand(len(lengths), device=lengths.device) < teacher_probability
        condition = torch.where(teacher[:, None, None], target, m.hard_prosody(prediction.detach()))
        decode_hidden = scale_gradient(prosody_hidden, prosody_grad_scale)
        acoustic = m.decode(decode_hidden, condition.to(prosody_hidden.dtype), frame_mask)
        mel_target = (batch["mel"] - c.mel_mean) / c.mel_scale
        mel_head = masked_mean((self.mel_head(acoustic).float() - mel_target).abs().mean(-1), frame_mask)
        phase = 2 * math.pi * torch.rand(len(lengths), device=lengths.device) if random_phase else None
        wave = m.vocoder(acoustic, m.frequency(target[..., 0]), lengths, phase=phase)
        stft, mel = self.spectral(wave, batch["wave"], lengths * c.hop)
        terms = dict(duration=duration, pause=pause, pitch=pitch, pitch_spread=pitch_spread, voicing=voicing, energy=energy,
                     mel_head=mel_head, stft=stft, mel=mel)
        loss = sum(weights[name] * value for name, value in terms.items())
        if pitch_shape_weight > 0.0:
            loss = loss + pitch_shape_weight * pitch_shape
        terms["pitch_shape"] = pitch_shape
        with torch.no_grad():
            voiced = sung
            predicted = m.predict_durations(hidden, token_mask, batch["tokens"])
            predicted_frames = predicted.sum(1)
            pause_frames = (batch["durations"] * pausable).sum().clamp_min(1)
            terms.update(
                base_loss=loss.detach(), teacher_fraction=teacher.float().mean(),
                pitch_error_semitones=SEMITONES * masked_mean((prediction[..., 0] - target[..., 0]).abs(), voiced),
                pitch_spread_ratio=masked_mean(predicted_spread / recorded_spread.clamp_min(1e-3), enough),
                voicing_accuracy=masked_mean(((prediction[..., 1] > 0) == target[..., 1].bool()).float(), frame_mask),
                pause_accuracy=masked_mean(((output[..., 1] > 0) == present).float(), pausable),
                pause_length_ratio=(predicted * pausable).sum() / pause_frames,
                duration_length_ratio=(predicted_frames / lengths).mean(),
                duration_length_error=((predicted_frames - lengths).abs() / lengths).mean())
        return loss, terms, {"wave": wave, "reconstruction": weights["stft"] * stft + weights["mel"] * mel}


def crops(fake, real, lengths, size, count):
    """Aligned random windows [B*count, size] from generated and recorded audio,
    plus the sample indices they cover ([B, count*size])."""
    size = min(size, int(lengths.min()))
    start = (torch.rand(len(lengths), count, device=fake.device) * (lengths[:, None] - size + 1)).long()
    index = (start[..., None] + torch.arange(size, device=fake.device)).flatten(1)
    return fake.gather(1, index).view(-1, size), real.gather(1, index).view(-1, size), index


def learning_rate(step, base, warmup, half_life, start=0):
    """Linear warm-up from ``start`` (0, or the step a run was branched at),
    then halving every ``half_life`` steps."""
    return base * min(1.0, (step - start + 1) / max(warmup, 1)) * 0.5 ** (step / half_life)


def teacher_probability(step, hold, ramp, final):
    """1 while the prosody predictor is still poor, then down to ``final``."""
    return 1.0 if step < hold else max(final, 1.0 - (1.0 - final) * (step - hold) / max(ramp, 1))


ACOUSTIC_PREFIXES = ("mel_head.", "model.acoustic_stack.", "model.prosody_projection.", "model.vocoder.")


def split_prosody_acoustic_parameters(system):
    """Partition system parameters into (prosody/encoder/duration, acoustic/vocoder/mel_head)."""
    prosody, acoustic = [], []
    for name, param in system.named_parameters():
        if name.startswith(ACOUSTIC_PREFIXES):
            acoustic.append(param)
        else:
            prosody.append(param)
    return prosody, acoustic


def clip_system_gradients(system, max_norm, separate_prosody=False):
    """Global gradient clipping (default) or separate prosody vs acoustic/vocoder clipping."""
    if not separate_prosody:
        grad = nn.utils.clip_grad_norm_(system.parameters(), max_norm, error_if_nonfinite=True)
        return float(grad), {}
    prosody, acoustic = split_prosody_acoustic_parameters(system)
    g_pros = float(nn.utils.clip_grad_norm_(prosody, max_norm, error_if_nonfinite=True))
    g_acou = float(nn.utils.clip_grad_norm_(acoustic, max_norm, error_if_nonfinite=True))
    return math.hypot(g_pros, g_acou), {"grad_norm_prosody": g_pros, "grad_norm_acoustic": g_acou}


class GradientSync:
    """Data-parallel training without a DDP wrapper: after each backward pass the gradients are averaged over
    the ranks, so every rank applies the same update to identical weights. A parameter that received a gradient
    on any rank gets one everywhere (zeros where it had none); parameters without a gradient on every rank keep
    None, as in a single process."""

    def __init__(self, world):
        import torch.distributed as dist
        self.dist, self.world = dist, world

    def average(self, module):
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            return
        device = params[0].device
        present = torch.tensor([p.grad is not None for p in params], dtype=torch.float32, device=device)
        self.dist.all_reduce(present, op=self.dist.ReduceOp.MAX)
        chosen = [p for p, flag in zip(params, present.tolist()) if flag]
        if not chosen:
            return
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in chosen])
        self.dist.all_reduce(flat)
        flat /= self.world
        offset = 0
        for p in chosen:
            n = p.numel()
            p.grad = flat[offset:offset + n].view_as(p).clone()
            offset += n

    def sum(self, tensor):
        tensor = tensor.clone()
        self.dist.all_reduce(tensor)
        return tensor


def optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe, sync=None):
    weights = recipe["weights"]
    branch = recipe.get("branched_from") or {}
    start = 0 if branch.get("keep_schedule") else branch.get("step", 0)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step, recipe["learning_rate"], recipe["warmup"], recipe["lr_half_life"], start)
    for group in critic_optimizer.param_groups:
        group["lr"] = learning_rate(step, recipe["learning_rate"], 1, recipe["lr_half_life"], start)
    optimizer.zero_grad(set_to_none=True)
    probability = teacher_probability(step, recipe["teacher_hold"], recipe["teacher_ramp"], recipe["teacher_final"])
    prosody_grad_scale = float(recipe.get("prosody_grad_scale", 1.0))
    loss, metrics, auxiliary = system.objective(batch, probability, weights, prosody_grad_scale=prosody_grad_scale)
    if step >= recipe["gan_start"]:
        wave = auxiliary["wave"]
        fake, real, index = crops(wave, batch["wave"], batch["frame_lengths"] * system.model.config.hop, recipe["crop_samples"], recipe["crops"])
        critic_optimizer.zero_grad(set_to_none=True)
        d_loss = discriminator_loss(critics(real), critics(fake.detach()))
        d_loss.backward()
        if sync is not None:
            sync.average(critics)
        nn.utils.clip_grad_norm_(critics.parameters(), 10, error_if_nonfinite=True)
        critic_optimizer.step()
        critics.requires_grad_(False)
        try:
            with torch.no_grad():
                real_features = critics(real)
            adversarial, matching = generator_gan_loss(real_features, critics(fake))
            gan = adversarial + recipe["feature_matching"] * matching
            # Fixed weights cannot balance these terms: measured at the waveform, the
            # adversarial gradient was under 1% of the spectral one, i.e. switched off.
            # Scale the GAN term so that, inside the windows the critics saw, its
            # gradient is ``gan_ratio`` times the reconstruction gradient.
            reconstruction_gradient = torch.autograd.grad(auxiliary["reconstruction"], wave, retain_graph=True)[0].gather(1, index)
            gan_gradient = torch.autograd.grad(gan, wave, retain_graph=True)[0].gather(1, index)
            ramp = min(1.0, (step - recipe["gan_start"] + 1) / max(recipe["gan_ramp"], 1))
            if sync is None:
                reconstruction_norm, gan_norm = reconstruction_gradient.norm(), gan_gradient.norm()
            else:   # norms over the whole (all-rank) batch, so every rank uses the same scale
                squares = sync.sum(torch.stack((reconstruction_gradient.square().sum(), gan_gradient.square().sum())))
                reconstruction_norm, gan_norm = squares[0].sqrt(), squares[1].sqrt()
            scale = (recipe["gan_ratio"] * ramp * reconstruction_norm / gan_norm.clamp_min(1e-12))
            scale = scale.clamp(max=recipe["gan_scale_max"]).detach()
            loss = loss + scale * gan
            metrics.update(discriminator=d_loss, adversarial=adversarial, feature_matching=matching, gan_scale=scale)
            loss.backward()
        finally:
            critics.requires_grad_(True)
    else:
        if recipe.get("align_gan_rng"):
            # Consume the exact same RNG draw as crops() so delayed-GAN replays stay RNG-aligned.
            torch.rand(len(batch["frame_lengths"]), recipe["crops"], device=batch["wave"].device)
        loss.backward()
    if sync is not None:
        sync.average(system)
    grad, clip_metrics = clip_system_gradients(
        system, recipe["grad_clip"], separate_prosody=bool(recipe.get("separate_prosody_clip", False))
    )
    optimizer.step()
    return {**{k: float(v.detach()) for k, v in metrics.items()}, **clip_metrics, "loss": float(loss.detach()), "grad_norm": float(grad),
            "frames": int(batch["frame_lengths"].sum()), "utterances": len(batch["frame_lengths"])}


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def export_model(model, path, step, annotation=None, frontend_rules=None):
    atomic_save({"format_version": CHECKPOINT_FORMAT, "config": model.config.to_dict(), "step": step,
                 "frontend": {**frontend_identity(), "fingerprint": frontend_fingerprint(),
                              "rules_version": FRONTEND_RULES_VERSION,
                              "rules_fingerprint": frontend_rules or frontend_rules_fingerprint()},
                 "annotation": annotation, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def load_compatible(module, state):
    """Load weights for a branched run. A tensor that grew (same rank, every
    dimension at least as large) keeps the old values in its leading block and
    its fresh initialisation elsewhere. Returns what was not an exact copy."""
    current = module.state_dict()
    report = {"grown": [], "new": [], "dropped": sorted(set(state) - set(current))}
    for name, tensor in current.items():
        if name not in state:
            report["new"].append(name)
        elif state[name].shape == tensor.shape:
            current[name] = state[name]
        elif state[name].ndim == tensor.ndim and all(a <= b for a, b in zip(state[name].shape, tensor.shape)):
            merged = tensor.clone()
            merged[tuple(slice(0, size) for size in state[name].shape)] = state[name]
            current[name] = merged
            report["grown"].append(name)
        else:
            raise ValueError(f"cannot branch: {name} changed shape from {tuple(state[name].shape)} to {tuple(tensor.shape)}")
    module.load_state_dict(current)
    return report


def balanced_indices(dataset, limit):
    """Stable round-robin over voices, independent of manifest order."""
    groups = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        groups[row["speaker"]].append(i)
    limit = min(max(limit, len(groups)), len(dataset.rows))
    selected = []
    for offset in range(limit):
        for voice in sorted(groups):
            if offset < len(groups[voice]):
                selected.append(groups[voice][offset])
                if len(selected) == limit:
                    return selected
    return selected


def check_leakage(train, validation):
    ids = {r["id"] for r in train.rows}
    texts = defaultdict(list)
    for row in train.rows:
        texts[row["text_hash"]].append(row["id"])
    collisions = []
    for row in validation.rows:
        if row["id"] in ids or row["text_hash"] in texts:
            collisions.append({"validation_id": row["id"], "train_ids": texts.get(row["text_hash"], [])[:3], "text": row.get("text", "")[:160]})
            if len(collisions) == 5:
                break
    if collisions:
        raise ValueError("train/validation leakage; split by sentence/text_id across BOTH voices. Examples: " + json.dumps(collisions, ensure_ascii=False))


@torch.no_grad()
def validate(system, dataset, indices, device, weights=WEIGHTS, batch_size=8):
    """Held-out metrics on a fixed balanced subset.

    ``teacher_forced`` conditions the decoder on recorded prosody, ``predicted``
    on the model's own; both use recorded durations and pitch for the oscillator
    so the spectra are comparable with the recording. Durations, pauses, pitch
    and voicing errors are those of the free-running predictors. This is a
    training monitor; pronunciation is judged separately with ASR and by listening.
    """
    system.eval()
    order = sorted(indices, key=lambda i: dataset.lengths[i])
    totals, weight = defaultdict(float), 0
    for start in range(0, len(order), batch_size):
        batch = collate([dataset[i] for i in order[start:start + batch_size]], device)
        size = len(batch["frame_lengths"])
        for name, probability in (("teacher_forced", 1.0), ("predicted", 0.0)):
            _, metrics, _ = system.objective(batch, probability, weights, random_phase=False)
            totals[f"validation_{name}_mel"] += float(metrics["mel"]) * size
            totals[f"validation_{name}_stft"] += float(metrics["stft"]) * size
        for key in ("duration", "pause_accuracy", "pause_length_ratio", "pitch_error_semitones", "pitch_spread_ratio",
                    "pitch_shape", "voicing_accuracy", "duration_length_ratio", "duration_length_error"):
            totals["validation_" + key] += float(metrics[key]) * size
        weight += size
    system.train()
    return {key: value / weight for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Annotation manifest (train.jsonl)")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-frames", type=int, default=9000, help="Padded acoustic frames per batch")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--lr-half-life", type=float, default=100000)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--weights", type=json.loads, default={}, help="JSON overrides for the loss weights, e.g. '{\"mel\": 5}'")
    parser.add_argument("--prosody-grad-scale", type=float, default=1.0,
                        help="Gradient gate alpha in [0, 1] from decode back into prosody_hidden (1.0 = reference)")
    parser.add_argument("--separate-prosody-clip", action="store_true",
                        help="Clip (encoder/duration/prosody) separately from (acoustic/vocoder/mel_head) at --grad-clip")
    parser.add_argument("--align-gan-rng", action="store_true",
                        help="When step < gan_start, consume the same RNG draw as GAN crops() to stay RNG-aligned with GAN arms")
    parser.add_argument("--gan-start", type=int, default=8000)
    parser.add_argument("--gan-ratio", type=float, default=0.7, help="Adversarial gradient as a fraction of the reconstruction gradient at the waveform")
    parser.add_argument("--gan-ramp", type=int, default=3000, help="Steps over which the adversarial term fades in while the critics warm up")
    parser.add_argument("--feature-matching", type=float, default=30.0, help="Feature-matching weight inside the adversarial term")
    parser.add_argument("--gan-scale-max", type=float, default=2000.0)
    parser.add_argument("--crop-samples", type=int, default=8192)
    parser.add_argument("--crops", type=int, default=2, help="Discriminator windows per utterance")
    parser.add_argument("--teacher-hold", type=int, default=3000, help="Steps with recorded prosody conditioning only")
    parser.add_argument("--teacher-ramp", type=int, default=7000)
    parser.add_argument("--teacher-final", type=float, default=0.0, help="Share of utterances still conditioned on recorded prosody "
                        "after the ramp (0 = the decoder sees only what it gets at inference)")
    parser.add_argument("--subset", type=int, help="Train on the first N utterances (balanced by voice): memorisation check")
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-every", type=int, default=10000, help="Keep a numbered checkpoint at this interval and at the GAN boundary")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--max-seconds", type=float, default=3600, help="Wall-time cap per invocation; save and exit at a step boundary")
    parser.add_argument("--deadline-file", type=Path, help="Budget JSON with an absolute 'deadline'; training saves and exits before it")
    parser.add_argument("--deadline-margin", type=float, default=300)
    parser.add_argument("--resume", type=Path, help="Continue the same run exactly (same recipe, data and configuration)")
    parser.add_argument("--init-from", type=Path, help="Start a NEW run from another run's checkpoint with a changed recipe: "
                        "weights, step counter and already-trained critics are taken over; the data order starts fresh")
    parser.add_argument("--restore-state-from-init", action="store_true",
                        help="With --init-from: restore generator, critics, both optimizers, sampler, CPU/CUDA RNG and LR schedule "
                             "identically to --resume while allowing recipe overrides for paired causal replays")
    parser.add_argument("--new-data", action="store_true", help="With --init-from: the checkpoint comes from a different annotation "
                        "run or front end (same symbol inventory)")
    parser.add_argument("--stop-file", type=Path, help="Save and exit after a step when this file appears")
    parser.add_argument("--recorded-voicing", action="store_true", help="Use the pitch tracker's voicing as it is, "
                        "without marking vowel frames voiced")
    parser.add_argument("--no-preload", action="store_true", help="Load feature files lazily instead of before the first step")
    parser.add_argument("--distributed", action="store_true", help="Data-parallel over the GPUs of this machine: launch with "
                        "torchrun --nproc_per_node N. Each rank takes 1/N of the utterances and of --batch-frames / --max-batch "
                        "(same global batch); gradients are averaged; rank 0 validates, logs and saves")
    parser.add_argument("--compact-cache", action="store_true", help="Keep cached waveforms as int16 when exact (half the RAM)")
    args = parser.parse_args()
    if min(args.steps, args.batch_frames, args.max_batch, args.save_every, args.keep_every, args.log_every, args.crops, args.gan_ramp) < 1:
        parser.error("steps, batch limits and intervals must be positive")
    if args.max_seconds <= 0 or args.learning_rate <= 0 or args.lr_half_life <= 0 or not 0 <= args.teacher_final <= 1:
        parser.error("time cap, learning rate and half life must be positive; teacher_final in [0, 1]")
    if not 0.0 <= args.prosody_grad_scale <= 1.0:
        parser.error("--prosody-grad-scale must be in [0, 1]")
    allowed_weights = set(WEIGHTS) | set(OPTIONAL_WEIGHTS)
    if set(args.weights) - allowed_weights or min(args.gan_ratio, args.feature_matching, args.gan_scale_max) < 0:
        parser.error(f"--weights may only override {sorted(allowed_weights)}; GAN settings must be non-negative")
    if any(float(v) < 0 for v in args.weights.values()):
        parser.error("--weights values must be non-negative")
    if args.resume and args.init_from:
        parser.error("--init-from starts a new run; --resume continues one")
    if args.restore_state_from_init and (not args.init_from or args.new_data):
        parser.error("--restore-state-from-init requires --init-from without --new-data")

    # The budget file is re-read while training so a deliberate extension (or cut) takes effect.
    budget = {"deadline": math.inf, "checked": time.time()}
    if args.deadline_file:
        budget["deadline"] = float(json.loads(args.deadline_file.read_text())["deadline"]) - args.deadline_margin

    def deadline():
        if args.deadline_file and time.time() - budget["checked"] > 30:
            budget["checked"] = time.time()
            try:
                budget["deadline"] = float(json.loads(args.deadline_file.read_text())["deadline"]) - args.deadline_margin
            except (OSError, ValueError, KeyError):
                pass    # keep the last valid deadline while the file is being rewritten
        return budget["deadline"]
    if time.time() >= deadline():
        raise SystemExit("the cloud budget deadline has passed (or is too close); not starting")
    c = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    rank, world, sync = 0, 1, None
    if args.distributed:
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if str(args.device).startswith("cuda"):
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"
        dist.init_process_group("nccl" if str(args.device).startswith("cuda") else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if args.batch_frames % world or args.max_batch % world:
            parser.error("--batch-frames and --max-batch must be divisible by the number of ranks")
        sync = GradientSync(world) if world > 1 else None
    lead = rank == 0

    def barrier():
        if world > 1:
            torch.distributed.barrier()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train = AnnotatedDataset(args.train, c, voiced_vowels=not args.recorded_voicing, compact=args.compact_cache)
    validation = AnnotatedDataset(args.validation, c, voiced_vowels=not args.recorded_voicing, compact=args.compact_cache)
    check_leakage(train, validation)
    if train.annotation != validation.annotation:
        raise ValueError("train and validation manifests come from different annotation runs")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("output directory is not empty; use --resume or a new directory")
    barrier()   # every rank has checked the output directory before rank 0 writes into it
    pool = balanced_indices(train, args.subset) if args.subset else list(range(len(train)))
    pool_size = len(pool)
    pool = pool[rank::world]   # each rank trains on (and caches) its own share of the utterances
    validation_indices = balanced_indices(validation, args.validation_size)
    if not args.no_preload:
        train.preload(pool)
        if lead:
            validation.preload(validation_indices)
    system, critics = TrainingSystem(c).to(args.device), Discriminators().to(args.device)
    count = sum(p.numel() for p in system.model.parameters())
    if count > 5_000_000:
        raise ValueError(f"inference parameter budget exceeded: {count:,}")
    optimizer = torch.optim.AdamW(system.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
    critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.0)
    sampler = BucketSampler([train.lengths[i] for i in pool], args.batch_frames // world, args.max_batch // world, args.seed + rank)
    if world > 1:
        torch.manual_seed(args.seed + 1000 * rank)   # weights were initialised identically; now per-rank randomness
    fingerprints = [train.fingerprint, validation.fingerprint]
    recipe = {key: getattr(args, key) for key in ("batch_frames", "max_batch", "learning_rate", "warmup", "lr_half_life", "grad_clip",
                                                   "gan_start", "gan_ratio", "gan_ramp", "feature_matching", "gan_scale_max",
                                                   "crop_samples", "crops", "teacher_hold", "teacher_ramp", "teacher_final",
                                                   "subset", "validation_size", "seed")}
    recipe["voiced_vowel_targets"] = not args.recorded_voicing
    recipe["weights"] = {**WEIGHTS, **{key: float(value) for key, value in args.weights.items() if key in WEIGHTS or float(value) > 0}}
    if args.prosody_grad_scale != 1.0:
        recipe["prosody_grad_scale"] = float(args.prosody_grad_scale)
    if args.separate_prosody_clip:
        recipe["separate_prosody_clip"] = True
    if args.align_gan_rng:
        recipe["align_gan_rng"] = True
    if world > 1:
        recipe["world_size"] = world
    recipe["branched_from"] = None
    step = 0

    def compatible(state, what, same_data=True):
        if state.get("format_version") != CHECKPOINT_FORMAT:
            raise ValueError(f"checkpoint predates phoneme tokens and supervised durations (format < 4); it cannot be {what}")
        if same_data and (state["frontend"] != frontend_fingerprint()
                          or state.get("frontend_rules") != frontend_rules_fingerprint()
                          or state["annotation"] != train.annotation):
            raise ValueError("checkpoint was trained with a different front end, phonetic rules or annotation run "
                             "(branch with --init-from and --new-data to fine-tune on new annotations)")
        if state["config"] != c.to_dict():
            raise ValueError("checkpoint was trained with a different model configuration")
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        compatible(state, "resumed")
        recipe["branched_from"] = state["recipe"].get("branched_from")
        if state["datasets"] != fingerprints or state["recipe"] != recipe:
            raise ValueError("resume recipe or dataset mismatch (use --init-from to branch a new run with a changed recipe)")
        system.load_state_dict(state["system"])
        critics.load_state_dict(state["critics"])
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        own = state["ranks"][rank] if world > 1 else state
        sampler.load_state(own["sampler"])
        torch.set_rng_state(own["torch_rng"].cpu())
        if str(args.device).startswith("cuda") and own["cuda_rng"] is not None:
            if world > 1:
                torch.cuda.set_rng_state(own["cuda_rng"].cpu())
            else:
                torch.cuda.set_rng_state_all([x.cpu() for x in own["cuda_rng"]])
        step = state["step"]
    elif args.init_from:
        state = torch.load(args.init_from, map_location=args.device, weights_only=True)
        # The symbol inventory is fixed (vocabulary size is part of the configuration),
        # so a checkpoint can be fine-tuned on another annotation run or front end.
        compatible(state, "branched", same_data=not args.new_data)
        step = state["step"]
        loaded = load_compatible(system, state["system"])
        if args.restore_state_from_init:
            if state["datasets"] != fingerprints:
                raise ValueError("--restore-state-from-init requires identical datasets")
            if loaded["grown"] or loaded["new"] or loaded["dropped"]:
                raise ValueError("--restore-state-from-init requires identical model parameter shapes")
            critics.load_state_dict(state["critics"])
            optimizer.load_state_dict(state["optimizer"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
            sampler.load_state(state["sampler"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if str(args.device).startswith("cuda") and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
            loaded.update(critics=True, optimizer=True, sampler=True, rng=True)
            recipe["branched_from"] = {
                "step": step,
                "keep_schedule": True,
                "recipe": state["recipe"],
                "datasets": state["datasets"],
                "loaded": loaded,
            }
        else:
            # Critics that have already been trained are taken over, otherwise a branch
            # in the adversarial phase would hand full-strength gradients from untrained
            # critics to the generator. Optimizer moments are kept when no tensor changed.
            loaded["critics"] = step > state["recipe"]["gan_start"]
            if loaded["critics"]:
                critics.load_state_dict(state["critics"])
                critic_optimizer.load_state_dict(state["critic_optimizer"])
            loaded["optimizer"] = not loaded["grown"] and not loaded["new"] and not loaded["dropped"]
            if loaded["optimizer"]:
                optimizer.load_state_dict(state["optimizer"])
            recipe["branched_from"] = {"step": step, "recipe": state["recipe"], "datasets": state["datasets"], "loaded": loaded}
            if args.new_data:
                recipe["branched_from"].update(frontend=state["frontend"], frontend_rules=state.get("frontend_rules"),
                                               annotation=state["annotation"])
    if step >= args.steps:
        raise ValueError("checkpoint already reached requested total steps")
    if lead:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "config.json").write_text(json.dumps(c.to_dict(), indent=2) + "\n")
        (args.output / "recipe.json").write_text(json.dumps({**recipe, "annotation": train.annotation, "frontend": frontend_fingerprint(),
                                                             "frontend_rules": frontend_rules_fingerprint(),
                                                             "train_utterances": pool_size, "datasets": fingerprints}, indent=2) + "\n")
        print(json.dumps({"inference_parameters": count, "training_only_parameters": sum(p.numel() for p in system.mel_head.parameters())
                          + sum(p.numel() for p in critics.parameters()), "train_utterances": pool_size, "start_step": step,
                          "branched": bool(args.init_from), "resumed": bool(args.resume), "world_size": world}), flush=True)
    barrier()
    start = time.monotonic()
    system.train()

    def own_state():
        if world > 1:   # this rank's device only
            cuda = torch.cuda.get_rng_state() if str(args.device).startswith("cuda") else None
        else:
            cuda = torch.cuda.get_rng_state_all() if str(args.device).startswith("cuda") else None
        return {"sampler": sampler.state(), "torch_rng": torch.get_rng_state(), "cuda_rng": cuda}

    def snapshot():
        """Collective when world > 1 (every rank must call it); the full state is returned on rank 0."""
        mine = own_state()
        ranks = None
        if world > 1:
            ranks = [None] * world
            torch.distributed.all_gather_object(ranks, mine)
        state = {"format_version": CHECKPOINT_FORMAT, "config": c.to_dict(), "frontend": frontend_fingerprint(),
                 "frontend_rules": frontend_rules_fingerprint(), "annotation": train.annotation,
                 "step": step, "datasets": fingerprints, "recipe": recipe, "system": system.state_dict(), "critics": critics.state_dict(),
                 "optimizer": optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(), **mine}
        if ranks is not None:
            state["ranks"] = ranks
        return state
    if step == args.gan_start and not args.resume:
        state = snapshot()
        if lead:
            atomic_save(state, args.output / f"step-{step:08d}.pt")
        barrier()
    window, window_steps, window_start = defaultdict(float), 0, time.monotonic()
    while step < args.steps:
        batch = collate([train[pool[i]] for i in sampler.next()], args.device)
        metrics = optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe, sync)
        step += 1
        if world > 1 and os.environ.get("TINY_VOCA_CHECK_SYNC"):   # debugging: the weights must stay identical on all ranks
            checksum = torch.stack([torch.cat([p.detach().double().reshape(-1) for p in m.parameters()]).sum()
                                    for m in (system, critics)])
            gathered = [torch.zeros_like(checksum) for _ in range(world)]
            torch.distributed.all_gather(gathered, checksum)
            if any(not torch.equal(g, gathered[0]) for g in gathered):
                raise RuntimeError(f"ranks diverged at step {step}: {[g.tolist() for g in gathered]}")
        for key, value in metrics.items():
            window[key] += value
        window_steps += 1
        elapsed = time.monotonic() - start
        should_stop = (elapsed >= args.max_seconds or step == args.steps or time.time() >= deadline()
                       or (args.stop_file is not None and args.stop_file.exists()))
        if world > 1:   # all ranks stop at the same step
            flag = torch.tensor([float(should_stop)], device=args.device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            should_stop = bool(flag.item())
        retain = step == args.gan_start or step % args.keep_every == 0
        save = step % args.save_every == 0 or should_stop or retain
        if step % args.log_every == 0 or save:
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.monotonic() - window_start
            record = {key: (value if key in ("frames", "utterances") else value / window_steps) for key, value in window.items()}
            record.update(step=step, elapsed_seconds=elapsed, steps_per_second=window_steps / max(seconds, 1e-9),
                          audio_seconds_per_second=window["frames"] * c.hop / c.sample_rate / max(seconds, 1e-9),
                          learning_rate=optimizer.param_groups[0]["lr"])
            if world > 1:
                record["world_size"] = world
            if save:
                if lead:   # validation first, as before: the saved RNG state is the one training continues from
                    record.update(validate(system, validation, validation_indices, args.device, recipe["weights"]))
                state = snapshot()
                if lead:
                    atomic_save(state, args.output / "last.pt")
                    export_model(system.model, args.output / "inference.pt", step, train.annotation)
                    if retain:
                        atomic_save(state, args.output / f"step-{step:08d}.pt")
                del state
                barrier()
            if lead:
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            window, window_steps, window_start = defaultdict(float), 0, time.monotonic()
        if should_stop:
            break
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
