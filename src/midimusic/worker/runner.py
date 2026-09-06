"""Standalone generation worker.

This file runs in the *provisioned runtime* -- a separate Python environment
containing torch and the model libraries -- not in the application process.
That boundary is deliberate and load-bearing:

* The frozen application excludes torch entirely, so the installer is around a
  hundred megabytes rather than several gigabytes. It could not import torch
  in-process even if it wanted to.
* One build serves CPU, CUDA and ROCm users, because the GPU-specific wheels
  live in the runtime rather than in the bundle.
* A native crash inside torch -- a real possibility on a new GPU stack -- kills
  this process and fails one job, instead of taking the user's session with it.

It therefore imports nothing from ``midimusic``: it must run under an
interpreter that has never heard of the application package.

Protocol: newline-delimited JSON on stdin and stdout. Audio is handed back
through a WAV file on disk rather than down the pipe, because multi-megabyte
base64 payloads through a pipe are slow and easy to deadlock.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

PROTOCOL_VERSION = 1


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def progress(fraction: float, text: str = "", stage: str = "") -> None:
    send({"event": "progress", "fraction": round(float(fraction), 4),
          "message": text, "stage": stage})


class Cancelled(Exception):
    pass


def _resolve_device(preferred: str) -> str:
    import torch

    if preferred and preferred not in ("auto", "rocm", "rocm-windows"):
        return preferred
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _dtype_for(device: str):
    import torch

    return torch.float32 if device in ("cpu", "mps") else torch.float16


def _write_wav(path: str, samples, rate: int) -> None:
    import numpy as np
    import soundfile as sf

    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim == 2 and data.shape[0] <= 2 < data.shape[1]:
        data = data.T
    sf.write(path, data, rate, subtype="FLOAT")


# -- backends ---------------------------------------------------------------

def run_musicgen(req: dict) -> dict:
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    repo = req.get("repo") or "facebook/musicgen-small"
    device = _resolve_device(req.get("device", "auto"))
    progress(0.05, f"Loading {repo}", "load")

    kwargs = {"token": req.get("token") or None,
              "local_files_only": bool(req.get("offline"))}
    processor = AutoProcessor.from_pretrained(repo, **kwargs)
    model = MusicgenForConditionalGeneration.from_pretrained(
        repo, dtype=_dtype_for(device), **kwargs
    ).to(device).eval()

    seconds = min(float(req.get("duration", 15.0)), float(req.get("max_duration", 30.0)))
    max_new_tokens = int(seconds * 50)  # MusicGen decodes at 50 Hz

    seed = req.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    inputs = processor(text=[req.get("prompt") or "instrumental music"],
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    progress(0.15, f"Generating {seconds:.0f}s", "generate")
    state = {"step": 0}

    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Reporter(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw) -> bool:
            state["step"] += 1
            if state["step"] % 10 == 0:
                progress(0.15 + 0.8 * min(1.0, state["step"] / max(1, max_new_tokens)),
                         "Generating audio", "generate")
            return False

    with torch.inference_mode():
        audio = model.generate(
            **inputs, do_sample=True,
            guidance_scale=float(req.get("guidance", 3.0)),
            temperature=max(0.1, float(req.get("temperature", 1.0))),
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList([_Reporter()]),
        )

    rate = int(model.config.audio_encoder.sampling_rate)
    _write_wav(req["output_path"], audio[0].to(torch.float32).cpu().numpy(), rate)
    return {"sample_rate": rate, "device": device, "model": repo}


def run_stable_audio(req: dict) -> dict:
    import torch
    from diffusers import StableAudioPipeline

    repo = req.get("repo") or "stabilityai/stable-audio-open-1.0"
    device = _resolve_device(req.get("device", "auto"))
    progress(0.05, f"Loading {repo}", "load")

    pipe = StableAudioPipeline.from_pretrained(
        repo, torch_dtype=_dtype_for(device),
        token=req.get("token") or None,
        local_files_only=bool(req.get("offline")),
    ).to(device)

    steps = int(req.get("steps", 100))
    generator = torch.Generator(device)
    if req.get("seed") is not None:
        generator.manual_seed(int(req["seed"]))

    def on_step(_pipe, step, _t, kwargs):
        progress(0.15 + 0.8 * (step / max(1, steps)), "Denoising", "generate")
        return kwargs

    progress(0.15, "Generating", "generate")
    out = pipe(
        prompt=req.get("prompt") or "ambient texture",
        negative_prompt=req.get("negative_prompt") or None,
        num_inference_steps=steps,
        audio_end_in_s=float(req.get("duration", 30.0)),
        num_waveforms_per_prompt=1,
        generator=generator,
        callback_on_step_end=on_step,
    )
    rate = int(pipe.vae.sampling_rate)
    _write_wav(req["output_path"], out.audios[0].to(torch.float32).cpu().numpy(), rate)
    return {"sample_rate": rate, "device": device, "model": repo}


def run_ace_step(req: dict) -> dict:
    # ACE-Step's language-model stage defaults to vLLM, which does not exist on
    # Windows or ROCm; the PyTorch backend must be selected before import.
    os.environ.setdefault("ACESTEP_LM_BACKEND", "pt")
    for key, value in (req.get("env") or {}).items():
        os.environ[str(key)] = str(value)

    device = _resolve_device(req.get("device", "auto"))
    progress(0.05, "Loading ACE-Step (first run compiles GPU kernels)", "load")

    from acestep.pipeline_ace_step import ACEStepPipeline

    pipeline = ACEStepPipeline(
        checkpoint_dir=req.get("models_dir") or None,
        device_id=0 if device != "cpu" else -1,
        dtype="float16" if device != "cpu" else "float32",
        torch_compile=False,  # no Triton on Windows, so compiling would fail
    )

    lyrics = (req.get("lyrics") or "").strip()
    if req.get("instrumental", True) or not lyrics:
        lyrics = "[instrumental]"

    progress(0.2, "Generating", "generate")
    params = {
        "prompt": req.get("prompt") or "instrumental music",
        "lyrics": lyrics,
        "audio_duration": float(req.get("duration", 120.0)),
        "infer_step": int(req.get("steps", 60)),
        "guidance_scale": float(req.get("guidance", 15.0)),
    }
    if req.get("seed") is not None:
        params["manual_seeds"] = str(req["seed"])
    if req.get("negative_prompt"):
        params["negative_prompt"] = req["negative_prompt"]

    try:
        output = pipeline(**params)
    except TypeError:
        output = pipeline(prompt=params["prompt"], lyrics=lyrics,
                          audio_duration=params["audio_duration"])

    samples, rate = _coerce(output, int(req.get("sample_rate", 48000)))
    _write_wav(req["output_path"], samples, rate)
    return {"sample_rate": rate, "device": device, "model": req.get("repo", "")}


def _coerce(output, default_rate: int):
    import numpy as np

    rate = default_rate
    data = output
    if isinstance(output, dict):
        rate = int(output.get("sample_rate", default_rate))
        data = output.get("audio", output.get("waveform", output.get("audios")))
    elif isinstance(output, (list, tuple)):
        if len(output) == 2 and isinstance(output[1], int):
            data, rate = output[0], int(output[1])
        else:
            data = output[0]
    if isinstance(data, str):
        import soundfile as sf

        samples, rate = sf.read(data, dtype="float32", always_2d=True)
        return samples, int(rate)
    if hasattr(data, "detach"):
        data = data.detach().to("cpu").float().numpy()
    samples = np.asarray(data, dtype=np.float32)
    while samples.ndim > 2:
        samples = samples[0]
    return samples, rate


def run_probe(_req: dict) -> dict:
    """Report what this runtime can actually do.

    Importing torch and reading ``cuda.is_available()`` is not enough. A ROCm
    build reports itself through the CUDA API, and whether its kernels really
    work on a given card is a question only running one answers -- so the probe
    executes a matmul and an attention call and reports whether they survived.
    """
    info: dict = {"python": sys.version.split()[0], "protocol": PROTOCOL_VERSION}
    try:
        import torch

        info["torch"] = torch.__version__
        info["hip"] = getattr(torch.version, "hip", None)
        info["cuda_build"] = getattr(torch.version, "cuda", None)
        info["is_rocm"] = bool(info["hip"])
        info["cuda"] = bool(torch.cuda.is_available())
        try:
            info["xpu"] = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        except Exception:
            info["xpu"] = False
        try:
            mps = getattr(torch.backends, "mps", None)
            info["mps"] = bool(mps and mps.is_available())
        except Exception:
            info["mps"] = False

        if info["cuda"]:
            device = "cuda"
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1
            )
        elif info["xpu"]:
            device = "xpu"
            info["device_name"] = torch.xpu.get_device_name(0)
        else:
            device = "cpu"
        info["device"] = device

        dtype = torch.float16 if device not in ("cpu", "mps") else torch.float32
        try:
            a = torch.randn(512, 512, device=device, dtype=dtype)
            info["matmul_ok"] = bool(torch.isfinite((a @ a).sum()).item())
        except Exception as exc:
            info["matmul_ok"] = False
            info["matmul_error"] = f"{type(exc).__name__}: {exc}"
        try:
            import torch.nn.functional as F

            q = torch.randn(1, 4, 128, 64, device=device, dtype=dtype)
            F.scaled_dot_product_attention(q, q, q)
            info["sdpa_ok"] = True
        except Exception as exc:
            info["sdpa_ok"] = False
            info["sdpa_error"] = f"{type(exc).__name__}: {exc}"
        info["usable"] = bool(info.get("matmul_ok") and info.get("sdpa_ok"))
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
        info["usable"] = False

    for module in ("transformers", "diffusers", "acestep", "soundfile", "numpy"):
        try:
            __import__(module)
            info[module] = True
        except Exception:
            info[module] = False
    return info


BACKENDS = {
    "probe": run_probe,
    "hf-musicgen": run_musicgen,
    "diffusers-audio": run_stable_audio,
    "ace-step": run_ace_step,
}


def main() -> int:
    send({"event": "ready", "protocol": PROTOCOL_VERSION, "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            send({"event": "error", "message": "malformed request"})
            continue

        command = request.get("cmd")
        if command == "shutdown":
            send({"event": "bye"})
            return 0

        handler = BACKENDS.get(command)
        if handler is None:
            send({"event": "error", "message": f"unknown command: {command}"})
            continue

        try:
            meta = handler(request)
            send({"event": "result", "meta": meta,
                  "output_path": request.get("output_path", "")})
        except Cancelled:
            send({"event": "cancelled"})
        except Exception as exc:
            send({"event": "error", "message": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-4000:]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
