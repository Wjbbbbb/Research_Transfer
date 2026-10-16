"""M0 smoke test for proving route-level model separation.

Run one route per process.  By default this only loads a route.  ``--image``
also runs a no-generation SELF forward to verify that no geometry input is
required before adding dataset-specific preparation or collection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from .loading import load_seek_model, load_self_model
from .types import Route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route", choices=[route.value for route in Route], required=True)
    parser.add_argument("--geometry-encoder-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", help="Optional JSON report path for long-running smoke tests.")
    parser.add_argument(
        "--image",
        help="Optional local image. On SELF, run a no-generation multimodal forward without geometry inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": args.device,
    }
    if args.route == Route.SELF.value:
        model = load_self_model(args.checkpoint, **load_kwargs)
    else:
        model = load_seek_model(
            args.checkpoint,
            geometry_encoder_path=args.geometry_encoder_path,
            **load_kwargs,
        )

    geometry_modules = [name for name, _ in model.named_modules() if "geometry_encoder" in name]
    report = {
        "route": args.route,
        "geometry_module_count": len(geometry_modules),
        "first_parameter_device": str(next(model.parameters()).device),
        "first_parameter_dtype": str(next(model.parameters()).dtype),
    }
    if args.image:
        if args.route != Route.SELF.value:
            raise ValueError("--image forward smoke is implemented for SELF only during M0")
        image = Image.open(args.image).convert("RGB")
        processor = AutoProcessor.from_pretrained(args.checkpoint)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        report["forward_logits_shape"] = list(outputs.logits.shape)
    serialized = json.dumps(report, sort_keys=True)
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
