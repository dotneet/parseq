#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a PARSeq model to ONNX.

The exported graph always decodes `max_label_length + 1` steps: the data-dependent early exit of AR decoding
(stop once every sample has produced [EOS]) cannot be expressed in a static graph. The output logits can be decoded
with `model.tokenizer.decode()` as usual; everything after the first [EOS] is discarded.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from strhub.data.module import SceneTextDataModule  # noqa: E402
from strhub.models.utils import load_from_checkpoint, parse_model_args  # noqa: E402


class FixedLengthDecoding(nn.Module):
    """Decode the full max_label_length so the graph doesn't depend on the input data."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.max_length = model.hparams.max_label_length

    def forward(self, images):
        return self.model(images, self.max_length)


@torch.inference_mode()
def verify(model, onnx_path: Path, images: list[str]) -> bool:
    import onnxruntime as ort

    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    x = torch.stack([img_transform(Image.open(f).convert('RGB')) for f in images])
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    batch_size = sess.get_inputs()[0].shape[0]
    if isinstance(batch_size, int):  # Fixed batch size: run in chunks, padding the last one.
        chunks = []
        for i in range(0, len(x), batch_size):
            chunk = x[i : i + batch_size]
            pad = chunk[-1:].expand(batch_size - len(chunk), -1, -1, -1)
            chunks.append(sess.run(None, {'image': torch.cat([chunk, pad]).numpy()})[0][: len(chunk)])
        onnx_logits = torch.from_numpy(np.concatenate(chunks))
    else:
        onnx_logits = torch.from_numpy(sess.run(None, {'image': x.numpy()})[0])
    torch_logits = FixedLengthDecoding(model)(x)
    max_diff = (onnx_logits - torch_logits).abs().max().item()
    onnx_labels, _ = model.tokenizer.decode(onnx_logits.softmax(-1))
    torch_labels, _ = model.tokenizer.decode(model(x).softmax(-1))
    print(f'Max. abs. difference of logits (ONNX vs PyTorch): {max_diff:.2e}')
    ok = True
    for fname, onnx_label, torch_label in zip(images, onnx_labels, torch_labels):
        status = 'OK' if onnx_label == torch_label else f'MISMATCH (PyTorch: {torch_label})'
        ok &= onnx_label == torch_label
        print(f'{fname}: {onnx_label}  [{status}]')
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--output', '-o', type=Path, help='Output path (default: <model name>.onnx)')
    parser.add_argument(
        '--batch-size', type=int, default=0, help='Export with a fixed batch size (default: 0 = dynamic batch size)'
    )
    parser.add_argument('--verify', nargs='*', metavar='IMAGE', help='Compare against PyTorch using onnxruntime')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().cpu()
    if type(model).__name__ != 'PARSeq':
        raise SystemExit('Only PARSeq models are supported.')
    output = args.output or Path(f"{model.hparams.get('name', 'parseq')}.onnx")

    h, w = model.hparams.img_size
    # Use batch size > 1 for the dynamic case, otherwise torch.export specializes the batch dimension to 1.
    dummy = torch.rand(args.batch_size or 2, 3, h, w) * 2 - 1
    dynamic_shapes = None if args.batch_size else {'images': {0: torch.export.Dim('batch')}}
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            FixedLengthDecoding(model).eval(),
            (dummy,),
            str(output),
            input_names=['image'],
            output_names=['logits'],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            external_data=False,
        )
    print(f'Exported {output} in {time.perf_counter() - t0:.1f}s ({output.stat().st_size / 2**20:.1f} MB)')

    if args.verify is not None:
        images = args.verify or sorted(str(p) for p in (Path(__file__).parents[1] / 'demo_images').iterdir())
        if not verify(model, output, images):
            raise SystemExit('Verification failed.')


if __name__ == '__main__':
    main()
