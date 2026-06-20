#!/usr/bin/env python3
"""Export ABodyBuilder3 to ONNX for the Rust (ort) inference path.

This is the canonical producer of `vendored/abb3.onnx` (the model `abfv` runs).
It wraps the StructureModule forward (incl. all IPA/quaternion/frame geometry)
plus the atom14->atom37 conversion, traces it to ONNX, and verifies that
onnxruntime reproduces the PyTorch atom37 coordinates before writing the file.

Steps:
  1. Load checkpoint, build PyTorch baseline output for the example Fv.
  2. Wrap the model: plain tensors (single, pair, aatype) -> (atom37, mask).
  3. torch.onnx.export (opset 17), dynamic seq-length axis.
  4. onnxruntime run, compare to the PyTorch baseline; write atom37 reference.

Regenerate the committed model with:
    python workers/export_onnx.py   # writes vendored/abb3.onnx
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import numpy as np
import torch

from abodybuilder3.utils import string_to_input, add_atom37_to_output
from abodybuilder3.lightning_module import LitABB3
from abodybuilder3.openfold.data.data_transforms import make_atom14_masks
from abodybuilder3.openfold.utils.feats import atom14_to_atom37
import abodybuilder3.openfold.model.structure_module as sm


def permute_final_dims_onnx(tensor, inds):
    """ONNX-safe permute_final_dims: emits POSITIVE perm indices.

    The original (tensor_utils.py) returns a perm whose tail uses negative
    indices (zero_index + i), which torch.onnx exports as a Transpose with
    negative `perm` (e.g. {0,-3,-1,-4,-2}). onnxruntime rejects that. During
    tracing the rank is concrete, so we can resolve negatives to positives.
    """
    rank = len(tensor.shape)
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    tail = [rank + (zero_index + i) for i in inds]
    return tensor.permute(first_inds + tail)


# Patch the reference imported into the structure_module namespace.
sm.permute_final_dims = permute_final_dims_onnx

HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS"
LIGHT = "DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK"


class Wrapper(torch.nn.Module):
    """Plain-tensor-in / atom37-out wrapper around the StructureModule.

    The real model takes a dict {"single","pair"} + aatype and emits atom14
    positions. We fold the atom14->atom37 conversion (add_atom37_to_output)
    into the graph so ONNX directly returns:
        atom37:       [N, 37, 3]  final-block coordinates, masked
        atom37_exists [N, 37]     per-residue atom presence mask (from aatype)
    Both are pure functions of the model output + aatype lookup tables, so the
    Rust side only has to write a PDB from these two arrays.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, single, pair, aatype):
        out = self.model({"single": single, "pair": pair}, aatype)
        atom14 = out["positions"][-1, 0]                  # [N, 14, 3]
        batch = make_atom14_masks({"aatype": aatype.squeeze(0)})
        atom37 = atom14_to_atom37(atom14, batch)          # [N, 37, 3]
        atom37_exists = batch["atom37_atom_exists"]       # [N, 37]
        return atom37, atom37_exists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", default="vendored/best_second_stage.ckpt")
    ap.add_argument("-o", "--onnx", default="vendored/abb3.onnx",
                    help="output ONNX path (default: the committed/LFS model)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.onnx) or ".", exist_ok=True)

    device = torch.device("cpu")
    module = LitABB3.load_from_checkpoint(args.checkpoint, map_location=device)
    model = module.model.to(device).eval()
    print(f"[load] model loaded, use_plddt={getattr(model, 'use_plddt', '?')}")

    ab_input = string_to_input(heavy=HEAVY, light=LIGHT)
    single = ab_input["single"].to(device)          # [1, N, 23]
    pair = ab_input["pair"].to(device)              # [1, N, N, 132]
    aatype = ab_input["aatype"].unsqueeze(0).to(device)  # [1, N]
    print(f"[input] single={tuple(single.shape)} pair={tuple(pair.shape)} aatype={tuple(aatype.shape)} "
          f"dtypes={single.dtype},{pair.dtype},{aatype.dtype}")

    wrapper = Wrapper(model).eval()

    with torch.no_grad():
        torch_atom37, torch_exists = wrapper(single, pair, aatype)  # [N,37,3], [N,37]
    print(f"[torch] atom37={tuple(torch_atom37.shape)} exists={tuple(torch_exists.shape)} "
          f"mean={torch_atom37.mean().item():.4f} std={torch_atom37.std().item():.4f}")

    # Cross-check the folded wrapper against the real add_atom37_to_output path.
    with torch.no_grad():
        ref = model({"single": single, "pair": pair}, aatype)
    ref = add_atom37_to_output(ref, aatype)
    assert torch.allclose(torch_atom37, ref["atom37"], atol=1e-5), "fold != reference atom37"
    assert torch.equal(torch_exists, ref["atom37_atom_exists"]), "fold != reference mask"
    print("[fold] wrapper atom37 matches add_atom37_to_output reference")

    # --- export ---
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (single, pair, aatype),
            args.onnx,
            input_names=["single", "pair", "aatype"],
            output_names=["atom37", "atom37_exists"],
            dynamic_axes={
                "single": {1: "n"},
                "pair": {1: "n", 2: "n"},
                "aatype": {1: "n"},
                "atom37": {0: "n"},
                "atom37_exists": {0: "n"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"[export] wrote {args.onnx} ({os.path.getsize(args.onnx)/1e6:.1f} MB)")

    # --- verify graph ---
    import onnx
    m = onnx.load(args.onnx)
    onnx.checker.check_model(m)
    print(f"[onnx] checker OK, {len(m.graph.node)} nodes")

    # --- run ORT ---
    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ort_atom37, ort_exists = sess.run(
        ["atom37", "atom37_exists"],
        {
            "single": single.cpu().numpy(),
            "pair": pair.cpu().numpy(),
            "aatype": aatype.cpu().numpy(),
        },
    )
    ort_atom37 = torch.from_numpy(ort_atom37)
    ort_exists = torch.from_numpy(ort_exists)

    diff = (ort_atom37 - torch_atom37).abs()
    print(f"[compare] atom37 max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}")
    mask_ok = torch.equal(ort_exists, torch_exists)
    pos_ok = torch.allclose(ort_atom37, torch_atom37, atol=1e-3, rtol=0)
    print(f"[result] atom37 allclose(atol=1e-3): {pos_ok}  mask exact: {mask_ok}")
    if not (pos_ok and mask_ok):
        raise SystemExit("MISMATCH: ORT output diverges from PyTorch")
    print("[result] EXPORT VERIFIED")

    # Dump the ORT atom37 reference (consumed by the Rust `e2e` test) into out/,
    # kept separate from the model output: header "N\n" then N*37*3 LE f32, C order.
    os.makedirs("out", exist_ok=True)
    ref_path = "out/atom37_ref.f32"
    arr = ort_atom37.contiguous().numpy().astype("<f4")
    with open(ref_path, "wb") as fh:
        fh.write(f"{arr.shape[0]}\n".encode())
        fh.write(arr.tobytes(order="C"))
    print(f"[dump] wrote {ref_path} (N={arr.shape[0]})")


if __name__ == "__main__":
    main()
