#!/usr/bin/env python3
"""Generate the Fv 3D structure (ABodyBuilder3 wrapper)"""

import argparse
# import re
# import sys
# from pathlib import Path
import os
import torch
from abodybuilder3.utils import string_to_input, output_to_pdb, add_atom37_to_output
from abodybuilder3.lightning_module import LitABB3


def main():
    parser = argparse.ArgumentParser(
        description="predict 3D structure"
    )
    parser.add_argument("light", help="Light-chain (VL) amino-acid sequence.")
    parser.add_argument("heavy", help="Heavy-chain (VH) amino-acid sequence.")
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        help="Path to the ABodyBuilder3 checkpoint (.ckpt) file.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="./out",
        help="Output directory for generated files (default: ./out).",
    )
    parser.add_argument(
        "-f",
        "--out-file",
        default="complex.pdb",
        help="Output PDB file name (default: complex.pdb).",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    module = LitABB3.load_from_checkpoint(args.checkpoint, map_location=device)
    model = module.model.to(device).eval()
    print("ABodyBuilder3 base model loaded")

    # https://github.com/Exscientia/abodybuilder3/blob/main/notebooks/example.ipynb
    ab_input = string_to_input(heavy=args.heavy, light=args.light)
    ab_input_batch = {
        key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"] else value.to(device))
        for key, value in ab_input.items()
    }
    print("inputs moved to the device")

    with torch.no_grad():
        output = model(ab_input_batch, ab_input_batch["aatype"])

    output = add_atom37_to_output(output, ab_input["aatype"].to(device))
    pdb = output_to_pdb(output, ab_input)

    complex_pdb = os.path.join(args.out_dir, args.out_file)
    with open(complex_pdb, "w") as fh:
        fh.write(pdb)

    print("Done")


if __name__ == "__main__":
    main()
