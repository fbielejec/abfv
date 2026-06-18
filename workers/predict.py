#!/usr/bin/env python3
"""Generate the Fv 3D structure (ABodyBuilder3 wrapper)"""

import argparse
import re
import sys
from pathlib import Path
import os
import torch
from abodybuilder3.utils import string_to_input, output_to_pdb, add_atom37_to_output
from abodybuilder3.lightning_module import LitABB3

# CKPT = "/home/filip/CloudStation/Python/abodybuilder3/output/plddt-loss/best_second_stage.ckpt"
# OUT_DIR = "/home/filip/CloudStation/Python/abfv/out"
# os.makedirs(OUT_DIR, exist_ok=True)

def main():

    parser = argparse.ArgumentParser(
        description = "TODO"
    )
    # parser.add_argument("video", help="YouTube URL or video ID")
    # parser.add_argument(
    #     "-l",
    #     "--language",
    #     action="append",
    #     help="Preferred language code (e.g. en, pl). Can be repeated. "
    #     "Defaults to English, then first available.",
    # )
    # parser.add_argument(
    #     "-o", "--output", help="Output file path (default: <video_id>.md)"
    # )
    # parser.add_argument(
    #     "-t",
    #     "--timestamps",
    #     action="store_true",
    #     help="Include timestamps in output",
    # )
    # parser.add_argument(
    #     "--stdout",
    #     action="store_true",
    #     help="Print to stdout instead of saving to file",
    # )

    # args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device:", device)
    # module = LitABB3.load_from_checkpoint(CKPT, map_location = device)
    # model = module.model.to(device).eval()
    # print("ABodyBuilder3 base model loaded")

if __name__ == "__main__":
    main()
