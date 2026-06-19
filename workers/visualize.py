#!/usr/bin/env python3
"""Visualize the contacts: per residue dSASA barchart"""


import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(
        description="visualize contacts"
    )
    parser.add_argument("contacts", help="contacts csv file")

    parser.add_argument(
        "-t",
        "--threshold",
        default=0.1,
        help="Contact metric threshold value",
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
        default="dsasa_barplot.png",
        help="Output PNG file name.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.contacts)

    figure, axes = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True)

    for axis, chain in zip(axes, ["H", "L"]):
        sub_df = df[df.chain == chain]
        colors = ["#ff6347" if c else "#4787ff" for c in sub_df.is_contact]

        axis.bar(sub_df.residue_number, sub_df.contact_metric.fillna(0),
                 color=colors, width=0.9)

        axis.axhline(args.threshold, color="black", ls="--", lw=1)
        axis.set_title(f"{chain} chain")
        axis.set_ylabel("Δ side-chain SASA\n(% of isolated)")
        axis.set_xlabel("residue #")
        axis.margins(x=0.01)

    figure.suptitle(f"per-residue side-chain contact (threshold {args.threshold*100:.0f}%)")
    png_path = os.path.join(args.out_dir, args.out_file)
    figure.savefig(png_path, dpi=150)


if __name__ == "__main__":
    main()
