#!/usr/bin/env python3
"""Visualize the contacts: per residue dSASA barchart"""


import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import os


def visualize(df, metric, threshold, out_dir, out_file):
    figure, axes = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True)

    for axis, chain in zip(axes, ["H", "L"]):
        sub_df = df[df.chain == chain]
        colors = ["#ff6347" if c else "#4787ff" for c in sub_df[metric] > threshold]

        axis.bar(sub_df["residue_number"], sub_df[metric].fillna(0),
           color=colors, width=0.9)

        axis.axhline(threshold, color="black", ls="--", lw=1)
        axis.set_title(f"{chain} chain")
        axis.set_ylabel("Δ side-chain SASA")
        axis.set_xlabel("residue #")
        axis.margins(x=0.01)

        # labelled
        axis.xaxis.set_major_locator(MultipleLocator(5))
        # unlabelled
        axis.xaxis.set_minor_locator(MultipleLocator(1))
        axis.tick_params(axis="x", which="major", labelsize=8, rotation=90)
        axis.grid(axis="x", which="both", alpha=0.15)

    figure.suptitle(f"{metric.upper()} (threshold {threshold*100:.0f}%)")
    png_path = os.path.join(out_dir, out_file)
    figure.savefig(png_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(
        description="visualize contacts"
    )
    parser.add_argument("contacts", help="contacts csv file")

    parser.add_argument(
        "-m",
        "--metric",
        default="contact_metric_rel",
        help="Which contact metric (csv column) to use.",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
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

    df = pd.read_csv(args.contacts).sort_values(["chain", "residue_number"]).reset_index(drop = True)
    visualize(df, args.metric, args.threshold,
              args.out_dir, args.out_file)

if __name__ == "__main__":
    main()
