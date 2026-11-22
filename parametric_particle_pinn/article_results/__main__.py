"""Render article graphics from the archived campaign."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND","Agg")
os.environ.setdefault("MPLCONFIGDIR","/tmp/pinn-article-mpl")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/pinn-article-cache")

import matplotlib.pyplot as plt

from .data import OUTPUT, SOURCE, Evidence
from . import plots


def save(output,name,figure):
    figure.savefig(output/"figures"/f"{name}.pdf")
    figure.savefig(output/"figures"/f"{name}.png",dpi=320)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--stage",choices=("mandatory","all"),default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(SOURCE):
        parser.error("Choose an output folder outside the pinned archive.")
    for directory in ("data","figures"):
        (output/directory).mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8,
                         "axes.labelsize":8,"pdf.fonttype":42,"axes.spines.top":False,
                         "axes.spines.right":False,"legend.fontsize":7.5})
    evidence = Evidence()
    for name,render in (("F06_duck",plots.duck),("F07_clearance_ablation",plots.ablation)):
        figure,_ = render(evidence,output)
        save(output,name,figure)
    if args.stage=="all":
        from .package import build
        build(evidence,output,save)
    print(output)


if __name__=="__main__":
    main()
