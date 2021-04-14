import sys
sys.dont_write_bytecode = True

def get_args ():

  import argparse
  # Get input file from command line
  parser = argparse.ArgumentParser(description="")
  parser.add_argument("--inputs", nargs="+",
    help="Input directories.")
  parser.add_argument("--input", default=None,
    help="Input directorie.")
  parser.add_argument("--outdir", default="./fig",
    help="Output directory.")
  parser.add_argument("--entries", nargs="+", default=[],
    help="Input directories.")
  parser.add_argument("--text", nargs="+", default=[],
    help="To be displayed on plots")
  parser.add_argument("--title-z", default="Bottleneck size",
    help="title of the Z axis of the histogram.")
  parser.add_argument("--n-samples", default=1000,
    help="Number of samples used fro plotting.")
  parser.add_argument("--col", default="z_vae_",
    help="Column to read from dataframe.")
  return parser.parse_args()
