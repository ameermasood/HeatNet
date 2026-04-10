"""Evaluate saved 2D keypoint predictions with PnP and ADD."""

from _bootstrap import add_src_to_path
add_src_to_path()

from heatnet.commands.evaluate import main


if __name__ == "__main__":
    main()
