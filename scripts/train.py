"""Train baseline or cross-fusion models."""

from _bootstrap import add_src_to_path

add_src_to_path()

from heatnet.commands.train import main

if __name__ == "__main__":
    main()
