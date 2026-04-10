"""Unified data preparation entrypoint for HeatNet."""

from _bootstrap import add_src_to_path

add_src_to_path()

from heatnet.commands.prepare_data import main


if __name__ == "__main__":
    main()
