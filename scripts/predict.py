"""Run HeatNet inference and save predicted poses."""

from _bootstrap import add_src_to_path
add_src_to_path()

from heatnet.commands.predict import main


if __name__ == "__main__":
    main()
