import sys

from test_graph_generation import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("msd")
    main()
