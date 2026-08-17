from hita.testing.golden import render_golden_stills

if __name__ == "__main__":
    for name, digest in render_golden_stills().items():
        print(f"{name}\t{digest}")
