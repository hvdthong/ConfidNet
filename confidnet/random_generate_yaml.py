import argparse

def load_file(path_file):
    lines = list(open(path_file, "r").readlines())
    lines = [l.strip() for l in lines]
    return lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")