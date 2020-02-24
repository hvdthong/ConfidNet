import argparse
from test import write_file

def load_file_yaml(path_file):
    lines = list(open(path_file, "r").readlines())
    lines = [l.rstrip() for l in lines]
    return lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--r", "-r", help="How many times we want to sample dataset", type=int, default=100)
    parser.add_argument("--s", "-s", help="Start times of random sampling", type=int, default=0)
    parser.add_argument("--e", "-e", help="End times of random sampling", type=int, default=100)
    args = parser.parse_args()
    print(args)

    for t in range(args.s, args.e):
        if args.d == 'mnist':
            yaml = load_file_yaml('./confs/selfconfid_%s.yaml' % (args.d))
        elif args.d == 'cifar':
            yaml = load_file_yaml('./confs/selfconfid_%s10.yaml' % (args.d))
        
        yaml[9] = yaml[9].rstrip() + '/' + str(t)
        write_file('./random_confs/%s_selfconfid_%i.yaml' % (args.d, t), yaml)
    

    