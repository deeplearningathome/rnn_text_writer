import sys


def main():
    out = open(sys.argv[2], 'w')
    for line in open(sys.argv[1], 'r').readlines():
        out.write(' '.join([c for c in unicode(line.lower().strip(), errors='ignore').replace(' ', '_') if c.isalpha() or
                            c == '_' or c == '.' or c == ',' or c == '!' or c == '?']) + ' \n ')

if __name__ == "__main__":
    main()

