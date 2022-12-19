import sys
n=int(sys.argv[1])
lines=[l.strip() for l in sys.stdin]
print('\n'.join(lines[0::n]))
