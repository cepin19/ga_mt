import sys,itertools
for pair in sys.stdin:
    src,tgt=pair.split('\t')
    if ',' in src or ',' in tgt:
        product=itertools.product(src.split(','),tgt.split(','))
        print('\n'.join('\t'.join(p).strip() for p in product).strip())
    else:
        print(pair.strip())


