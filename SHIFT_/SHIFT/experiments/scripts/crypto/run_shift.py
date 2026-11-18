import sys
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'protocol_registry.json').exists())
sys.path.insert(0,str(ROOT))
from run_paper_protocols import main

if __name__=='__main__':
    if '--dataset' in sys.argv or '--comparison' in sys.argv:raise ValueError('Dataset and comparison are fixed by this entry point')
    main(['--dataset','crypto','--comparison','shift',*sys.argv[1:]])
