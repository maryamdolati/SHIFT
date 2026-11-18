import runpy
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'protocol_registry.json').exists())
if __name__=='__main__':runpy.run_path(str(ROOT/'scripts'/'prepare_crypto_weights.py'),run_name='__main__')
