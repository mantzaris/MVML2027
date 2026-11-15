"""Hard workload deadline and atomic PyTorch checkpoints (Pod stays running)."""
from datetime import datetime, timezone
from pathlib import Path
import os, random, tempfile, json
import numpy as np

_ROOT=Path(__file__).resolve().parents[2]
_EXECUTION=json.loads((_ROOT/'configs/execution.json').read_text())
DEADLINE=datetime.fromisoformat(_EXECUTION['research_deadline_utc'].replace('Z','+00:00'))

def budget_check(reserve_seconds=300):
    remaining=(DEADLINE-datetime.now(timezone.utc)).total_seconds()
    if remaining<reserve_seconds:raise RuntimeError('Research workload deadline reached; preserve results and leave Pod running')
    return remaining

def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def rng_state():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(state):
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def atomic_torch(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='checkpoint-',suffix='.pt');os.close(fd)
    try:
        torch.save(value,tmp)
        with open(tmp,'rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
