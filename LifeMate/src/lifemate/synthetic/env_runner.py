import sys
from pathlib import Path
from .common import SYN_ROOT

def make_env(sid):
    p=str(SYN_ROOT.resolve())
    if p not in sys.path: sys.path.insert(0,p)
    from zaic_tools import ToolEnv
    return ToolEnv("synthetic", sid, root=Path("."))

def replay(env, calls):
    outs=[]
    for c in calls:
        r=env.call(c["name"],c["arguments"])
        if not r.get("ok"): return None
        outs.append(r)
    return outs