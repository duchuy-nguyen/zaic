import sys, argparse
from .common import OFFICIAL
ap=argparse.ArgumentParser(); ap.add_argument("--split"); ap.add_argument("--sample")
a=ap.parse_args()
sys.path.insert(0,str((OFFICIAL/a.split).resolve()))
from zaic_tools import ToolEnv
import zaic_tools; print("module:",zaic_tools.__file__)
env=ToolEnv(a.split,a.sample,root=OFFICIAL)
print("validate ok:",env.validate("sys.ask_missing_info",{"tool_name":"contacts.get","params":["contact_id"]}))
print("state hash:",env.get_state_hash())