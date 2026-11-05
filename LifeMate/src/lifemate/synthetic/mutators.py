from copy import deepcopy
from .common import norm_text, tokens
GROUND_KEYS = ("name","room","relation","nickname","label","title","display_name","type","category")
ID_KEY = "id"

def _walk(o, ref_id, path=()):
    """Tìm dict có id==ref_id ở bất kỳ đâu trong state; trả (dict, đường_dẫn)."""
    if isinstance(o, dict):
        if o.get(ID_KEY) == ref_id:
            return o, path
        for k, v in o.items():
            r = _walk(v, ref_id, path+(k,))
            if r: return r
    elif isinstance(o, list):
        for i, v in enumerate(o):
            r = _walk(v, ref_id, path+(i,))
            if r: return r
    return None

def _parent_list(state, path):
    o = state
    for p in path[:-1]:
        o = o[p]
    return o, path[-1]

def duplicate_entity(state, ref_id, rng):
    """Generic: nhân bản entity theo ref_id; trả (grounds:list[str]) hoặc None."""
    found = _walk(state, ref_id)
    if not found: return None
    ent, path = found
    grounds = [str(ent[k]) for k in GROUND_KEYS if isinstance(ent.get(k), str) and ent[k]]
    if not grounds: return None                      # không có gì để ground => bỏ
    dup = deepcopy(ent)
    dup[ID_KEY] = str(ent[ID_KEY]) + "_syn"
    for pk in ("phone","number","msisdn"):
        if dup.get(pk):
            d = list(str(dup[pk])); rng.shuffle(d); dup[pk] = "".join(d)
    par, idx = _parent_list(state, path)
    if isinstance(par, list): par.insert(idx+1, dup)  # chèn cạnh bản gốc
    elif isinstance(par, dict): par[list(par.keys())[-1]] = dup  # fallback (hiếm)
    return grounds

def groundable(grounds, audio_text):
    """1 ground string khớp audio = substring HOẶC share token len>=3."""
    at = norm_text(audio_text); atok = set(tokens(audio_text))
    for g in grounds:
        gn = norm_text(g)
        if gn and gn in at: return True
        if any(t in atok and len(t) >= 3 for t in tokens(g)): return True
    return False