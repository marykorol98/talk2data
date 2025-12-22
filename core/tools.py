import json
import re
from core.prompts import CODE_BEGIN, CODE_END, PLAN_BEGIN, PLAN_END


def _extract_between(text: str, begin: str, end: str) -> str:
    if not text:
        return ""
    s = text.find(begin)
    if s == -1:
        return ""
    s += len(begin)
    e = text.find(end, s)
    if e == -1:
        return ""
    return text[s:e].strip()


def _safe_json_loads(s: str) -> dict | None:
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = s.strip().strip("`")
        s2 = re.sub(r",\s*}", "}", s2)
        s2 = re.sub(r",\s*]", "]", s2)
        try:
            return json.loads(s2)
        except Exception:
            return None


def _strip_markers_and_fences(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = t.replace("```python", "").replace("```", "")
    for m in (PLAN_BEGIN, PLAN_END, CODE_BEGIN, CODE_END):
        t = t.replace(m, "")
    return t.strip()


def code_is_plausible(code: str, intent: str) -> bool:
    if not code or not code.strip():
        return False
    if "df" not in code:
        return False
    if intent == "plot":
        return ("px." in code) or ("go." in code)
    return True

    
def process_output_code(code_output: str) -> str:
    code = _extract_between(code_output + CODE_END, CODE_BEGIN, CODE_END)
    code = _strip_markers_and_fences(code)

    # salvage fallback
    if not code:
        code = _strip_markers_and_fences(code_output)
        
    return code

def process_plan(generated_plan: str) -> dict | None:
    """
    because stop cuts after PLAN_END, we re-append it for extraction
    """
    plan_str = _extract_between(generated_plan + PLAN_END, PLAN_BEGIN, PLAN_END)
    plan = _safe_json_loads(plan_str)
    
    return plan
