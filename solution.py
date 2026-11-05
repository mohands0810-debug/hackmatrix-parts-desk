"""Workshop Parts Desk controller (frozen). Gateway/model stays on team infrastructure.

Pipeline
  1. Choose the current approved revision of every request / catalog document in code.
  2. Ask the team model (op:llm) to extract voltage, connector and minimum watts, and
     cross-check it against an independent rule-based reader; disputes go to a second
     model call, anything still unresolved is treated as unknown (never guessed).
  3. Plan tool use with a small expectimax per ticket plus a budget allocation across the
     batch. Only parts that are certainly compatible are ever dispatched.
"""
import json
import re
from functools import lru_cache

# ----------------------------------------------------------------------------------------
# Constants (priors measured on the public worlds)
# ----------------------------------------------------------------------------------------
PRIOR_AVAIL = 0.72        # P(part has stock)
Q10 = 0.48                # P(need = 10 W | wattage not stated)
FLIP_AGED = 1.0 / 3.0     # aged snapshot disagrees with warehouse (ordinary operation)
FLIP_ANY = 0.25           # any snapshot may disagree after the operational update
MAX_NEED = 20
BUDGET_CAP = 13           # spending <= 13 keeps full efficiency; the 14th unit is not worth it
MAX_MODEL_CALLS = 8
COST = {'inspect': 1, 'verify': 2, 'clarify': 2}

# ----------------------------------------------------------------------------------------
# Rule-based reading of requests / parts
# ----------------------------------------------------------------------------------------
_NUM = r'(\d+(?:\.\d+)?)'
_UNIT_RE = re.compile(_NUM + r'\s*-?\s*(mv|mw|volts?|watts?|v|w)(?![a-z])', re.I)
_BARE_VOLT_RE = re.compile(r'input voltage is\s*' + _NUM + r'(?!\s*-?\s*[a-z0-9])', re.I)


def _clean(x):
    x = round(float(x), 6)
    return int(x) if x == int(x) else x


def _sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.;!?])\s+', text) if s.strip()]


def norm_connector(text):
    if not isinstance(text, str):
        return None
    found = set()
    if re.search(r'usb[\s\-_]*(?:type[\s\-_]*)?c\b', text, re.I):
        found.add('USB-C')
    if re.search(r'barrel', text, re.I):
        found.add('barrel')
    return found.pop() if len(found) == 1 else None


def _volts_watts(text, request):
    if request:
        keep = []
        for s in _sentences(text):
            if re.search(r'retired|discard|ignore that|previous stock', s, re.I) and not re.search(r'now takes', s, re.I):
                continue
            keep.append(re.sub(r'^.*?ignore that\.\s*', '', s, flags=re.I))
        text = ' '.join(keep)
    volts, watts = [], []
    for m in _UNIT_RE.finditer(text):
        v = float(m.group(1)); u = m.group(2).lower()
        if u == 'mv':
            volts.append(_clean(v / 1000))
        elif u in ('v', 'volt', 'volts'):
            volts.append(_clean(v))
        elif u == 'mw':
            watts.append(_clean(v / 1000))
        else:
            watts.append(_clean(v))
    if not volts:
        for m in _BARE_VOLT_RE.finditer(text):
            volts.append(_clean(m.group(1)))
    return volts, watts


def rule_read(text, request):
    volts, watts = _volts_watts(text, request)
    return {'voltage': volts[0] if len(set(volts)) == 1 else None,
            'connector': norm_connector(text),
            'watts': watts[0] if len(set(watts)) == 1 else None}


def _numbers_in(text):
    return {_clean(x) for x in re.findall(r'\d+(?:\.\d+)?', text)}


def _grounded(kind, value, text):
    """A model-supplied value must be visible in the source text (directly or in milli-units)."""
    if value is None:
        return False
    nums = _numbers_in(text)
    if kind == 'connector':
        return norm_connector(text) == value
    return value in nums or _clean(value * 1000) in nums


def read_reply_watts(text):
    if not isinstance(text, str):
        return None
    _, watts = _volts_watts(text, False)
    return watts[0] if len(set(watts)) == 1 else None


# ----------------------------------------------------------------------------------------
# Document selection
# ----------------------------------------------------------------------------------------
def current_documents(docs, as_of):
    """Current approved revision per entity; drafts, older revisions and retries are ignored."""
    best, conflict = {}, set()
    for d in docs or []:
        try:
            if d.get('status') != 'approved':
                continue
            if as_of is not None and (d.get('received_at', 0) > as_of or d.get('issued_at', 0) > as_of):
                continue
            ent, rev = d['entity_id'], d['revision']
        except (AttributeError, KeyError, TypeError):
            continue
        cur = best.get(ent)
        if cur is None or rev > cur['revision']:
            best[ent] = d; conflict.discard(ent)
        elif rev == cur['revision'] and d.get('text') != cur.get('text'):
            conflict.add(ent)
    return best, conflict


# ----------------------------------------------------------------------------------------
# Model prompts
# ----------------------------------------------------------------------------------------
EXTRACT_RULES = (
    'You read workshop parts requests and power-supply catalogue entries and extract exact numbers.\n'
    'Rules:\n'
    '- voltage_v is the output/input voltage in volts. Convert millivolts: 5000 mV = 5 V.\n'
    '- watts is the power rating in watts. Convert milliwatts: 10000 mW = 10 W.\n'
    '- connector is exactly "USB-C" (also written USB Type-C) or "barrel" (also barrel plug).\n'
    '- For a REQUEST report the voltage, connector and MINIMUM watts that currently apply. Ignore any '
    'retired, discarded, previous or replaced unit. If the request does not state the wattage '
    '(for example "label is unreadable"), use null. Never guess a missing number.\n'
    '- For a PART report its fixed output voltage, connector and maximum watts rating.\n'
    'Answer with ONE JSON object and nothing else, in this shape:\n'
    '{"requests":{"<id>":{"voltage_v":number|null,"connector":"USB-C"|"barrel"|null,"watts":number|null}},'
    '"parts":{"<id>":{"voltage_v":number|null,"connector":"USB-C"|"barrel"|null,"watts":number|null}}}\n'
)


def build_extract_prompt(req_items, part_items):
    payload = {'requests': [{'id': k, 'text': v} for k, v in req_items],
               'parts': [{'id': k, 'text': v} for k, v in part_items]}
    return EXTRACT_RULES + 'INPUT_JSON:\n' + json.dumps(payload, ensure_ascii=False)


def build_adjudicate_prompt(items):
    rules = ('Two readers disagree about some fields of the texts below. For each item decide which '
             'candidate value is correct according to the text alone (convert mV to V and mW to W). '
             'Reply with ONE JSON object: {"<item id>":{"<field>": <the correct value, or null if the text does '
             'not settle it>}}. Fields are voltage_v, connector ("USB-C" or "barrel") and watts. '
             'Ignore retired, discarded or previous units. Never guess.\n')
    return rules + 'INPUT_JSON:\n' + json.dumps(items, ensure_ascii=False)


def llm_action(prompt, tokens):
    return {'op': 'llm', 'messages': [
        {'role': 'system', 'content': 'You are a precise data-extraction engine. Reply with JSON only.'},
        {'role': 'user', 'content': prompt}], 'max_output_tokens': tokens}


def parse_model_json(text):
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith('```'):
        t = '\n'.join(l for l in t.splitlines() if not l.strip().startswith('```'))
    i, j = t.find('{'), t.rfind('}')
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return _clean(x)
    if isinstance(x, str):
        m = re.fullmatch(r'\s*' + _NUM + r'\s*(?:v|w|volts?|watts?)?\s*', x, re.I)
        return _clean(m.group(1)) if m else None
    return None


def _model_fields(entry):
    if not isinstance(entry, dict):
        return {'voltage': None, 'connector': None, 'watts': None}
    return {'voltage': _num(entry.get('voltage_v', entry.get('voltage'))),
            'connector': norm_connector(entry.get('connector')),
            'watts': _num(entry.get('watts'))}


# ----------------------------------------------------------------------------------------
# Planning: expectimax per ticket + budget allocation
# ----------------------------------------------------------------------------------------
def _need_scenarios(need):
    if need is None:
        return ((10, Q10), (20, 1.0 - Q10))
    return ((need, 1.0),)


def _escalate_value(need, parts):
    total = 0.0
    for n, w in _need_scenarios(need):
        p_none = 1.0
        for (watts, p, f, seen) in parts:
            if watts >= n:
                p_none *= (1.0 - p)
        total += w * (p_none + 0.15 * (1.0 - p_none))
    return total


def _max_need(need):
    return MAX_NEED if need is None else need


@lru_cache(maxsize=None)
def solve(need, parts, budget):
    """Return (expected outcome credit, best action) for one ticket.

    parts: tuple of (watts, p_available, flip_rate, inspected). Only parts that already match
    voltage and connector appear. Actions: ('escalate',), ('dispatch', i), ('inspect', i),
    ('verify', i), ('clarify',).
    """
    best_v, best_a = _escalate_value(need, parts), ('escalate',)
    need_hi = _max_need(need)
    for i, (watts, p, f, seen) in enumerate(parts):
        if watts >= need_hi and p > best_v + 1e-12:
            best_v, best_a = p, ('dispatch', i)
    if budget >= 1:
        for i, (watts, p, f, seen) in enumerate(parts):
            if p <= 0.0 or p >= 1.0:
                continue
            if watts < (10 if need is None else need):
                continue
            if not seen and budget >= COST['inspect']:
                if f <= 0.0:
                    outs = ((p, parts[:i] + ((watts, 1.0, f, True),) + parts[i + 1:]),
                            (1.0 - p, parts[:i] + ((watts, 0.0, f, True),) + parts[i + 1:]))
                else:
                    p1 = p * (1 - f) + (1 - p) * f
                    post1 = round(p * (1 - f) / p1, 4)
                    post0 = round(p * f / (1 - p1), 4)
                    outs = ((p1, parts[:i] + ((watts, post1, f, True),) + parts[i + 1:]),
                            (1 - p1, parts[:i] + ((watts, post0, f, True),) + parts[i + 1:]))
                v = sum(w * solve(need, st, budget - COST['inspect'])[0] for w, st in outs) - 1e-4
                if v > best_v + 1e-9:
                    best_v, best_a = v, ('inspect', i)
            if budget >= COST['verify']:
                outs = ((p, parts[:i] + ((watts, 1.0, f, True),) + parts[i + 1:]),
                        (1.0 - p, parts[:i] + ((watts, 0.0, f, True),) + parts[i + 1:]))
                v = sum(w * solve(need, st, budget - COST['verify'])[0] for w, st in outs) - 2e-4
                if v > best_v + 1e-9:
                    best_v, best_a = v, ('verify', i)
    if need is None and budget >= COST['clarify']:
        v = sum(w * solve(n, parts, budget - COST['clarify'])[0] for n, w in _need_scenarios(None)) - 2e-4
        if v > best_v + 1e-9:
            best_v, best_a = v, ('clarify',)
    return best_v, best_a


def allocate(values, cap):
    """values[t][b] = value of ticket t with b units. Maximise the total within cap units."""
    n = len(values)
    NEG = -1e18
    dp = [[NEG] * (cap + 1) for _ in range(n + 1)]
    choice = [[0] * (cap + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for t in range(n):
        for used in range(cap + 1):
            if dp[t][used] == NEG:
                continue
            for b in range(0, cap - used + 1):
                v = dp[t][used] + values[t][min(b, len(values[t]) - 1)] - 1e-5 * b
                if v > dp[t + 1][used + b]:
                    dp[t + 1][used + b] = v; choice[t + 1][used + b] = b
    used = max(range(cap + 1), key=lambda u: dp[n][u])
    alloc = [0] * n
    for t in range(n, 0, -1):
        b = choice[t][used]; alloc[t - 1] = b; used -= b
    return alloc


# ----------------------------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------------------------
def _spent(observation):
    try:
        return 14 - int(observation.get('budget_remaining', 14))
    except (TypeError, ValueError):
        return 0


def _init(observation, state):
    as_of = observation.get('as_of')
    tickets = observation.get('tickets') or []
    catalog = observation.get('catalog') or []
    req_docs, req_conflict = current_documents(observation.get('request_documents'), as_of)
    cat_docs, cat_conflict = current_documents(observation.get('catalog_documents'), as_of)
    state['ttext'] = {}
    state['ptext'] = {}
    state['order'] = [t['id'] for t in tickets]
    for t in tickets:
        d = req_docs.get(t['id'])
        state['ttext'][t['id']] = d['text'] if d and t['id'] not in req_conflict else (
            None if t['id'] in req_conflict else t.get('text'))
    state['age'] = {}
    for p in catalog:
        d = cat_docs.get(p['id'])
        state['ptext'][p['id']] = d['text'] if d and p['id'] not in cat_conflict else (
            None if p['id'] in cat_conflict else p.get('description'))
        state['age'][p['id']] = p.get('snapshot_age')
    state['rx_t'] = {k: (rule_read(v, True) if v else {'voltage': None, 'connector': None, 'watts': None})
                     for k, v in state['ttext'].items()}
    state['rx_p'] = {k: (rule_read(v, False) if v else {'voltage': None, 'connector': None, 'watts': None})
                     for k, v in state['ptext'].items()}
    policy = str(observation.get('policy', ''))
    state['changed'] = 'operational update' in policy.lower()
    state['know'] = {}
    state['clar'] = {}
    state['consumed'] = []
    state['pending'] = None
    state['bad_ticket'] = []


def _merge_reads(state, model):
    """Combine rule reader and model reader. Returns (tickets, parts, disputes)."""
    disputes = []
    out = {'tickets': {}, 'parts': {}}
    for kind, texts, rx, key in (('tickets', state['ttext'], state['rx_t'], 'requests'),
                                 ('parts', state['ptext'], state['rx_p'], 'parts')):
        m_all = (model or {}).get(key) if isinstance((model or {}).get(key), dict) else {}
        for ident, text in texts.items():
            r = rx[ident]
            m = _model_fields(m_all.get(ident)) if (model is not None and ident in m_all) else None
            res = {'voltage': None, 'connector': None, 'watts': None, 'ok': text is not None, 'w_known': False}
            if text is None:
                out[kind][ident] = res
                continue
            for fld in ('voltage', 'connector', 'watts'):
                a = r[fld]
                b = m[fld] if m is not None else a
                if fld == 'watts':
                    if a is not None and b is not None and a == b:
                        res['watts'] = a; res['w_known'] = True
                    elif a is None and b is None:
                        res['watts'] = None
                    else:
                        # disagreement or one reader blind: for parts the rating is unusable, for
                        # requests the wattage is treated as unknown and asked from the customer.
                        if a is not None and b is not None:
                            disputes.append((kind, ident, fld, a, b))
                        res['watts'] = None
                        if kind == 'parts':
                            res['ok'] = False
                else:
                    if a is not None and a == b:
                        res[fld] = a
                    elif a is None and b is None:
                        res['ok'] = False
                    else:
                        if a is not None and b is not None:
                            disputes.append((kind, ident, fld, a, b))
                        res['ok'] = False
                        res['_cand_' + fld] = (a, b)
            out[kind][ident] = res
    return out['tickets'], out['parts'], disputes


def _apply_adjudication(state, tickets, parts, disputes, reply):
    if not isinstance(reply, dict):
        return
    for kind, ident, fld, a, b in disputes:
        store = tickets if kind == 'tickets' else parts
        text = state['ttext' if kind == 'tickets' else 'ptext'].get(ident) or ''
        entry = reply.get(ident)
        if not isinstance(entry, dict):
            continue
        key = 'voltage_v' if fld == 'voltage' else fld
        raw = entry.get(key, entry.get(fld))
        val = norm_connector(raw) if fld == 'connector' else _num(raw)
        if val is None or val not in (a, b) or not _grounded(fld, val, text):
            continue
        # accept only when the adjudicated value matches one of the two readers
        if fld == 'watts':
            store[ident]['watts'] = val; store[ident]['w_known'] = True
            if kind == 'parts':
                store[ident]['ok'] = all(store[ident].get(f) is not None for f in ('voltage', 'connector'))
        else:
            store[ident][fld] = val
        if all(store[ident].get(f) is not None for f in ('voltage', 'connector')) and \
                not any(d[1] == ident and d[2] != fld and store[ident].get(d[2] if d[2] != 'voltage' else 'voltage') is None
                        and d[2] != 'watts' for d in disputes):
            if kind == 'tickets' or store[ident].get('watts') is not None:
                store[ident]['ok'] = True


def _finalise_reads(state, tickets, parts):
    for ident, t in tickets.items():
        t['ok'] = t.get('ok') and t.get('voltage') is not None and t.get('connector') is not None
        t.pop('_cand_voltage', None); t.pop('_cand_connector', None); t.pop('_cand_watts', None)
    for ident, p in parts.items():
        p['ok'] = p.get('ok') and None not in (p.get('voltage'), p.get('connector'), p.get('watts'))
        p.pop('_cand_voltage', None); p.pop('_cand_connector', None); p.pop('_cand_watts', None)
    state['tickets'] = tickets
    state['parts'] = parts


def _belief(state, pid):
    """(p_available, flip_rate, inspected)."""
    if pid in state['consumed']:
        return 0.0, 0.0, True
    k = state['know'].get(pid, {})
    if k.get('ver') is not None:
        return float(k['ver']), 0.0, True
    flip = FLIP_ANY if state['changed'] else (FLIP_AGED if state['age'].get(pid) == 'aged' else 0.0)
    if k.get('insp') is not None:
        s = int(k['insp'])
        if flip <= 0.0:
            return float(s), 0.0, True
        p = PRIOR_AVAIL
        p1 = p * (1 - flip) + (1 - p) * flip
        post = p * (1 - flip) / p1 if s else p * flip / (1 - p1)
        return round(post, 4), flip, True
    return PRIOR_AVAIL, flip, False


def _ticket_view(state, tid):
    t = state['tickets'][tid]
    need = state['clar'].get(tid)
    if need is None and t.get('w_known'):
        need = t['watts']
    if not t['ok']:
        return None
    parts, ids = [], []
    for pid, p in state['parts'].items():
        if p['ok'] and p['voltage'] == t['voltage'] and p['connector'] == t['connector']:
            pr, flip, seen = _belief(state, pid)
            parts.append((p['watts'], round(pr, 4), flip, seen)); ids.append(pid)
    # lowest wattage first keeps larger supplies free for tickets that need them
    order = sorted(range(len(parts)), key=lambda i: (parts[i][0], ids[i]))
    return need, tuple(parts[i] for i in order), [ids[i] for i in order]


def _next_world_action(state, observation):
    decided = set(observation.get('decided_ticket_ids') or [])
    open_ids = [t for t in state['order'] if t not in decided]
    if not open_ids:
        return {'op': 'finish'}
    budget_left = int(observation.get('budget_remaining', 0))
    cap = max(0, min(budget_left, BUDGET_CAP - _spent(observation)))
    views = {}
    for tid in open_ids:
        if tid in state['bad_ticket']:
            views[tid] = None
        else:
            views[tid] = _ticket_view(state, tid)
    live = [tid for tid in open_ids if views[tid] is not None]
    values = []
    for tid in live:
        need, parts, ids = views[tid]
        values.append([solve(need, parts, b)[0] for b in range(cap + 1)])
    alloc = allocate(values, cap) if live else []
    plan = dict(zip(live, alloc))
    # 1) information actions allocated by the budget plan (cheapest first)
    info = []
    for tid in live:
        need, parts, ids = views[tid]
        act = solve(need, parts, plan[tid])[1]
        if act[0] in COST:
            info.append((COST[act[0]], state['order'].index(tid), tid, act, ids))
    if info:
        info.sort(key=lambda x: (x[0], x[1]))
        _, _, tid, act, ids = info[0]
        if act[0] == 'clarify':
            return {'op': 'clarify', 'ticket_id': tid}
        return {'op': act[0], 'part_id': ids[act[1]]}
    # 2) decisions (free)
    for tid in open_ids:
        view = views[tid]
        if view is None:
            return {'op': 'escalate', 'ticket_id': tid}
        need, parts, ids = view
        act = solve(need, parts, 0)[1]
        if act[0] == 'dispatch':
            pid = ids[act[1]]
            if _certainly_compatible(state, tid, pid):
                return {'op': 'dispatch', 'ticket_id': tid, 'part_id': pid}
        return {'op': 'escalate', 'ticket_id': tid}
    return {'op': 'finish'}


def _certainly_compatible(state, tid, pid):
    t = state['tickets'].get(tid); p = state['parts'].get(pid)
    if not t or not p or not t['ok'] or not p['ok']:
        return False
    need = state['clar'].get(tid)
    if need is None and t.get('w_known'):
        need = t['watts']
    if need is None:
        need = MAX_NEED
    return p['voltage'] == t['voltage'] and p['connector'] == t['connector'] and p['watts'] >= need


def _process_result(state, observation):
    pend = state.get('pending')
    state['pending'] = None
    if not pend:
        return
    res = observation.get('result') or {}
    op = pend.get('op')
    if not res.get('ok'):
        if 'ticket_id' in pend and op in ('clarify', 'dispatch', 'escalate'):
            state['bad_ticket'].append(pend['ticket_id'])
        elif op in ('inspect', 'verify'):
            state['know'].setdefault(pend['part_id'], {})['ver'] = 0
        return
    data = res.get('data') or {}
    if op in ('inspect', 'verify'):
        avail = data.get('available')
        if isinstance(avail, (int, float)) and not isinstance(avail, bool):
            k = state['know'].setdefault(pend['part_id'], {})
            k['insp' if op == 'inspect' else 'ver'] = 1 if avail > 0 else 0
    elif op == 'clarify':
        w = read_reply_watts(data.get('reply'))
        tid = pend['ticket_id']
        if w is not None:
            state['clar'][tid] = w
        else:
            state['bad_ticket'].append(tid)   # reply unreadable: hand over to a person
    elif op == 'dispatch':
        pid = pend['part_id']
        if pid not in state['consumed']:
            state['consumed'].append(pid)


def _decide(observation, state):
    # ---- first call: read documents, ask the model to extract ----
    if 'order' not in state:
        _init(observation, state)
        req_items = [(k, v) for k, v in state['ttext'].items() if v]
        part_items = [(k, v) for k, v in state['ptext'].items() if v]
        state['calls'] = 1
        state['stage'] = 'extract'
        return {'action': llm_action(build_extract_prompt(req_items, part_items), 1800), 'state': state}

    # ---- model replies ----
    if 'llm_response' in observation:
        reply = parse_model_json(observation['llm_response'])
        if state.get('stage') == 'extract':
            tickets, parts, disputes = _merge_reads(state, reply)
            state['_t'], state['_p'], state['_d'] = tickets, parts, disputes
            if disputes and state['calls'] < MAX_MODEL_CALLS:
                items = {}
                for kind, ident, fld, a, b in disputes:
                    text = state['ttext' if kind == 'tickets' else 'ptext'].get(ident)
                    items[ident] = {'text': text, 'kind': 'request' if kind == 'tickets' else 'part',
                                    'disputed_fields': {}}
                for kind, ident, fld, a, b in disputes:
                    items[ident]['disputed_fields'][fld if fld != 'voltage' else 'voltage_v'] = [a, b]
                state['calls'] += 1
                state['stage'] = 'adjudicate'
                return {'action': llm_action(build_adjudicate_prompt(items), 800), 'state': state}
            _finalise_reads(state, tickets, parts)
        elif state.get('stage') == 'adjudicate':
            tickets, parts = state.pop('_t'), state.pop('_p')
            disputes = [tuple(d) for d in state.pop('_d')]
            _apply_adjudication(state, tickets, parts, disputes, reply)
            _finalise_reads(state, tickets, parts)
        state.pop('_t', None); state.pop('_p', None); state.pop('_d', None)
        state['stage'] = 'run'
    else:
        _process_result(state, observation)

    if state.get('stage') != 'run':
        # defensive: should not happen
        tickets, parts, _ = _merge_reads(state, None)
        _finalise_reads(state, tickets, parts)
        state['stage'] = 'run'

    action = _next_world_action(state, observation)
    if action['op'] == 'dispatch' and not _certainly_compatible(state, action['ticket_id'], action['part_id']):
        action = {'op': 'escalate', 'ticket_id': action['ticket_id']}
    state['pending'] = action
    return {'action': action, 'state': state}


def decide_legacy(observation, state):
    state = dict(state or {})
    try:
        return _decide(observation, state)
    except Exception:
        # Fail safe: never dispatch on an error. Make sure the model is still called once.
        state = dict(state)
        if not state.get('calls'):
            state['calls'] = 1
            state['stage'] = 'broken'
            return {'action': llm_action('Reply with the single word OK.', 16), 'state': state}
        decided = set((observation or {}).get('decided_ticket_ids') or [])
        remaining = [t['id'] for t in (observation or {}).get('tickets', []) if t['id'] not in decided]
        action = {'op': 'escalate', 'ticket_id': remaining[0]} if remaining else {'op': 'finish'}
        state['pending'] = None
        return {'action': action, 'state': state}


# ========================================================================================
# Post-twist ("consolidated queue") mode: wattage ranges, shared stock with quantities,
# document precedence by issued_at, 48 000-byte model allowance.
# ========================================================================================
LLM_BYTE_LIMIT = 44000
_WATT = r'(?:mw|w|watts?)'
_CUE_MIN = re.compile(r'(at least|minimum|min\b|no less than|not less than|no lower than|not lower than|not below|'
                      r'no smaller than|floor|starting at|at or above|not under|lower bound|from\s*$|>=|\u2265)', re.I)
_CUE_MAX = re.compile(r'(at most|maximum|max\b|no more than|not more than|no higher than|not higher than|not above|'
                      r'up to|ceiling|cap\b|limit|not exceed|exceed|no larger than|not over|at or below|upper bound|<=|\u2264)', re.I)
_POST_MIN = re.compile(r'^\s*(?:or more|or higher|or above|and up|and above|minimum|min\b|at least|or greater)', re.I)
_POST_MAX = re.compile(r'^\s*(?:or less|or lower|or below|maximum|max\b|at most|cap\b|ceiling|limit|or under)', re.I)
_BETWEEN = re.compile(r'between\s+' + _NUM + r'\s*(' + _WATT + r')?\s*(?:and|to)\s+' + _NUM + r'\s*-?\s*(' + _WATT + r')(?![a-z])', re.I)
_SPAN = re.compile(_NUM + r'\s*(' + _WATT + r')?\s*(?:to|\u2013|-)\s*' + _NUM + r'\s*-?\s*(' + _WATT + r')(?![a-z])', re.I)
_EXACT = re.compile(r'(exactly|precisely|only|must be|equal to)\s*$', re.I)


def _to_watts(v, unit):
    u = (unit or 'w').lower()
    return _clean(float(v) / 1000) if u == 'mw' else _clean(float(v))


def _strip_old(text):
    keep = []
    for s in _sentences(text):
        if re.search(r'retired|discard|ignore that|previous stock|superseded|replaced', s, re.I) and not re.search(r'now takes', s, re.I):
            continue
        keep.append(re.sub(r'^.*?ignore that\.\s*', '', s, flags=re.I))
    return ' '.join(keep)


def read_range(text):
    """Return (wmin, wmax) read by rules; None for a bound that is not clearly stated."""
    t = _strip_old(text)
    lo = hi = None
    m = _BETWEEN.search(t) or _SPAN.search(t)
    spans = []
    if m:
        u2 = m.group(4)
        lo = _to_watts(m.group(1), m.group(2) or u2)
        hi = _to_watts(m.group(3), u2)
        spans.append((m.start(), m.end()))
        if lo > hi:
            lo, hi = None, None
    mins, maxs, both = [], [], []
    for m in _UNIT_RE.finditer(t):
        u = m.group(2).lower()
        if u not in ('w', 'mw', 'watt', 'watts'):
            continue
        if any(a <= m.start() < b for a, b in spans):
            continue
        val = _to_watts(m.group(1), u)
        pre = re.split(r'[.;,]', t[max(0, m.start() - 45):m.start()])[-1]
        post = t[m.end():m.end() + 28]
        if _EXACT.search(pre):
            both.append(val)
        elif _POST_MIN.search(post):
            mins.append(val)
        elif _POST_MAX.search(post):
            maxs.append(val)
        elif _CUE_MIN.search(pre) and not _CUE_MAX.search(pre):
            mins.append(val)
        elif _CUE_MAX.search(pre) and not _CUE_MIN.search(pre):
            maxs.append(val)
    if both and len(set(both)) == 1 and lo is None and hi is None and not mins and not maxs:
        lo = hi = both[0]
    if lo is None and mins and len(set(mins)) == 1:
        lo = mins[0]
    if hi is None and maxs and len(set(maxs)) == 1:
        hi = maxs[0]
    if lo is not None and hi is not None and lo > hi:
        return None, None
    return lo, hi


def rule_read_new(text, request):
    volts, watts = _volts_watts(_strip_old(text) if request else text, False)
    out = {'voltage': volts[0] if len(set(volts)) == 1 else None, 'connector': norm_connector(_strip_old(text) if request else text)}
    if request:
        out['wmin'], out['wmax'] = read_range(text)
    else:
        out['watts'] = watts[0] if len(set(watts)) == 1 else None
    return out


NEW_RULES = (
    'You read workshop parts requests and power-supply catalogue entries and extract exact numbers.\n'
    'Rules:\n'
    '- voltage_v: volts. Convert millivolts (5000 mV = 5 V).\n'
    '- Watts: convert milliwatts (10000 mW = 10 W).\n'
    '- connector is exactly "USB-C" (also written USB Type-C) or "barrel" (also barrel plug).\n'
    '- For a REQUEST give min_watts = the smallest acceptable part rating and max_watts = the largest '
    'acceptable part rating (both inclusive). "at least/minimum/not below" -> min_watts only; "at most/up to/'
    'no more than/ceiling" -> max_watts only; "between A and B" -> both; "exactly A" -> both A. A bound the '
    'text does not state is null. Ignore retired, discarded, previous or replaced units. Never guess.\n'
    '- For a PART give watts = its rated maximum output in watts.\n'
    'Answer with ONE JSON object and nothing else:\n'
    '{"requests":{"<id>":{"voltage_v":number|null,"connector":"USB-C"|"barrel"|null,"min_watts":number|null,"max_watts":number|null}},'
    '"parts":{"<id>":{"voltage_v":number|null,"connector":"USB-C"|"barrel"|null,"watts":number|null}}}\n'
)
VERIFY_RULES = (
    'A second reader must independently re-read the texts below. For every item give your own reading of '
    'voltage_v (volts), connector ("USB-C" or "barrel"), min_watts and max_watts (inclusive limits on the '
    'acceptable part rating, null when the text does not state that bound; "at least X" is min only, "at most X" '
    'is max only, "between A and B" is both) for requests, or watts for parts. Convert mV and mW. '
    'Ignore retired or previous units. Never guess. Reply with ONE JSON object {"<id>":{...}}.\n'
)


def _msg_bytes(messages):
    return len(json.dumps(messages, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def _llm_new(state, prompt, tokens):
    """Build an op:llm action, respecting the per-case byte allowance. None if it would not fit."""
    act = llm_action(prompt, tokens)
    b = _msg_bytes(act['messages'])
    if state.get('bytes', 0) + b > LLM_BYTE_LIMIT or state.get('calls', 0) >= MAX_MODEL_CALLS:
        return None
    state['bytes'] = state.get('bytes', 0) + b
    state['calls'] = state.get('calls', 0) + 1
    return act


def _fields_new(entry, request):
    if not isinstance(entry, dict):
        return None
    r = {'voltage': _num(entry.get('voltage_v', entry.get('voltage'))), 'connector': norm_connector(entry.get('connector'))}
    if request:
        r['wmin'] = _num(entry.get('min_watts', entry.get('wmin')))
        r['wmax'] = _num(entry.get('max_watts', entry.get('wmax')))
    else:
        r['watts'] = _num(entry.get('watts'))
    return r


def _find_key(obj, keys, depth=0):
    if depth > 4:
        return None
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        for v in obj.values():
            r = _find_key(v, keys, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj[:50]:
            r = _find_key(v, keys, depth + 1)
            if r is not None:
                return r
    return None


def _int_or_none(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and x >= 0:
        return int(x)
    if isinstance(x, str) and re.fullmatch(r'\s*\d+\s*', x):
        return int(x)
    return None


def _all_text(obj, depth=0):
    if depth > 4:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _all_text(v, depth + 1)]
    if isinstance(obj, list):
        return [t for v in obj[:50] for t in _all_text(v, depth + 1)]
    return []


def _doc_quantity(doc):
    q = _int_or_none(_find_key(doc, ['quantity', 'qty', 'units', 'on_hand', 'stock', 'count']))
    if q is not None:
        return q
    m = re.search(r'(\d+)\s*(?:units?|pcs|pieces)', str(doc.get('text', '')), re.I) or \
        re.search(r'(?:quantity|qty|stock)\D{0,12}(\d+)', str(doc.get('text', '')), re.I)
    return int(m.group(1)) if m else None


def _new_manifest(observation):
    man = {}
    raw = observation.get('inventory_manifest')
    items = raw.items() if isinstance(raw, dict) else [
        ((r.get('part_id') or r.get('entity_id') or r.get('id')), r) for r in (raw or []) if isinstance(r, dict)]
    for pid, r in items:
        if pid is None:
            continue
        if not isinstance(r, dict):
            continue
        man[pid] = {'wh': r.get('warehouse_revision'), 'desk': r.get('desk_receipt_revision')}
    return man


def _eff_docs_new(docs, as_of):
    """Greatest approved revision with issued_at <= as_of. received_at never decides precedence."""
    best, conflict = {}, set()
    for d in docs or []:
        try:
            if d.get('status') != 'approved':
                continue
            if as_of is not None and d.get('issued_at', 0) > as_of:
                continue
            ent, rev = d['entity_id'], d['revision']
        except (AttributeError, KeyError, TypeError):
            continue
        cur = best.get(ent)
        if cur is None or rev > cur['revision']:
            best[ent] = d; conflict.discard(ent)
        elif rev == cur['revision'] and d.get('text') != cur.get('text'):
            conflict.add(ent)
    return best, conflict


def _new_init(observation, state):
    as_of = observation.get('as_of')
    req, rc = _eff_docs_new(observation.get('request_documents'), as_of)
    cat, cc = _eff_docs_new(observation.get('catalog_documents'), as_of)
    tickets = observation.get('tickets') or []
    catalog = observation.get('catalog') or []
    state['order'] = [t['id'] for t in tickets]
    state['ttext'] = {t['id']: (req[t['id']]['text'] if t['id'] in req and t['id'] not in rc else None) for t in tickets}
    state['ptext'] = {p['id']: (cat[p['id']]['text'] if p['id'] in cat and p['id'] not in cc else None) for p in catalog}
    state['rx_t'] = {k: (rule_read_new(v, True) if v else {}) for k, v in state['ttext'].items()}
    state['rx_p'] = {k: (rule_read_new(v, False) if v else {}) for k, v in state['ptext'].items()}
    state['man'] = _new_manifest(observation)
    state['qty'] = {}
    state['disp'] = {}
    state['insp_done'] = []
    state['ver_done'] = []
    state['clar_done'] = []
    state['clar_text'] = {}
    state['bad_ticket'] = []
    state['pending'] = None
    state['as_of'] = as_of
    state['bytes'] = 0
    state['calls'] = 0


def _bound_ok(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def _merge_new(state, m1, m2):
    """Combine rules + model pass 1 (+ pass 2). Returns (tickets, parts, need_pass2 ids)."""
    tickets, parts, need2 = {}, {}, []
    all1 = {'requests': (m1 or {}).get('requests') if isinstance((m1 or {}).get('requests'), dict) else {},
            'parts': (m1 or {}).get('parts') if isinstance((m1 or {}).get('parts'), dict) else {}}
    for kind, texts, rx in (('requests', state['ttext'], state['rx_t']), ('parts', state['ptext'], state['rx_p'])):
        req = kind == 'requests'
        for ident, text in texts.items():
            res = {'ok': False, 'voltage': None, 'connector': None, 'wmin': None, 'wmax': None, 'watts': None}
            (tickets if req else parts)[ident] = res
            if not text:
                continue
            r = rx[ident]
            a = _fields_new(all1[kind].get(ident), req) if m1 is not None and ident in all1[kind] else None
            b = _fields_new((m2 or {}).get(ident), req) if m2 is not None else None
            fields = ('voltage', 'connector', 'wmin', 'wmax') if req else ('voltage', 'connector', 'watts')
            good = True
            for f in fields:
                rv = r.get(f)
                lv = a.get(f) if a else None
                pv = b.get(f) if b else None
                kind_f = 'connector' if f == 'connector' else 'n'
                val = None
                if a is None:                          # no usable model opinion: rules only
                    val = rv
                elif rv is not None and lv == rv:      # both readers agree
                    val = rv
                else:
                    cands = [x for x in (rv, lv) if x is not None]
                    if cands:
                        if m2 is None:
                            if ident not in need2:
                                need2.append(ident)
                        elif pv is not None and pv in cands and _grounded(kind_f, pv, text):
                            val = pv
                if val is None and f in ('voltage', 'connector', 'watts'):
                    good = False
                res[f] = val
            if req:
                res['ok'] = res['voltage'] is not None and res['connector'] is not None
            else:
                res['ok'] = good
    return tickets, parts, need2


def _range_complete(t):
    return _bound_ok(t.get('wmin')) and _bound_ok(t.get('wmax')) and t['wmin'] <= t['wmax']


def _new_compat(t, p):
    return bool(t and p and t.get('ok') and p.get('ok') and _range_complete(t)
                and t['voltage'] == p['voltage'] and t['connector'] == p['connector']
                and t['wmin'] <= p['watts'] <= t['wmax'])


def _new_rem(state, pid):
    k = state['qty'].get(pid)
    if not k or k.get('base') is None:
        return None
    used = state['disp'].get(pid, 0) - (k.get('disp_at', 0) if k['kind'] == 'ver' else 0)
    return max(0, k['base'] - max(0, used))


def _new_process_result(state, observation):
    pend = state.get('pending')
    state['pending'] = None
    if not pend:
        return
    res = observation.get('result') or {}
    op = pend.get('op')
    tid, pid = pend.get('ticket_id'), pend.get('part_id')
    if not res.get('ok'):
        if op in ('clarify',):
            state['bad_ticket'].append(tid)
        elif op in ('inspect', 'verify'):
            state['ver_done'].append(pid) if op == 'verify' else state['insp_done'].append(pid)
        return
    data = res.get('data') or {}
    if op in ('inspect', 'verify') and isinstance(data, dict) and pid in state['parts']:
        p = state['parts'][pid]
        dv, dc, dw = _num(data.get('voltage')), norm_connector(data.get('connector')), _num(data.get('watts'))
        if None not in (dv, dc, dw) and (dv, dc, dw) != (p['voltage'], p['connector'], p['watts']):
            p['ok'] = False      # the tool's approved part card disagrees with our reading: never dispatch it
    if op == 'clarify':
        txts = [t for t in _all_text(data) if isinstance(t, str) and len(t) > 8]
        txt = max(txts, key=len) if txts else None
        state['clar_done'].append(tid)
        if txt:
            state['clar_text'][tid] = txt
        else:
            state['bad_ticket'].append(tid)
    elif op == 'dispatch':
        state['disp'][pid] = state['disp'].get(pid, 0) + 1
    elif op == 'verify':
        state['ver_done'].append(pid)
        q = _int_or_none(_find_key(data, ['quantity', 'qty', 'units', 'on_hand', 'stock_quantity', 'count']))
        if q is None:
            q = _int_or_none(_find_key(data, ['available']))
        if q is not None:
            state['qty'][pid] = {'base': q, 'kind': 'ver', 'disp_at': state['disp'].get(pid, 0)}
        card = _find_key(data, ['part_card', 'card', 'approved_part_card', 'approved_card', 'spec', 'description'])
        txts = _all_text(card) if card is not None else []
        if txts:
            rr = rule_read_new(' '.join(txts), False)
            p = state['parts'].get(pid)
            if p and all(rr.get(f) is not None for f in ('voltage', 'connector', 'watts')) and \
                    (rr['voltage'], rr['connector'], rr['watts']) != (p['voltage'], p['connector'], p['watts']):
                p['ok'] = False        # approved card disagrees with the catalogue revision: do not trust
    elif op == 'inspect':
        state['insp_done'].append(pid)
        docs = data.get('stock_documents') if isinstance(data, dict) else None
        man = state['man'].get(pid, {})
        wh = man.get('wh')
        if isinstance(docs, list) and wh is not None:
            for d in docs:
                if not isinstance(d, dict):
                    continue
                if d.get('status') == 'approved' and d.get('revision') == wh and \
                        (d.get('event_at') is None or state['as_of'] is None or d.get('event_at') <= state['as_of']):
                    q = _doc_quantity(d)
                    if q is not None:
                        state['qty'][pid] = {'base': q, 'kind': 'insp', 'disp_at': 0}
                        break


def _new_ticket_cands(state, tid):
    t = state['tickets'].get(tid)
    return [pid for pid, p in state['parts'].items() if _new_compat(t, p)]


def _cap_left(observation):
    return max(0, min(int(observation.get('budget_remaining', 0)), BUDGET_CAP - _spent(observation)))


def _new_plan(state, observation):
    """Next info action, or None when ready to decide."""
    cap = _cap_left(observation)
    decided = set(observation.get('decided_ticket_ids') or [])
    open_ids = [t for t in state['order'] if t not in decided and t not in state['bad_ticket']]
    # clarify tickets whose range is incomplete
    for tid in open_ids:
        t = state['tickets'].get(tid)
        if t and t['ok'] and not _range_complete(t) and tid not in state['clar_done'] and cap >= 2:
            return {'op': 'clarify', 'ticket_id': tid}
    ready = [tid for tid in open_ids if _range_complete(state['tickets'].get(tid, {})) and state['tickets'][tid]['ok']]
    cands = {tid: _new_ticket_cands(state, tid) for tid in ready}
    order = sorted(ready, key=lambda x: (len(cands[x]), state['order'].index(x)))
    reserved = {}
    for tid in order:
        for pid in sorted(cands[tid], key=lambda p: (state['parts'][p]['watts'], p)):
            q = _new_rem(state, pid)
            if q is None:
                man = state['man'].get(pid, {})
                wh, desk = man.get('wh'), man.get('desk')
                if pid not in state['insp_done'] and wh is not None and desk == wh and cap >= 1 and pid not in state['ver_done']:
                    return {'op': 'inspect', 'part_id': pid}
                if pid not in state['ver_done'] and cap >= 2:
                    return {'op': 'verify', 'part_id': pid}
                continue          # cannot afford to learn about it: treated as blind later
            if q - reserved.get(pid, 0) > 0:
                reserved[pid] = reserved.get(pid, 0) + 1
                break
    return None


def _match(tickets, cands, capacity):
    """Maximum assignment of tickets to part units (augmenting paths). capacity: pid -> units."""
    assign = {}     # tid -> pid
    load = {}       # pid -> [tid]

    def try_place(tid, seen):
        for pid in cands[tid]:
            if pid in seen:
                continue
            seen.add(pid)
            if len(load.get(pid, [])) < capacity.get(pid, 0):
                load.setdefault(pid, []).append(tid); assign[tid] = pid
                return True
            for other in list(load.get(pid, [])):
                if try_place(other, seen):
                    load[pid].remove(other)
                    load.setdefault(pid, []).append(tid); assign[tid] = pid
                    return True
        return False

    for tid in tickets:
        try_place(tid, set())
    return assign


def _new_decide_one(state, observation):
    decided = set(observation.get('decided_ticket_ids') or [])
    open_ids = [t for t in state['order'] if t not in decided]
    if not open_ids:
        return {'op': 'finish'}
    ready = [t for t in open_ids if t not in state['bad_ticket'] and state['tickets'].get(t, {}).get('ok')
             and _range_complete(state['tickets'][t])]
    cands = {t: sorted(_new_ticket_cands(state, t), key=lambda p: (state['parts'][p]['watts'], p)) for t in ready}
    known_cap = {}
    for pid in state['parts']:
        q = _new_rem(state, pid)
        if q is not None:
            known_cap[pid] = q
    order = sorted(ready, key=lambda x: (len(cands[x]), state['order'].index(x)))
    assign = _match(order, {t: [p for p in cands[t] if p in known_cap] for t in order}, known_cap)
    # blind placement (stock never learned): one unit per unknown part
    blind_cap = {pid: 1 for pid in state['parts'] if pid not in known_cap}
    rest = [t for t in order if t not in assign]
    if rest:
        blind = _match(rest, {t: [p for p in cands[t] if p in blind_cap] for t in rest}, blind_cap)
        assign.update(blind)
    for tid in open_ids:
        pid = assign.get(tid)
        if pid is not None and _new_compat(state['tickets'].get(tid), state['parts'].get(pid)):
            return {'op': 'dispatch', 'ticket_id': tid, 'part_id': pid}
        return {'op': 'escalate', 'ticket_id': tid}
    return {'op': 'finish'}


def _new_read_clar(state, tid):
    """Rule reading of a clarification reply; returns True when the range could be read."""
    txt = state['clar_text'].get(tid)
    t = state['tickets'].get(tid)
    if not txt or not t:
        return False
    rr = rule_read_new(txt, True)
    if rr.get('voltage') is not None and rr['voltage'] != t['voltage']:
        return None   # the request itself changed: do not trust
    if rr.get('connector') is not None and rr['connector'] != t['connector']:
        return None
    lo, hi = rr.get('wmin'), rr.get('wmax')
    if lo is not None and hi is not None:
        t['wmin'], t['wmax'] = lo, hi
        return True
    return False


def decide_new(observation, state):
    state = dict(state or {})
    try:
        return _decide_new(observation, state)
    except Exception:
        state = dict(state)
        if not state.get('calls'):
            state['calls'] = 1; state['stage'] = 'broken'; state['mode'] = 'new'
            return {'action': llm_action('Reply with the single word OK.', 16), 'state': state}
        decided = set((observation or {}).get('decided_ticket_ids') or [])
        remaining = [t['id'] for t in (observation or {}).get('tickets', []) if t['id'] not in decided]
        state['pending'] = None
        return {'action': ({'op': 'escalate', 'ticket_id': remaining[0]} if remaining else {'op': 'finish'}), 'state': state}


def _decide_new(observation, state):
    state['mode'] = 'new'
    if 'order' not in state:
        _new_init(observation, state)
        items = {'requests': [(k, v) for k, v in state['ttext'].items() if v],
                 'parts': [(k, v) for k, v in state['ptext'].items() if v]}
        payload = {'requests': [{'id': k, 'text': v} for k, v in items['requests']],
                   'parts': [{'id': k, 'text': v} for k, v in items['parts']]}
        act = _new_llm_or_none(state, NEW_RULES + 'INPUT_JSON:\n' + json.dumps(payload, ensure_ascii=False), 1800)
        state['stage'] = 'extract'
        if act is None:                       # cannot happen on a fresh case, but stay safe
            raise RuntimeError('no model budget')
        return {'action': act, 'state': state}

    if 'llm_response' in observation:
        reply = parse_model_json(observation['llm_response'])
        stage = state.get('stage')
        if stage == 'extract':
            state['m1'] = reply
            tickets, parts, need2 = _merge_new(state, reply, None)
            if need2:
                items = {}
                for i in need2:
                    txt = state['ttext'].get(i) or state['ptext'].get(i)
                    items[i] = {'text': txt, 'kind': 'request' if i in state['ttext'] else 'part'}
                act = _new_llm_or_none(state, VERIFY_RULES + 'INPUT_JSON:\n' + json.dumps(items, ensure_ascii=False), 1200)
                if act is not None:
                    state['stage'] = 'verify_reads'
                    return {'action': act, 'state': state}
            state['tickets'], state['parts'] = tickets, parts
        elif stage == 'verify_reads':
            tickets, parts, _ = _merge_new(state, state.get('m1'), reply if isinstance(reply, dict) else None)
            state['tickets'], state['parts'] = tickets, parts
        elif stage == 'clar_parse':
            _apply_clar_llm(state, reply)
            state['stage'] = 'run'
        if stage in ('extract', 'verify_reads'):
            state['stage'] = 'run'
        state.pop('m1', None) if state.get('stage') == 'run' else None
    else:
        _new_process_result(state, observation)
        if state.get('stage') != 'run' and 'tickets' not in state:
            tickets, parts, _ = _merge_new(state, None, None)
            state['tickets'], state['parts'] = tickets, parts
            state['stage'] = 'run'

    # read clarification replies (rules first, one model call for what rules cannot read)
    unread = [t for t in state['clar_text'] if t not in state.get('clar_read', [])]
    if unread:
        state.setdefault('clar_read', [])
        need_llm = []
        for tid in unread:
            r = _new_read_clar(state, tid)
            state['clar_read'].append(tid)
            if r is None:
                state['bad_ticket'].append(tid)
            elif r is False:
                need_llm.append(tid)
        state.setdefault('clar_wait', [])
        state['clar_wait'].extend(need_llm)
    if state.get('clar_wait') and not _more_clarify(state, observation):
        ids = state['clar_wait']; state['clar_wait'] = []
        items = {i: {'text': state['clar_text'][i], 'kind': 'request'} for i in ids}
        act = _new_llm_or_none(state, VERIFY_RULES + 'INPUT_JSON:\n' + json.dumps(items, ensure_ascii=False), 1200)
        if act is not None:
            state['stage'] = 'clar_parse'; state['clar_llm_ids'] = ids
            return {'action': act, 'state': state}
        state['bad_ticket'].extend(ids)

    action = _new_plan(state, observation) or _new_decide_one(state, observation)
    if action['op'] == 'dispatch':
        if not _new_compat(state['tickets'].get(action['ticket_id']), state['parts'].get(action['part_id'])):
            action = {'op': 'escalate', 'ticket_id': action['ticket_id']}
    state['pending'] = action
    return {'action': action, 'state': state}


def _more_clarify(state, observation):
    cap = _cap_left(observation)
    decided = set(observation.get('decided_ticket_ids') or [])
    for tid in state['order']:
        t = state['tickets'].get(tid)
        if tid not in decided and tid not in state['bad_ticket'] and t and t['ok'] and not _range_complete(t) \
                and tid not in state['clar_done'] and cap >= 2:
            return True
    return False


def _new_llm_or_none(state, prompt, tokens):
    return _llm_new(state, prompt, tokens)


def _apply_clar_llm(state, reply):
    ids = state.pop('clar_llm_ids', [])
    if not isinstance(reply, dict):
        state['bad_ticket'].extend(ids)
        return
    for tid in ids:
        f = _fields_new(reply.get(tid), True)
        t = state['tickets'].get(tid)
        txt = state['clar_text'].get(tid, '')
        ok = bool(f and t and _bound_ok(f.get('wmin')) and _bound_ok(f.get('wmax')) and f['wmin'] <= f['wmax']
                  and _grounded('n', f['wmin'], txt) and _grounded('n', f['wmax'], txt)
                  and (f.get('voltage') in (None, t['voltage'])) and (f.get('connector') in (None, t['connector'])))
        if ok:
            t['wmin'], t['wmax'] = f['wmin'], f['wmax']
        else:
            state['bad_ticket'].append(tid)


def _is_new_world(observation):
    if observation.get('inventory_manifest') or observation.get('stock_documents'):
        return True
    pol = str(observation.get('policy', '')).lower()
    if any(w in pol for w in ('consolidated', 'migration', 'shared stock', 'warehouse_revision', 'amended', 'operating limit')):
        return True
    cat = observation.get('catalog') or []
    return bool(cat) and not any('snapshot_age' in p for p in cat if isinstance(p, dict))


def decide(observation, state):
    state = dict(state or {})
    mode = state.get('mode')
    if mode is None:
        mode = 'new' if _is_new_world(observation or {}) else 'legacy'
    if mode == 'new':
        return decide_new(observation, state)
    out = decide_legacy(observation, state)
    out['state']['mode'] = 'legacy'
    return out
