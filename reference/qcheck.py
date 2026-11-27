#!/usr/bin/env python3
"""Independent checker for Q plans (docs/pairs.md sections 3, 4, 6, 7, 11).

check(mc, src, tables, plan) -> [(index, message)]

src: decoded sm_75 instructions.
tables: {byte_offset_of_BRX: [byte_offsets]}, already parsed.
plan: dict with 'loads', 'sites' (with cx, cy), 'conversions', 'share'.

Assumes but does not check: sequences, model.py arrangement, register allocations for E(Rd),
scheduling, that aligned instructions run with whole warps, that rows are aligned, that the
function is entered at its start only.

Implements the predicate-hiding clause of section 3 for 'also': a write of the register
under a predicate hides the load from readers under the same predicate (same register,
same polarity) until that predicate register is written.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qflow import (CFG, reaching_definitions, ENTRY, CONFLICT,
                   is_unguarded, is_guarded, guard_predicate, writes_zero)


def check(mc, src, tables, plan):
    n = len(src)
    errors = []
    def err(idx, msg):
        errors.append((idx, msg))

    cfg = CFG(src, tables)
    cfg.build(mc)
    if cfg.error:
        err(cfg.error.index, 'CFG: %s' % cfg.error.reason)
        return errors

    loads = plan.get('loads', {})
    sites = plan.get('sites', {})
    conversions = plan.get('conversions', [])
    share = plan.get('share', {})
    roots = plan.get('roots', {})
    made_before = plan.get('made_before', {})
    made_after = plan.get('made_after', {})
    through = plan.get('through', {})

    # Refuse sites/loads inside a callee's body.
    # Use the callee bodies computed by _compute_called (which stops at RET and callee calls).
    callee_body = set()
    if cfg.has_calls:
        for call_i, target in cfg.call_targets.items():
            if target is None: continue
            c = cfg.called.get(call_i)
            if c is None:
                # External or complex: cannot determine body, be conservative
                continue
            # Walk the callee body using only plain successors (stop at RET, calls)
            body, work = set(), [target]
            while work:
                k = work.pop()
                if k in body: continue
                body.add(k)
                # Stop at RET (the callee returns) or at another call
                if src[k].mnemonic == 'RET' or cfg.kind[k] == 'call':
                    continue
                for s in cfg.succ[k]:
                    # Don't follow edges to meeting places that are outside the callee
                    # (meeting places are reachable from BREAK but not part of the callee)
                    if s not in body:
                        work.append(s)
            callee_body |= body
    for i in sites:
        if i in callee_body:
            err(i, 'cannot check: site %d inside callee' % i)
    for i in loads:
        if i in callee_body:
            err(i, 'cannot check: load %d inside callee' % i)

    # Reaching definitions: watch load-destination registers AND sites' operand registers.
    # Without the operand registers, condition 4 reads an empty set and passes silently.
    watch = set()
    for li, linfo in loads.items():
        for letter, reg in src[li].writes:
            if letter == 'R': watch.add(reg)
    for si, site in sites.items():
        d_si = src[si]
        watch.add(d_si.ops['Ra'])
        watch.add(d_si.ops['Ra'] + 1)
        watch.add(d_si.ops['Rb'])
        # B pair registers
        rdb, p_ = site['rb'], site['p']
        watch.add(rdb + 2 * p_)
        watch.add(rdb + 2 * p_ + 1)
    reach = reaching_definitions(cfg, watch) if watch else [None] * n

    # Bug 1a: refuse sites/loads at unreached instructions (outside callee body)
    for i in sites:
        if i not in callee_body and reach[i] is None:
            err(i, 'cannot check: site %d not reached from entry' % i)
    for i in loads:
        if i not in callee_body and reach[i] is None:
            err(i, 'cannot check: load %d not reached from entry' % i)

    # Index conversions
    conv_before, conv_after = {}, {}
    for entry in conversions:
        idx, pos, reg, target, guard = entry
        (conv_before if pos == 'before' else conv_after).setdefault(idx, []).append((reg, target, guard))

    # -------------------------------------------------------------------
    # Helpers for cx/cy
    # -------------------------------------------------------------------
    def acc_regs(c_field):
        """The four registers of an accumulator field, or empty set for None (zero)."""
        if c_field is None: return set()
        return set(range(c_field, c_field + 4))

    # -------------------------------------------------------------------
    # Section 3: pairs
    # -------------------------------------------------------------------
    for i, site in sites.items():
        d_i = src[i]
        half = site['half']
        j = site['partner']
        dx, dy = site['x'], site['y']
        cx, cy = site.get('cx'), site.get('cy')
        rda, a_ = site['ra'], site['a']
        rdb, p_ = site['rb'], site['p']

        if d_i.c.name != 'hmma_x8_':
            err(i, 'site %d not HMMA' % i); continue
        if is_guarded(d_i):
            err(i, 'product %d under predicate' % i)
        if d_i.ops['Rd'] % 2 != 0:
            err(i, 'product %d odd Rd' % i)

        # Check Rc: must be Rd (in-place), cx/cy (other), or 255 (zero)
        rc_i = d_i.ops['Rc']
        rd_i = d_i.ops['Rd']
        if rc_i != rd_i and rc_i != 255:
            # Must match cx or cy
            pass  # checked below in cx/cy validation

        if j not in sites:
            err(i, 'partner %d not a site' % j); continue
        d_j = src[j]
        site_j = sites[j]
        if d_j.c.name != 'hmma_x8_':
            err(j, 'partner %d not HMMA' % j); continue

        if half == 1:
            # Bug 4: a half-1 site must have a half-0 partner that names it back and comes before
            if j not in sites or sites[j].get('half') != 0 or sites[j].get('partner') != i or j >= i:
                err(i, 'site %d half=1: partner %d is not a valid half=0 site before it' % (i, j))
            continue
        if half != 0:
            err(i, 'site %d: half=%s not 0 or 1' % (i, half)); continue
        if i >= j:
            err(i, 'pair: i=%d >= j=%d' % (i, j)); continue

        # Verify site_j matches site_i
        if site_j['partner'] != i:
            err(j, 'pair (%d,%d): j.partner=%d' % (i, j, site_j['partner']))
        if site_j['half'] != 1:
            err(j, 'pair (%d,%d): j.half=%d' % (i, j, site_j['half']))
        for key in ('x', 'y', 'cx', 'cy', 'ra', 'rb', 'p'):
            if site.get(key) != site_j.get(key):
                err(j, 'pair (%d,%d): j.%s=%s != i.%s=%s' % (i, j, key, site_j.get(key), key, site.get(key)))
        expected_a_j = (d_j.ops['Ra'] - rda) // 2 if (d_j.ops['Ra'] - rda) in (0, 2) else -1
        if site_j.get('a') != expected_a_j:
            err(j, 'pair (%d,%d): j.a=%s expected %d' % (i, j, site_j.get('a'), expected_a_j))
        if 'transposed' in site and 'transposed' in site_j and site['transposed'] != site_j['transposed']:
            err(j, 'pair (%d,%d): transposed mismatch' % (i, j))

        # Condition 1
        for k in range(i + 1, j + 1):
            if k in cfg.entries:
                err(i, 'pair (%d,%d): entry at %d' % (i, j, k)); break
        for k in range(i, j):
            if cfg.succ[k] != [k + 1]:
                err(i, 'pair (%d,%d): transfer at %d' % (i, j, k)); break

        # Condition 2
        if d_i.ops['Ra'] != d_j.ops['Ra']:
            err(i, 'pair (%d,%d): Ra differs' % (i, j))
        if d_i.ops['Ra'] != rda + 2 * a_:
            err(i, 'pair (%d,%d): Ra=%d expected %d' % (i, j, d_i.ops['Ra'], rda + 2 * a_))
        bi, bj = d_i.ops['Rb'], d_j.ops['Rb']
        eb0, eb1 = rdb + 2 * p_, rdb + 2 * p_ + 1
        if not ((bi == eb0 and bj == eb1) or (bi == eb1 and bj == eb0)):
            err(i, 'pair (%d,%d): B regs (%d,%d) not (%d,%d)' % (i, j, bi, bj, eb0, eb1))

        # x/y: x has the even B, y the odd
        if bi == eb0: x_idx, y_idx = i, j
        else: x_idx, y_idx = j, i
        if dx != src[x_idx].ops['Rd']:
            err(i, 'pair (%d,%d): Dx mismatch' % (i, j))
        if dy != src[y_idx].ops['Rd']:
            err(i, 'pair (%d,%d): Dy mismatch' % (i, j))

        # cx/cy: what each adds to
        # x's Rc must be dx (in-place), cx (other), or 255 (zero)
        x_rc = src[x_idx].ops['Rc']
        if cx is None:
            if x_rc != 255:
                err(i, 'pair (%d,%d): cx=None but x.Rc=%d (expected RZ=255)' % (i, j, x_rc))
        elif cx == dx:
            if x_rc != dx:
                err(i, 'pair (%d,%d): cx=dx=%d but x.Rc=%d' % (i, j, dx, x_rc))
        else:
            if x_rc != cx:
                err(i, 'pair (%d,%d): cx=%d but x.Rc=%d' % (i, j, cx, x_rc))

        y_rc = src[y_idx].ops['Rc']
        if cy is None:
            if y_rc != 255:
                err(i, 'pair (%d,%d): cy=None but y.Rc=%d' % (i, j, y_rc))
        elif cy == dy:
            if y_rc != dy:
                err(i, 'pair (%d,%d): cy=dy=%d but y.Rc=%d' % (i, j, dy, y_rc))
        else:
            if y_rc != cy:
                err(i, 'pair (%d,%d): cy=%d but y.Rc=%d' % (i, j, cy, y_rc))

        # Condition 3: eight destination registers distinct
        dest = set(range(dx, dx + 4)) | set(range(dy, dy + 4))
        if len(dest) != 8:
            err(i, 'pair (%d,%d): dest not distinct' % (i, j))
        # None is Ra, Ra+1, or B pair
        forbidden = {d_i.ops['Ra'], d_i.ops['Ra'] + 1, eb0, eb1}
        if dest & forbidden:
            err(i, 'pair (%d,%d): dest in forbidden %s' % (i, j, sorted(dest & forbidden)))
        # The accumulator each reads: its dest (in-place), or cx/cy (must not share dest regs), or zero
        for c_field, c_name in [(cx, 'cx'), (cy, 'cy')]:
            if c_field is not None and c_field not in (dx, dy):
                c_regs = set(range(c_field, c_field + 4))
                if c_regs & dest:
                    err(i, 'pair (%d,%d): %s=%d overlaps dest' % (i, j, c_name, c_field))

        # Condition 4 at i and j (unchanged from before)
        a_regs_i = (d_i.ops['Ra'], d_i.ops['Ra'] + 1)
        b_pair_regs = (eb0, eb1)
        if reach[i] is not None:
            for r in a_regs_i + b_pair_regs:
                defs = reach[i].get(r, frozenset())
                if ENTRY in defs:
                    err(i, 'pair (%d,%d): R%d at %d has ENTRY' % (i, j, r, i))
                for di in defs:
                    if di == ENTRY: continue
                    if di not in loads:
                        err(i, 'pair (%d,%d): R%d at %d from %d not a load' % (i, j, r, i, di))
                    else:
                        role = loads[di]['role']
                        if r in a_regs_i:
                            if role != 'A': err(i, 'pair (%d,%d): R%d at %d from %d role %s not A' % (i, j, r, i, di, role))
                            if loads[di]['rd'] != rda: err(i, 'pair (%d,%d): R%d at %d from load %d Rd=%d' % (i, j, r, i, di, loads[di]['rd']))
                        else:
                            if role not in ('B', 'BT'): err(i, 'pair (%d,%d): B R%d at %d from %d role %s' % (i, j, r, i, di, role))
                            if loads[di]['rd'] != rdb: err(i, 'pair (%d,%d): B R%d at %d Rd=%d' % (i, j, r, i, loads[di]['rd']))
            b_roles = set()
            for r in b_pair_regs:
                for di in reach[i].get(r, frozenset()):
                    if di != ENTRY and di in loads: b_roles.add(loads[di]['role'])
            if 'B' in b_roles and 'BT' in b_roles:
                err(i, 'pair (%d,%d): B loads mix plain/transposed' % (i, j))
            if rda == rdb:
                err(i, 'pair (%d,%d): RdA==RdB' % (i, j))

            # Bug 5: transposed field must match B load roles
            if 'transposed' in site:
                expected_bt = bool(site['transposed'])
                actual_bt = 'BT' in b_roles and 'B' not in b_roles
                actual_plain = 'B' in b_roles and 'BT' not in b_roles
                if expected_bt and not actual_bt:
                    err(i, 'pair (%d,%d): transposed=True but B loads are plain' % (i, j))
                if not expected_bt and not actual_plain and b_roles:
                    err(i, 'pair (%d,%d): transposed=False but B loads are transposed' % (i, j))

        # Condition 4 at j
        j_b_reg = d_j.ops['Rb']
        j_read_regs = (d_j.ops['Ra'], d_j.ops['Ra'] + 1, j_b_reg)
        if reach[j] is not None:
            for r in j_read_regs:
                defs = reach[j].get(r, frozenset())
                if ENTRY in defs:
                    err(j, 'pair (%d,%d): R%d at j=%d has ENTRY' % (i, j, r, j))
                for di in defs:
                    if di == ENTRY: continue
                    if di not in loads:
                        err(j, 'pair (%d,%d): R%d at j=%d from %d not a load' % (i, j, r, j, di))

        # Condition 5: between i and j, no read/write of dest registers,
        # no write of j's accumulator, no matrix load with dest RdA or RdB.
        # j's accumulator is what the product at j reads: j's Rc registers (or zero).
        j_acc = acc_regs(site_j.get('cx') if y_idx == j else site_j.get('cy') if x_idx == j else None)
        # Actually j's accumulator: the Rc of j. If Rc == Rd: in-place (dest regs). If Rc == 255: zero.
        # Otherwise: cx or cy for the j-side product.
        rc_j = d_j.ops['Rc']
        if rc_j == 255:
            j_acc_between = set()  # zero: no registers to protect
        elif rc_j == d_j.ops['Rd']:
            j_acc_between = set(range(rc_j, rc_j + 4))  # already in dest
        else:
            j_acc_between = set(range(rc_j, rc_j + 4))

        for k in range(i + 1, j):
            dk = src[k]
            for l, r, *_ in dk.reads:
                if l == 'R' and r in dest:
                    err(k, 'pair (%d,%d): %d reads dest R%d' % (i, j, k, r))
            for l, r in dk.writes:
                if l == 'R' and r in dest:
                    err(k, 'pair (%d,%d): %d writes dest R%d' % (i, j, k, r))
                if l == 'R' and r in j_acc_between:
                    err(k, 'pair (%d,%d): %d writes j-acc R%d' % (i, j, k, r))
            if dk.c.name.startswith('ldsm'):
                lrd = dk.ops['Rd']
                if lrd == rda or lrd == rdb:
                    err(k, 'pair (%d,%d): ldsm at %d Rd=%d' % (i, j, k, lrd))

    # -------------------------------------------------------------------
    # Section 3: loads
    # -------------------------------------------------------------------
    # escapes: does the load's result, in Turing's layout, reach a non-paired reader on
    # some way? A way: along the instructions; a write of the register under no predicate
    # ends it; a write under a predicate hides the load from readers under the SAME
    # predicate (same register, same polarity) until that predicate register is written
    # (any write of P, guarded or not). Each of the load's four registers is tracked
    # separately.
    def _guard_of(d):
        """The (predicate_register, negated) of an instruction's guard, or None if unguarded,
        or 'uniform' if guarded by a uniform predicate only."""
        if is_unguarded(d): return None
        ops = d.ops
        if 'UPg' in ops and ops['UPg'] != 7:
            if ops.get('Pg', 7) == 7:
                return 'uniform'
        return (ops.get('Pg', 7), ops.get('Pg@not', 0))

    def _reads_reg(d, r):
        return any(l == 'R' and rr == r for l, rr, *_ in d.reads)

    def _writes_reg(d, r):
        return any(l == 'R' and rr == r for l, rr in d.writes)

    def _writes_reg_all(d, r):
        """Writes r in every lane (unguarded)."""
        return is_unguarded(d) and _writes_reg(d, r)

    def load_escapes(load_idx, rd, site_set):
        """Whether the load at load_idx (destination rd..rd+3) has a result that reaches,
        on some way, a non-paired reader. Returns (escapes: bool, readers: list of (reg, instruction))."""
        readers = []
        for r in range(rd, rd + 4):
            # Forward walk from the load's successors.
            # State: (instruction_index, hidden_predicates) where hidden_predicates is a
            # frozenset of (predicate_register, polarity) pairs under which a guarded write
            # of r has been seen since the load.
            seen = set()
            work = [(s, frozenset()) for s in cfg.succ[load_idx]]
            while work:
                at = work.pop()
                if at in seen: continue
                seen.add(at)
                i, hidden = at
                d = src[i]
                guard = _guard_of(d)

                # Does this instruction read r? If so, is it hidden?
                if _reads_reg(d, r) and i not in site_set and guard not in hidden:
                    # A call that may read r counts too
                    readers.append((r, i))

                # Does this instruction read r through a call?
                if cfg.kind[i] == 'call' and cfg.call_reads(i, r) and i not in site_set:
                    if guard not in hidden:
                        readers.append((r, i))

                # Does this instruction write r?
                if _writes_reg(d, r) or (cfg.kind[i] == 'call' and cfg.call_writes(i, r)):
                    if _writes_reg_all(d, r):
                        continue  # unguarded write ends the way
                    # Guarded write: hides the load from readers under the same predicate
                    if guard not in (None, 'uniform') and guard is not None:
                        hidden = hidden | {guard}

                # Does this instruction write a predicate that unhides something?
                pred_written = {x for l, x in d.writes if l == 'P'}
                # (R2P writes the predicates of its mask, which are not among its operands: all
                # of them when the mask is not a constant.)
                if d.mnemonic == 'R2P':
                    pred_written |= {q for q in range(7) if d.ops['Sb'] >> q & 1} if d.c.name == 'r2p__RIR' else set(range(7))
                if pred_written:
                    hidden = frozenset(g for g in hidden if g != 'uniform' and g[0] not in pred_written)
                # (A called routine may write any predicate: after a call nothing is hidden.)
                if cfg.kind[i] == 'call':
                    hidden = frozenset()

                work.extend((s, hidden) for s in cfg.succ[i])
        return (len(readers) > 0, readers)

    for load_idx, load_info in loads.items():
        dl = src[load_idx]
        if not dl.c.name.startswith('ldsm'):
            err(load_idx, 'load %d not LDSM' % load_idx); continue
        if (1 << dl.ops.get('num', 0)) != 4:
            err(load_idx, 'load %d not .4' % load_idx)
        if is_guarded(dl):
            err(load_idx, 'load %d under predicate' % load_idx)
        if dl.ops.get('sz', 0) != 0:
            err(load_idx, 'load %d sz!=0' % load_idx)
        rd = dl.ops['Rd']
        if rd % 4 != 0:
            err(load_idx, 'load %d Rd not mod 4' % load_idx)
        if load_info['rd'] != rd:
            err(load_idx, 'load %d plan rd=%d vs Rd=%d' % (load_idx, load_info['rd'], rd))
        mode = dl.ops.get('mode', 0)
        role = load_info['role']
        if role == 'BT' and mode != 1:
            err(load_idx, 'load %d role BT but mode=%d' % (load_idx, mode))
        if role in ('A', 'B') and mode != 0:
            err(load_idx, 'load %d role %s but mode=%d' % (load_idx, role, mode))

        # Check 'also': if NOT also, the load's result must not escape to non-paired readers
        site_set = set(sites.keys())
        if not load_info.get('also', False):
            escapes, readers = load_escapes(load_idx, rd, site_set)
            for reg, ui in readers:
                err(load_idx, 'load %d (not also): R%d reaches non-paired %d (%s)' % (load_idx, reg, ui, src[ui].mnemonic))

    # -------------------------------------------------------------------
    # Section 6: register states
    # -------------------------------------------------------------------
    # Tracked registers: all destination registers plus cx/cy registers
    tracked = set()
    for idx, site in sites.items():
        for off in range(4):
            tracked.add(site['x'] + off)
            tracked.add(site['y'] + off)
        for c in ('cx', 'cy'):
            cv = site.get(c)
            if cv is not None:
                for off in range(4):
                    tracked.add(cv + off)

    if tracked:
        def merge_st(a, b):
            if a is None: return b
            if b is None: return a
            if a == CONFLICT or b == CONFLICT: return CONFLICT
            if a == b: return a
            if a == 'Z': return b
            if b == 'Z': return a
            return CONFLICT

        state_in = [None] * n
        state_in[0] = {r: 'T' for r in tracked}

        def apply_conv_before(i, state):
            st = dict(state)
            for r, tgt, guard in conv_before.get(i, []):
                if r in tracked: st[r] = tgt
            return st

        def apply_conv_after(i, state):
            st = dict(state)
            for r, tgt, guard in conv_after.get(i, []):
                if r in tracked: st[r] = tgt
            return st

        def apply_inst(i, state):
            st = dict(state)
            d = src[i]
            ug = is_unguarded(d)

            if i in sites:
                s = sites[i]
                # A paired product writes its destination registers -> Q
                dest_regs_i = set(range(s['x'], s['x'] + 4)) | set(range(s['y'], s['y'] + 4))
                for r in dest_regs_i:
                    if r in tracked:
                        st[r] = 'Q'
            else:
                for l, r in d.writes:
                    if l == 'R' and r in tracked:
                        if writes_zero(d) and ug: st[r] = 'Z'
                        elif ug: st[r] = 'T'
                        else: st[r] = merge_st(st.get(r, 'T'), 'T')
                if cfg.kind[i] == 'call':
                    for r in tracked:
                        if cfg.call_writes(i, r):
                            st[r] = merge_st(st.get(r, 'T'), 'T')
            return st

        work = [0]
        in_work = [False] * n
        in_work[0] = True
        while work:
            i = work.pop()
            in_work[i] = False
            cur = state_in[i]
            if cur is None: continue
            abc = apply_conv_before(i, cur)
            ai = apply_inst(i, abc)
            aac = apply_conv_after(i, ai)
            for j in cfg.succ[i]:
                arriving = aac if j == i + 1 else ai
                old = state_in[j]
                if old is None:
                    state_in[j] = dict(arriving)
                    if not in_work[j]: work.append(j); in_work[j] = True
                else:
                    changed = False
                    for r in tracked:
                        o = old.get(r)
                        a = arriving.get(r)
                        m = merge_st(o, a)
                        if m != o: old[r] = m; changed = True
                    if changed and not in_work[j]: work.append(j); in_work[j] = True

        # Check requirements
        for i in range(n):
            if state_in[i] is None: continue
            d = src[i]
            state = apply_conv_before(i, state_in[i])

            if i in sites:
                s = sites[i]
                # Section 6: what does this product require Q or Z?
                # First product (half=0): the accumulators BOTH products add to.
                # Second product (half=1): the eight destination registers.
                dest_regs = set(range(s['x'], s['x'] + 4)) | set(range(s['y'], s['y'] + 4))
                if s['half'] == 0:
                    # First product reads accumulators of both: cx and cy
                    # cx: if None (zero), no requirement. If == dx (in-place), requires dest regs.
                    # Otherwise: cx registers.
                    required_q = set()
                    for c_field in (s.get('cx'), s.get('cy')):
                        if c_field is not None:
                            required_q |= set(range(c_field, c_field + 4))
                else:
                    # Second product: requires the eight destination registers Q or Z
                    required_q = dest_regs

                for r in required_q:
                    if r in tracked:
                        st = state.get(r)
                        if st is not None and st not in ('Q', 'Z'):
                            err(i, 'product %d: R%d state %s, need Q/Z' % (i, r, st))
            else:
                for l, r, *_ in d.reads:
                    if l == 'R' and r in tracked:
                        st = state.get(r)
                        if st is not None and st not in ('T', 'Z'):
                            err(i, '%s at %d: R%d state %s, need T/Z' % (d.mnemonic, i, r, st))
                if is_guarded(d):
                    for l, r in d.writes:
                        if l == 'R' and r in tracked:
                            st = state.get(r)
                            if st is not None and st not in ('T', 'Z'):
                                err(i, '%s at %d writes R%d under pred, state %s' % (d.mnemonic, i, r, st))
                if cfg.kind[i] == 'call' or d.mnemonic == 'RET':
                    for r in tracked:
                        st = state.get(r)
                        if st is not None and st not in ('T', 'Z'):
                            err(i, '%s at %d: R%d state %s, need T/Z' % (d.mnemonic, i, r, st))

        # Check CONFLICT at uses
        for i in range(n):
            if state_in[i] is None: continue
            state = apply_conv_before(i, state_in[i])
            d = src[i]
            for r, st in state.items():
                if st == CONFLICT:
                    reads_it = any(l == 'R' and rr == r for l, rr, *_ in d.reads)
                    writes_g = is_guarded(d) and any(l == 'R' and rr == r for l, rr in d.writes)
                    # (A product of a pair is held to what it reads in the translation, above:
                    # a register it only writes, or does not touch, may be in any state: a
                    # loop that the compiler unrolled has destinations that are no accumulator
                    # of their pair and arrive T one way and Q another.)
                    if (reads_it or writes_g) and i not in sites:
                        err(i, 'R%d at %d CONFLICT' % (r, i))

    # -------------------------------------------------------------------
    # Section 7: conversions
    # -------------------------------------------------------------------
    # Bug 3b: at most one conversion of a register at a place in the same direction
    conv_seen = set()
    for entry in conversions:
        idx, pos, reg, target, guard = entry
        key = (idx, pos, reg, target)
        if key in conv_seen:
            err(idx, 'conv R%d to %s %s %d: duplicate' % (reg, target, pos, idx))
        conv_seen.add(key)

        if pos == 'before':
            if not cfg.whole[idx]:
                err(idx, 'conv R%d to %s before %d: not whole' % (reg, target, idx))
        elif pos == 'after':
            if not cfg._way_whole(idx, cfg.whole):
                err(idx, 'conv R%d to %s after %d: edge not whole' % (reg, target, idx))

        # Bug 3a: conversion of untracked register or at unreached instruction
        if reg not in tracked:
            err(idx, 'conv R%d to %s at %d: register not tracked (not an accumulator)' % (reg, target, idx))
        elif tracked:
            if pos == 'before':
                si = state_in[idx]
            else:
                si = state_in[idx]
            if si is None:
                err(idx, 'conv R%d to %s at %d: instruction not reached' % (reg, target, idx))
            else:
                if pos == 'before':
                    st = si.get(reg)
                else:
                    sb = apply_conv_before(idx, si)
                    sa = apply_inst(idx, sb)
                    st = sa.get(reg)
                if st is not None:
                    if target == 'Q' and st not in ('T', 'Z'):
                        err(idx, 'conv to Q of R%d at %d: state %s' % (reg, idx, st))
                    if target == 'T' and st not in ('Q', 'Z'):
                        err(idx, 'conv to T of R%d at %d: state %s' % (reg, idx, st))

        if guard is not None:
            pred, negated = guard
            d = src[idx]
            if pos != 'before':
                err(idx, 'guarded conv R%d at %d: after, not before' % (reg, idx))
            elif d.mnemonic != 'BRA':
                err(idx, 'guarded conv R%d at %d: not BRA' % (reg, idx))
            else:
                bg = guard_predicate(d)
                if bg is None:
                    err(idx, 'guarded conv R%d at %d: BRA not conditional' % (reg, idx))
                elif bg[1] != pred:
                    err(idx, 'guarded conv R%d: guard P%d != branch P%d' % (reg, pred, bg[1]))

    # -------------------------------------------------------------------
    # Section 4: sharing
    # -------------------------------------------------------------------
    if share:
        groups = {}
        for rd, base in share.items():
            groups.setdefault(base, set()).add(rd)
        also_dests = {v['rd'] for v in loads.values() if v.get('also')}

        for base, members in groups.items():
            if len(members) <= 1: continue
            group_loads = {li: linfo['rd'] for li, linfo in loads.items() if linfo['rd'] in members}
            products_check = []
            for si, site in sites.items():
                for rd_key in ('ra', 'rb'):
                    rd = site[rd_key]
                    if rd not in members: continue
                    reads_trans = site['half'] == 1 or rd in also_dests
                    if reads_trans: products_check.append((si, rd))
            if not products_check: continue

            last_loaded = [None] * n
            last_loaded[0] = 'none'
            work = [0]; in_work = [False] * n; in_work[0] = True
            def merge_last(a, b):
                if a is None: return b
                if b is None: return a
                return a if a == b else 'mixed'
            while work:
                i = work.pop(); in_work[i] = False
                cur = last_loaded[i]
                if cur is None: continue
                out = group_loads[i] if i in group_loads else cur
                for j in cfg.succ[i]:
                    m = merge_last(last_loaded[j], out)
                    if m != last_loaded[j]:
                        last_loaded[j] = m
                        if not in_work[j]: work.append(j); in_work[j] = True
            for pi, dest in products_check:
                ll = last_loaded[pi]
                if ll is None or ll == 'none':
                    err(pi, 'share: product %d needs Rd=%d but no load of group' % (pi, dest))
                elif ll == 'mixed':
                    err(pi, 'share: product %d needs Rd=%d but mixed' % (pi, dest))
                elif ll != dest:
                    err(pi, 'share: product %d needs Rd=%d but last load Rd=%d' % (pi, dest, ll))

    # -------------------------------------------------------------------
    # Section 12: companions
    # -------------------------------------------------------------------
    if roots or made_before or made_after or through:
        reached = [False] * n
        if n:
            rw = [0]
            reached[0] = True
            while rw:
                k = rw.pop()
                for s in cfg.succ[k]:
                    if not reached[s]:
                        reached[s] = True
                        rw.append(s)

        for idx, root_list in made_before.items():
            if not (0 <= idx < n):
                err(idx, 'companion before %d: out of range' % idx)
                continue
            if idx in callee_body:
                err(idx, 'cannot check: companion before %d inside callee' % idx)
            if not reached[idx]:
                err(idx, 'companion before %d: instruction not reached' % idx)
            if not cfg.whole[idx]:
                for root in root_list:
                    err(idx, 'companion of R%d before %d: not whole' % (root, idx))
            for root in root_list:
                if root not in roots:
                    err(idx, 'companion of R%d before %d: root not in plan' % (root, idx))

        for idx, root_list in made_after.items():
            if not (0 <= idx < n):
                err(idx, 'companion after %d: out of range' % idx)
                continue
            if idx in callee_body:
                err(idx, 'cannot check: companion after %d inside callee' % idx)
            if not reached[idx]:
                err(idx, 'companion after %d: instruction not reached' % idx)
            if not cfg._way_whole(idx, cfg.whole):
                for root in root_list:
                    err(idx, 'companion of R%d after %d: edge not whole' % (root, idx))
            for root in root_list:
                if root not in roots:
                    err(idx, 'companion of R%d after %d: root not in plan' % (root, idx))

        for root in sorted(roots):
            if root == 255 or root in tracked:
                err(None, 'companion root R%d: invalid or accumulator register' % root)

        mirrored = {'LOP3', 'IMAD', 'LEA', 'IADD3', 'SHF', 'MOV'}
        through_watch = {src[li].ops.get('Ra') for li in through if 0 <= li < n and 'Ra' in src[li].ops}
        through_watch.discard(None)
        through_reach = reaching_definitions(cfg, through_watch) if through_watch else [None] * n
        for li, (root, made_idx) in sorted(through.items()):
            if li not in loads:
                err(li, 'companion load %d: not a whole-row load' % li)
                continue
            if root not in roots or loads[li]['role'] not in roots[root]:
                err(li, 'companion load %d: root R%d lacks role %s' % (li, root, loads[li]['role']))
            ra = src[li].ops.get('Ra')
            if made_idx is None:
                if ra != root:
                    err(li, 'companion load %d: direct root R%d != Ra R%s' % (li, root, ra))
            else:
                if not (0 <= made_idx < n):
                    err(li, 'companion load %d: made %d out of range' % (li, made_idx))
                    continue
                defs = through_reach[li].get(ra, frozenset()) if through_reach[li] is not None else frozenset([ENTRY])
                if defs != frozenset([made_idx]):
                    err(li, 'companion load %d: Ra R%d reaching defs %s != [%d]' % (li, ra, sorted(defs), made_idx))
                dm = src[made_idx]
                reads = {(l, x) for l, x, *_ in dm.reads}
                if (dm.mnemonic not in mirrored or not is_unguarded(dm) or cfg.kind[made_idx] != 'plain'
                        or dm.writes != [('R', ra)] or reads != {('R', root)}):
                    err(li, 'companion load %d: made %d is not a plain mirrored op from R%d to R%d' % (li, made_idx, root, ra))
                ahead, work = set(), [s for s in cfg.succ[made_idx] if s != made_idx]
                while work:
                    k = work.pop()
                    if k in ahead or k == made_idx:
                        continue
                    ahead.add(k)
                    work.extend(cfg.succ[k])
                behind, work = set(), [p for p in cfg.pred[li] if p != made_idx]
                while work:
                    k = work.pop()
                    if k in behind or k == made_idx:
                        continue
                    behind.add(k)
                    work.extend(cfg.pred[k])
                for k in ahead & behind:
                    if _writes_reg(src[k], root) or (cfg.kind[k] == 'call' and cfg.call_writes(k, root)):
                        err(li, 'companion load %d: root R%d written at %d between %d and load' % (li, root, k, made_idx))
                        break

        for root in sorted(roots):
            before_r = {i for i, rs in made_before.items() if root in rs}
            after_r = {i for i, rs in made_after.items() if root in rs}
            write_r = {i for i in range(n)
                       if _writes_reg(src[i], root) or (cfg.kind[i] == 'call' and cfg.call_writes(i, root))}
            bare_entry = [i == 0 for i in range(n)]
            bare_body = [bare_entry[i] and i not in before_r for i in range(n)]
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if not reached[i]:
                        continue
                    bb = bare_entry[i] and i not in before_r
                    if bb != bare_body[i]:
                        bare_body[i] = bb
                        changed = True
                    after_inst = bare_body[i] or (i in write_r)
                    for s in cfg.succ[i]:
                        eb = False if (s == i + 1 and i in after_r) else after_inst
                        if eb and not bare_entry[s]:
                            bare_entry[s] = True
                            changed = True
            for li, (r_li, _) in sorted(through.items()):
                if r_li == root and 0 <= li < n and bare_body[li]:
                    err(li, 'companion of R%d does not stand on every way to load %d' % (root, li))

    return errors
