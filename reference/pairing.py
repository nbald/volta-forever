#!/usr/bin/env python3
"""Which products of a function the translator does as pairs on whole rows, which matrix loads it
does as whole rows, and where it converts the accumulator registers: its plan, by the design of
docs/pairs.md (sections 3, 6 and 7 are the rules; section 11 is the plan's format).

This is the translator's side: it chooses. A plan is checked on its own by other code that does
not read this (the validator; qcheck.py).

    pairing.py FILE.cubin...        what would be planned for each function with products
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOADS = ('ldsm__sImmOffset', 'ldsm__URsIR')
PRODUCT = 'hmma_x8_'
# Mnemonics that park lanes, take them out, or bring them back: nothing is known to be whole across them.
LANES = {'BSYNC', 'BREAK', 'KILL', 'PBK', 'PCNT', 'PEXIT', 'PLONGJMP', 'PRET', 'SSY', 'SYNC', 'BRK', 'CONT', 'LONGJMP', 'RTT'}
LEAVES = {'BREAK', 'BRK', 'CONT', 'PEXIT', 'PLONGJMP', 'PRET', 'LONGJMP'}     # of those, the ones after which lanes go on elsewhere
SAY = bool(os.environ.get('VF_PAIRS_SAY'))
# Whether the addresses of whole-row loads are taken from kept, shuffled copies of the registers
# they are made from, and paired HMMA steps use per-group scoreboards and native sm_70 scheduling
# (docs/pairs.md, sections 8 and 12): opt-in (`VF_COMPANIONS=1`), off by default.
COMPANIONS = os.environ.get('VF_COMPANIONS', '0') == '1'
ROLES = ('A', 'B', 'BT')
ROLE_ROWS = {'A': 2, 'B': 2, 'BT': 4}             # how many rows a load of each role takes: a companion for each
# What may be done again on a companion: arithmetic of one register and constants, lane by lane.
MIRRORED = {'LOP3', 'IMAD', 'LEA', 'IADD3', 'SHF', 'MOV'}
DEAD, WANTS_Q, WANTS_T = 0, 1, 2          # what a register is next used as (ordered: the join is the larger)
Z, T, Q = 'Z', 'T', 'Q'


def true(ops, p):
    return ops.get(p, 7) == 7 and not ops.get(p + '@not', 0)


def unguarded(d):
    return true(d.ops, 'Pg') and true(d.ops, 'UPg')


def written_predicates(d):
    """The predicates an instruction may write: those its operands say, and for R2P those of its
    mask (every one, when the mask is not a constant): they are not among its operands. (Left
    out until 2026-11-18: a guard that R2P changed still hid a load from the readers under it.)"""
    out = {x for l, x in d.writes if l == 'P'}
    if d.mnemonic == 'R2P':
        out |= {p for p in range(7) if d.ops['Sb'] >> p & 1} if d.c.name == 'r2p__RIR' else set(range(7))
    return out


class Flow:
    """The ways through a function: what the plan needs of them."""

    def __init__(self, f):
        import translate as TR
        self.f, self.src, self.n = f, f.src, len(f.src)
        src, n = self.src, self.n
        self.entered = f.entries()
        self.kind = [self.kind_of(i) for i in range(n)]
        self.known = self.entered is not None and 'unknown' not in self.kind
        if not self.known:
            return
        def named(i):
            rel = TR.relative(f.mc, 75, src[i])
            if rel is None:
                return None
            t = 16 * i + 16 + TR.signed(rel[0], src[i].word) * rel[1]
            return t // 16 if t % 16 == 0 and 0 <= t < 16 * n else None
        self.named = [named(i) for i in range(n)]
        # Where each instruction goes on. A meeting place is reached by branches and by the
        # instructions that leave a loop, not from the instruction that names it. A call within
        # the section goes on after it, with what its function reads and writes taken as read
        # and written by the call (`called`): the function's own instructions are reached from
        # nowhere here, and a return ends a way.
        meets = sorted({self.named[i] for i in range(n) if src[i].mnemonic in ('BSSY', 'PBK', 'SSY', 'PCNT') and self.named[i] is not None})
        self.succ = []
        for i in range(n):
            kind, after = self.kind[i], [i + 1] if i + 1 < n else []
            if kind == 'jump':
                out = [self.named[i]] if self.named[i] is not None else []
            elif kind == 'cond':
                out = after + ([self.named[i]] if self.named[i] is not None else [])
            elif kind == 'exit' or src[i].mnemonic == 'RET':
                out = [] if unguarded(src[i]) else after
            elif src[i].mnemonic == 'BRX':
                out = [t // 16 for t in f.tables[16 * i] if t % 16 == 0 and 0 <= t < 16 * n] + ([] if unguarded(src[i]) else after)
            elif src[i].mnemonic in LEAVES:
                out = after + meets
            else:
                out = after
            self.succ.append(sorted(set(out)))
        self.known = all(self.named[i] is not None for i in range(n) if self.kind[i] in ('jump', 'cond'))
        if not self.known:
            return
        self.pred = [[] for _ in range(n)]
        for i in range(n):
            for t in self.succ[i]:
                self.pred[t].append(i)
        # What a call reads and writes: everything its function does, and the functions that one calls.
        self.called = {}
        for i in range(n):
            if self.kind[i] != 'call':
                continue
            if not src[i].c.name.startswith('call_rel') or self.named[i] is None:
                self.called[i] = None                       # a function that is not here: anything
                continue
            body, work, simple = set(), [self.named[i]], True
            while work:
                k = work.pop()
                if k in body:
                    continue
                body.add(k)
                if self.kind[k] == 'call':
                    simple = False                          # a function that calls another: anything
                    break
                work += self.succ[k]
            if not simple:
                self.called[i] = None
                continue
            # Read by the function: what an instruction of it reads that it has not written before
            # on some way from its entry (the registers alive at its entry, a return leaving none).
            alive = {k: set() for k in body}
            changed = True
            while changed:
                changed = False
                for k in sorted(body, reverse=True):
                    after = set()
                    for t in self.succ[k]:
                        after |= alive[t]
                    kills = {x for l, x in src[k].writes if l == 'R'} if unguarded(src[k]) else set()
                    now = {x for l, x, *_ in src[k].reads if l == 'R'} | (after - kills)
                    if now != alive[k]:
                        alive[k], changed = now, True
            writes = {x for k in body for l, x in src[k].writes if l == 'R'}
            self.called[i] = (alive[self.named[i]], writes)
        self.aligned = [d.c.name in LOADS + (PRODUCT,) and unguarded(d) for d in src]
        self.whole_places()

    def reads(self, i, r):
        """Instruction i may read register r (a call: its function may)."""
        if self.kind[i] == 'call':
            return self.called[i] is None or r in self.called[i][0]
        return any(l == 'R' and x == r for l, x, *_ in self.src[i].reads)

    def writes(self, i, r):
        """Instruction i may write register r (a call: its function may)."""
        if self.kind[i] == 'call':
            return self.called[i] is None or r in self.called[i][1]
        return any(l == 'R' and x == r for l, x in self.src[i].writes)

    def all_of(self, i, r):
        """Instruction i writes register r in every lane that runs it: nothing of what r held is left."""
        return self.kind[i] != 'call' and unguarded(self.src[i]) and any(l == 'R' and x == r for l, x in self.src[i].writes)

    def kind_of(self, i):
        """plain: goes on to the next instruction and nowhere else. cond: a branch taken or not.
        jump: a branch always taken. call, exit (nothing follows), other: a transfer whose ways
        are in `succ` but before or after which nothing is converted. unknown: ways not known."""
        d, f = self.src[i], self.f
        m, name, ops = d.mnemonic, d.c.name, d.ops
        if m == 'BRA':
            # (BRA.U, a branch that all the lanes take alike, is taken always when it is under no
            # predicate: cuBLAS jumps over the other arm of a choice with it. BRA.DIV and BRA.CONV
            # are taken or not by what the lanes are doing.)
            return 'jump' if unguarded(d) and true(ops, 'Pp') and ops.get('cond', 0) in (0, 1) else 'cond'
        if m == 'BRX':
            return 'other' if name == 'brx_' and 16 * i in f.tables else 'unknown'
        if m in ('JMX', 'JMP'):
            return 'unknown'
        if m == 'CALL':
            return 'call'
        if m == 'EXIT':
            return 'exit' if unguarded(d) else 'other'
        if m == 'RET' or m in LANES or m in ('BSSY', 'BPT'):
            return 'other'
        return 'plain'

    def target(self, i):
        """Where the branch `i` leads when taken."""
        return self.named[i]

    def removes(self, i):
        """It may leave fewer lanes active after it than before (or more)."""
        d = self.src[i]
        return d.mnemonic in LANES or (d.mnemonic in ('EXIT', 'RET') and not unguarded(d))

    def whole_places(self):
        """`whole[i]`: whenever a lane runs instruction i, the 32 lanes of the warp run it together
        (docs/pairs.md, section 7). `uniform[i]`, for a conditional branch: all lanes go the same way."""
        n, kind = self.n, self.kind
        lead = [None] * n
        def leads(i):
            # Straight-line code from i reaches an aligned instruction.
            path = []
            while True:
                if i >= n or lead[i] is False or i in path:
                    result = False
                    break
                if lead[i] or self.aligned[i]:
                    result = True
                    break
                path.append(i)
                if kind[i] == 'plain' and not self.removes(i):
                    i += 1
                elif kind[i] == 'jump':
                    i = self.target(i)
                else:
                    result = False
                    break
            for k in path:
                lead[k] = result
            return result
        for i in range(n):
            lead[i] = leads(i)
        self.lead = lead
        self.uniform = [kind[i] == 'cond' and (lead[self.target(i)] or (i + 1 < n and lead[i + 1])) for i in range(n)]
        whole = list(lead)
        changed = True
        while changed:
            changed = False
            for b in range(n):
                if whole[b] or not self.pred[b]:
                    continue
                if all(self.goes_whole(p, whole) for p in self.pred[b]):
                    whole[b] = changed = True
        self.whole = whole

    def goes_whole(self, p, whole=None):
        """The lanes that leave instruction `p` by one of its ways are the 32 of the warp."""
        whole = self.whole if whole is None else whole
        if not whole[p] or self.removes(p):
            return False
        return self.kind[p] in ('plain', 'jump') or (self.kind[p] == 'cond' and self.uniform[p])

    def simple(self, i):
        """Instruction i is entered from the one before it and from nowhere else."""
        return i > 0 and self.pred[i] == [i - 1] and self.kind[i - 1] == 'plain' and i not in self.entered

    def escapes(self, load, rd, sites):
        """Whether what the load at `load` puts in registers rd to rd + 3 may be read by anything but
        a product of `sites`. Followed along the ways from the load: a write of the register under
        no predicate ends a way; a write under a predicate hides the load from the readers under the
        same predicate, until that predicate is written (lanes in which a guard holds at both ran both)."""
        src = self.src
        for r in range(rd, rd + 4):
            seen, work = set(), [(s, frozenset()) for s in self.succ[load]]
            while work:
                at = work.pop()
                if at in seen:
                    continue
                seen.add(at)
                i, hidden = at
                d = src[i]
                guard = None if unguarded(d) else (d.ops.get('Pg', 7), d.ops.get('Pg@not', 0)) if 'UPg' not in d.ops or true(d.ops, 'UPg') else 'uniform'
                if self.reads(i, r) and i not in sites and guard not in hidden:
                    return True
                if self.writes(i, r):
                    if self.all_of(i, r):
                        continue
                    if guard not in (None, 'uniform'):
                        hidden = hidden | {guard}
                written = written_predicates(d)
                if written:
                    hidden = frozenset(g for g in hidden if g[0] not in written)
                if self.kind[i] == 'call':
                    # (What a called routine writes of predicates is not known here: after a
                    # call nothing is hidden any more. Left out until 2026-11-18.)
                    hidden = frozenset()
                work += [(s, hidden) for s in self.succ[i]]
        return False

    def reaching(self, watch):
        """For each instruction, the definitions of the registers `watch` that reach it: {register:
        set of instructions}, with None in a set for what the function was entered with or what a
        call may have written."""
        src, n = self.src, self.n
        watch = sorted(watch)
        writes = [[r for r in watch if self.writes(i, r)] for i in range(n)]
        into = [None] * n
        into[0] = {r: frozenset([None]) for r in watch}
        work = [0]
        while work:
            i = work.pop()
            state = into[i]
            out = state
            if writes[i]:
                out = dict(state)
                for r in writes[i]:
                    if self.kind[i] == 'call':
                        out[r] = state[r] | {None}
                    else:
                        out[r] = frozenset([i]) if self.all_of(i, r) else state[r] | {i}
            for s in self.succ[i]:
                if into[s] is None:
                    into[s] = out
                    work.append(s)
                else:
                    merged = {r: into[s][r] | out[r] for r in watch} if into[s] is not out else into[s]
                    if merged != into[s]:
                        into[s] = merged
                        work.append(s)
        return into


def zero_writer(d):
    """An instruction, under no predicate, that writes zero in what it writes, as the compiler
    writes it in these kernels."""
    if not unguarded(d):
        return False
    if d.c.name == 'cs2r_':
        return d.ops.get('SRa') == 255
    if d.mnemonic == 'IMAD' and d.c.name.endswith('__RRR_RRR'):
        return d.ops.get('Ra') == 255 and d.ops.get('Rb') == 255 and d.ops.get('Rc') == 255
    return False


class Plan:
    """What `plan` decides for a function (docs/pairs.md, section 11), and what the translator emits from."""

    def __init__(self):
        self.loads, self.sites = {}, {}
        self.share = {}                        # {destination: the one whose registers it shares} (`sharing`)
        self.before, self.after = {}, {}       # {instruction: [(register, 'Q' or 'T')]}
        self.dropped = []                      # (instruction, why): products that are not done as pairs
        # Companions (`companions`): {root register: its roles, in the order of ROLES}; {load: (root,
        # None or the instruction that makes the load's address register from the root)}; where
        # the companions of roots are made: {instruction: [root]}, before it or after it.
        self.roots, self.through, self.made_before, self.made_after = {}, {}, {}, {}
        self.site_sb, self.site_seq_wait = {}, set()

    def conversions(self):
        return ([(i, 'before', r, to, None) for i, c in sorted(self.before.items()) for r, to in c]
                + [(i, 'after', r, to, None) for i, c in sorted(self.after.items()) for r, to in c])

    def rows(self):
        """{destination of whole-row loads: how many registers of the translation its rows take}:
        four, with the destination's own four, or all eight when the destination's registers
        also get what the load gives in Turing's layout."""
        out = {}
        for v in self.loads.values():
            out[v['rd']] = max(out.get(v['rd'], 4), 8 if v['also'] else 4)
        return out

    def shared(self):
        """{destination: the destination whose registers of the translation it shares}: `share`,
        when the function is alone in its group; each its own otherwise."""
        return dict(self.share)

    def companion_registers(self):
        """{root: how many registers of the translation its companions take}."""
        return {x: sum(ROLE_ROWS[r] for r in roles) for x, roles in self.roots.items()}

    def kept(self):
        """Which values are kept from the entry of the function."""
        out = set()
        for v in self.loads.values():
            out |= {'A': {'kA', 'kA2'}, 'B': {'kB', 'kB2'}, 'BT': {'kBT', 'cB'}}[v['role']]
        for c in list(self.before.values()) + list(self.after.values()):
            out |= {'pi' if to == Q else 'inv' for r, to in c}
        return out


def product_form(d):
    """The operands (Rd, Ra, Rb, Rc) of a product that may be half of a pair, or None. Rc, the
    accumulator it adds to: its destination (in place), four other registers apart from it, or
    None for zero."""
    ops = d.ops
    if d.c.name != PRODUCT or not unguarded(d):
        return None
    if ops.get('Ra@negate') or ops.get('Rb@negate') or ops.get('satfinite') or ops.get('size') or ops.get('altfmt') or ops['dstfmt'] != 1:
        return None
    rd, ra, rb, rc = ops['Rd'], ops['Ra'], ops['Rb'], ops['Rc']
    if 255 in (rd, ra, rb) or rd % 2 or ra % 2 or rd + 3 >= 255:
        return None
    if set(range(rd, rd + 4)) & {ra, ra + 1, rb}:
        return None
    if rc == 255:
        return rd, ra, rb, None
    if rc % 2 or rc + 3 >= 255 or (rc != rd and set(range(rc, rc + 4)) & set(range(rd, rd + 4))):
        return None
    return rd, ra, rb, rc


def load_form(d):
    """(Rd, transposed) of a matrix load that may be done as whole rows, or None."""
    ops = d.ops
    if d.c.name not in LOADS or not unguarded(d):
        return None
    if ops['sz'] != 0 or ops['num'] != 2 or ops['mode'] not in (0, 1):
        return None
    rd = ops['Rd']
    if rd == 255 or rd % 4 or rd + 3 >= 255:
        return None
    return rd, ops['mode'] == 1


def plan(f, with_companions=True):
    """The plan for function `f` (a translate.Function), or None when it has nothing to do as pairs."""
    src, n = f.src, len(f.src)
    if not any(d.c.name == PRODUCT for d in src):
        return None
    flow = Flow(f)
    if not flow.known:
        return None
    forms = {i: product_form(d) for i, d in enumerate(src) if d.c.name == PRODUCT}
    loads = {i: load_form(d) for i, d in enumerate(src) if d.c.name in LOADS}
    loads = {i: v for i, v in loads.items() if v}
    if not loads:
        return None
    watch = {r for rd, t in loads.values() for r in range(rd, rd + 4)}
    reach = flow.reaching(watch)
    excluded, result = set(), None
    for _ in range(len(forms) + 2):
        result = attempt(f, flow, forms, loads, reach, excluded)
        if result.bad is None:
            break
        excluded |= result.bad
    else:
        return None
    del result.bad
    if not result.sites:
        return None
    sharing(result, flow)
    if COMPANIONS:
        assign_scoreboards(result, f.src)
    else:
        result.site_seq_wait.clear()
    if COMPANIONS and with_companions:
        companions(result, flow, f)
    return result


def assign_scoreboards(plan, src):
    """Assign hardware scoreboards from {0..3} per interfering accumulator group (preferring
    scoreboards not used by the program or whole-row loads), keeping ARRIVES(4) and SEQUENCE(5)
    for whole-row address shuffles, un-barriered loads, conversions, and companions."""
    from machine import READ_BARRIER, WRITE_BARRIER
    sites = sorted(plan.sites)
    if not sites:
        return
    parent = {i: i for i in sites}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra
    acc_regs = {}
    wr_regs = {}
    src_regs = {}
    for i in sites:
        s = plan.sites[i]
        wr = {b + k for b in (s['x'], s['y']) for k in range(4)}
        acc = set(wr)
        for b in (s['cx'], s['cy']):
            if b is not None:
                acc.update(range(b, b + 4))
        a_base, b_base = s['ra'] + 2 * s['a'], s['rb'] + 2 * s['p']
        src_r = {a_base, a_base + 1, b_base, b_base + 1}
        wr_regs[i], acc_regs[i], src_regs[i] = wr, acc, src_r
    for idx, i in enumerate(sites):
        for j in sites[idx + 1:]:
            if (acc_regs[i] & acc_regs[j]) or (src_regs[i] & wr_regs[j]) or (src_regs[j] & wr_regs[i]):
                union(i, j)
    lds_used = {(src[j].word >> WRITE_BARRIER) & 7 for j in plan.loads} - {7}
    prog_used = {b for d in src for b in ((d.word >> WRITE_BARRIER) & 7, (d.word >> READ_BARRIER) & 7)} - {7}
    avail = (
        [sb for sb in (3, 2, 1, 0) if sb not in prog_used]
        + [sb for sb in (3, 2, 1, 0) if sb in prog_used and sb not in lds_used]
        + [sb for sb in (3, 2, 1, 0) if sb in prog_used and sb in lds_used]
    )
    rep_sb = {}
    for i in sites:
        rep = find(i)
        if rep not in rep_sb:
            rep_sb[rep] = avail[len(rep_sb) % len(avail)]
        plan.site_sb[i] = rep_sb[rep]


def sharing(plan, flow):
    """Which destinations of whole-row loads can have the same registers of the translation: those
    whose rows are never wanted at the same time. A destination's registers are read by the
    second product of each pair on it (and by the first when the destination's own registers
    are left to Turing's layout), and written by its loads: alive from a load to the last such
    product, along the ways. Two destinations cannot share when one is loaded while the other's
    rows are alive. (PyTorch's attention has two loops, each with four destinations, two rows in
    flight in each: eight destinations, four sets of registers.)"""
    n, size = flow.n, plan.rows()
    reads = {rd: set() for rd in size}
    for i, s in plan.sites.items():
        for rd in (s['ra'], s['rb']):
            if s['half'] or size[rd] == 8:
                reads[rd].add(i)
    writes = {rd: {j for j, v in plan.loads.items() if v['rd'] == rd} for rd in size}
    alive = {}
    for rd in size:
        out = [False] * n                           # alive after each instruction
        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                now = any(t in reads[rd] or (out[t] and t not in writes[rd]) for t in flow.succ[i])
                if now != out[i]:
                    out[i], changed = now, True
        alive[rd] = out
    clash = {rd: set() for rd in size}
    for a in size:
        for b in size:
            if a != b and any(alive[b][j] or j in reads[b] for j in writes[a]):
                clash[a].add(b)
                clash[b].add(a)
    classes = []
    for rd in sorted(size, key=lambda rd: (-size[rd], rd)):
        for members in classes:
            if not clash[rd] & set(members):
                members.append(rd)
                break
        else:
            classes.append([rd])
    plan.share = {rd: members[0] for members in classes for rd in members}


def companions(plan, flow, f):
    """Which whole-row loads take their rows' addresses from companions, of which registers, and
    where the companions are made (docs/pairs.md, section 12).

    A load's row is at the address that another lane has in the address register: today one
    shuffle of that register for each row, at the load. A **companion** of a register X (a
    root) for a row is a register of the translation that holds X as the lane of that row has
    it. It is made by a shuffle at a whole place and stands until X is written. A load whose
    address register is X reads its companions; a load whose address register is made from X
    by one instruction of arithmetic on X and constants (the only definition that reaches the
    load, with no write of X on any way from it to the load) does that instruction again on
    the companions: a shuffle commutes with what is done lane by lane.

    Taken only where it moves shuffles out of a loop: the load is on a cycle that has no
    write of the root."""
    src, n = f.src, flow.n
    loads = sorted(plan.loads)
    address = {i: src[i].ops['Ra'] for i in loads}
    accumulators = {r for s in plan.sites.values() for base in (s['x'], s['y'], s['cx'], s['cy']) if base is not None for r in range(base, base + 4)}
    writers = {}
    def written(x):
        if x not in writers:
            writers[x] = frozenset(i for i in range(n) if flow.writes(i, x))
        return writers[x]
    def turns(load, x):
        """The load is on a cycle with no write of x: it comes round again with x as it was."""
        stop, seen, work = written(x), set(), list(flow.succ[load])
        if load in stop:
            return False
        while work:
            k = work.pop()
            if k == load:
                return True
            if k in seen or k in stop:
                continue
            seen.add(k)
            work += flow.succ[k]
        return False
    def between(made, load, x):
        """Some way from `made` to the load, not through `made` again, has a write of x."""
        ahead, work = set(), list(flow.succ[made])
        while work:
            k = work.pop()
            if k in ahead or k == made:
                continue
            ahead.add(k)
            work += flow.succ[k]
        behind, work = set(), list(flow.pred[load])
        while work:
            k = work.pop()
            if k in behind or k == made:
                continue
            behind.add(k)
            work += flow.pred[k]
        return any(k in written(x) for k in ahead & behind)
    usable = lambda x: x != 255 and x not in accumulators
    reach = flow.reaching(set(address.values()))
    candidates = {}
    for i in loads:
        ra, found = address[i], []
        if usable(ra) and turns(i, ra):
            found.append((ra, None))
        defs = reach[i][ra] if reach[i] is not None else frozenset([None])
        if len(defs) == 1 and None not in defs:
            (made,) = defs
            d = src[made]
            read = {(l, x) for l, x, *_ in d.reads}
            if (d.mnemonic in MIRRORED and unguarded(d) and flow.kind[made] == 'plain' and d.writes == [('R', ra)] and len(read) == 1
                    and f.mc.decode(70, d.word) is not None):
                (letter, x), = read
                if letter == 'R' and x != ra and usable(x) and turns(i, x) and not between(made, i, x):
                    found.append((x, made))
        if found:
            candidates[i] = found
    # The roots: the one that serves the most loads first (the lowest register among equals), until no load is left.
    left = dict(candidates)
    while left:
        serves = {}
        for i, found in left.items():
            for x, made in found:
                serves.setdefault(x, []).append(i)
        root = min(serves, key=lambda x: (-len(serves[x]), x))
        for i in serves[root]:
            plan.through[i] = next(c for c in left.pop(i) if c[0] == root)
        plan.roots[root] = [r for r in ROLES if any(plan.loads[i]['role'] == r for i in serves[root])]
    # Where the companions of each root are made. They are wanted at a place when a load that
    # reads them is reached from it with no write of the root on the way; they can be made at a
    # whole place. Followed from the entry of the function and from every write of the root,
    # along the ways: the first place that is both gets the shuffles, before it (after the
    # instruction before it, when that one alone brings a root without companions, is a whole
    # place and other ways come in: the ways on which the companions stand do not make them
    # again). Past it they stand, through places that are not whole too, until the next write.
    def places(root, users):
        stop = written(root)
        want = [False] * n
        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                new = i in users or (i not in stop and any(want[t] for t in flow.succ[i]))
                if new != want[i]:
                    want[i], changed = new, True
        first = lambda i: want[i] and flow.whole[i]
        # bare[i]: some way into i brings the root without companions.
        bare = [i == 0 or not flow.pred[i] for i in range(n)]
        leaves_bare = lambda q: q in stop or (bare[q] and not first(q))
        work = [i for i in range(n) if bare[i]] + sorted(stop)
        while work:
            q = work.pop()
            if not leaves_bare(q):
                continue
            for t in flow.succ[q]:
                if not bare[t]:
                    bare[t] = True
                    work.append(t)
        before, after = [], []
        for i in range(n):
            if not (first(i) and bare[i]):
                continue
            bad = [q for q in flow.pred[i] if leaves_bare(q)]
            if i != 0 and len(flow.pred[i]) >= 2 and bad == [i - 1] and flow.kind[i - 1] == 'plain' and flow.goes_whole(i - 1):
                after.append(i - 1)
            else:
                before.append(i)
        return before, after
    def round_without(load, place):
        """The load is on a cycle that does not go through `place`."""
        seen, work = set(), list(flow.succ[load])
        while work:
            k = work.pop()
            if k == load:
                return True
            if k in seen or k == place:
                continue
            seen.add(k)
            work += flow.succ[k]
        return False
    for root in sorted(plan.roots):
        users = {i for i, (x, made) in plan.through.items() if x == root}
        while users:
            before, after = places(root, users)
            # (A load that comes round only through a place where the companions are made gains
            # nothing: they are made as often as it loads. It keeps its shuffles.)
            lost = {i for i in users if any(i == q or not round_without(i, q) for q in before + after)}
            if not lost:
                break
            users -= lost
        for i in [i for i, (x, made) in plan.through.items() if x == root and i not in users]:
            del plan.through[i]
        if not users:
            del plan.roots[root]
            continue
        plan.roots[root] = [r for r in ROLES if any(plan.loads[i]['role'] == r for i in users)]
        for i in before:
            plan.made_before.setdefault(i, []).append(root)
        for i in after:
            plan.made_after.setdefault(i, []).append(root)
        if SAY:
            print('    companions of R%d (%s) for the loads %s, made before %s, after %s' % (root, ','.join(plan.roots[root]), sorted(users), before, after), file=sys.stderr)


def attempt(f, flow, forms, loads, reach, excluded):
    """One try: the plan with the products `excluded` left out, and in `.bad` the products to leave
    out next time when a conversion it would need cannot be where it must be (None: the plan stands)."""
    src, n = f.src, len(f.src)
    out = Plan()
    out.bad = None

    # ---- which products could be half of a pair, by where their operands come from --------------------
    def source(i, registers):
        """The destination and form of the loads whose results, all of them, reach `registers` at i; or None."""
        found = set()
        for k in registers:
            defs = reach[i].get(k) if reach[i] is not None else None
            if not defs or None in defs or any(j not in loads for j in defs):
                return None
            found |= {loads[j] for j in defs}
        if len(found) != 1:
            return None
        rd, transposed = next(iter(found))
        return (rd, transposed) if all(rd <= k <= rd + 3 for k in registers) else None
    candidate = {}
    for i, form in forms.items():
        if form is None or i in excluded:
            continue
        rd, ra, rb, rc = form
        # (What a product reads must be what loads gave, where it is: its A and its own B
        # register. The first product of a pair reads, of the program's registers, the two
        # B registers of the pair as well (section 4): `first` says whether this one can be
        # that. The second reads registers of the translation only, and the program may
        # have written the other B register by then: PyTorch's backward kernels do.)
        a, b = source(i, (ra, ra + 1)), source(i, (rb,))
        if a is None or b is None or a[1] or a[0] == b[0] or (ra - a[0]) % 2:
            continue
        candidate[i] = dict(d=rd, c=rc, ra=a[0], a=(ra - a[0]) // 2, rb=b[0], p=(rb - b[0]) // 2, e=(rb - b[0]) % 2, transposed=b[1],
                            first=source(i, (rb - rb % 2, rb - rb % 2 + 1)) == b)
    # ---- pairs: the next product in straight-line code with the same A and the other B of the pair -----
    def touches(k, registers, writes_only=False):
        return any(flow.writes(k, r) or (not writes_only and flow.reads(k, r)) for r in registers)
    four = lambda base: set(range(base, base + 4)) if base is not None else set()
    while True:
        pairs, used = [], set()
        for i in sorted(candidate):
            if i in used or not candidate[i]['first']:
                continue
            c = candidate[i]
            mine = four(c['d'])
            for j in range(i + 1, n):
                if j in flow.entered or flow.kind[j - 1] != 'plain':
                    break
                o = candidate.get(j)
                if (o is not None and j not in used and (o['ra'], o['a'], o['rb'], o['p'], o['transposed']) == (c['ra'], c['a'], c['rb'], c['p'], c['transposed'])
                        and o['e'] != c['e'] and not mine & four(o['d'])):
                    theirs = four(o['d'])
                    # The accumulator each product reads is read where the first one is, and both
                    # destinations are written there: an accumulator read is its product's own
                    # destination or apart from both destinations (the second product does not
                    # add to what the first one gives), and nothing between the two writes the
                    # one the second product reads. (The one the first reads may be written
                    # after it: compilers load the next operands there.)
                    apart = all(x['c'] == x['d'] or not four(x['c']) & (mine | theirs) for x in (c, o))
                    late = four(o['c']) - theirs
                    if apart and not any(touches(k, theirs) or touches(k, late, True) for k in range(i + 1, j)):
                        pairs.append((i, j))
                        used |= {i, j}
                    break
                # Between the two: nothing touches the accumulators, and no load brings other rows
                # for the same destinations (the second product reads the rows the first one read).
                if touches(j, mine) or (j in loads and loads[j][0] in (c['ra'], c['rb'])):
                    break
        sites = {}
        for i, j in pairs:
            ci, cj = candidate[i], candidate[j]
            first, second = (ci, cj) if ci['e'] == 0 else (cj, ci)
            for k, half, partner in ((i, 0, j), (j, 1, i)):
                sites[k] = dict(half=half, partner=partner, x=first['d'], y=second['d'], cx=first['c'], cy=second['c'],
                                ra=ci['ra'], a=ci['a'], rb=ci['rb'], p=ci['p'], transposed=ci['transposed'])
        # ---- loads: whole rows when everything their results reach is a product of a pair, in one role ---
        role = {}
        for u, s in sites.items():
            if s['half']:
                continue
            for registers, as_ in (((s['ra'] + 2 * s['a'], s['ra'] + 2 * s['a'] + 1), 'A'), ((s['rb'] + 2 * s['p'], s['rb'] + 2 * s['p'] + 1), 'BT' if s['transposed'] else 'B')):
                for k in registers:
                    for j in reach[u][k]:
                        role[j] = as_ if role.get(j, as_) == as_ else None
        whole = {j: as_ for j, as_ in role.items() if as_ is not None}
        # A pair stays when every load that reaches what its first product reads is done as whole rows.
        gone = set()
        for i, s in sites.items():
            if s['half']:
                continue
            for registers, want in (((s['ra'] + 2 * s['a'], s['ra'] + 2 * s['a'] + 1), ('A',)), ((s['rb'] + 2 * s['p'], s['rb'] + 2 * s['p'] + 1), ('B', 'BT'))):
                if any(whole.get(j) not in want for k in registers for j in reach[i][k]):
                    gone |= {i, s['partner']}
        unpaired = set(candidate) - set(sites)
        if not gone and not unpaired:
            break
        for i in gone | unpaired:
            candidate.pop(i, None)
    out.sites = sites
    # (`also`: what the load gives in Turing's layout may be read by something else on some way,
    # as far as can be told: those registers are then made too, as before.)
    out.loads = {j: dict(role=as_, rd=loads[j][0], also=flow.escapes(j, loads[j][0], sites)) for j, as_ in whole.items()}
    from machine import WRITE_BARRIER
    for i, s in sites.items():
        regs = (s['ra'] + 2 * s['a'], s['ra'] + 2 * s['a'] + 1, s['rb'] + 2 * s['p'], s['rb'] + 2 * s['p'] + 1)
        for k in regs:
            if any(j in out.loads and ((src[j].word >> WRITE_BARRIER) & 7) == 7 for j in reach[i][k] if j is not None):
                out.site_seq_wait.add(i)
                break
    if not sites:
        return out

    # ---- the accumulator registers: what each is next used as, and its state on every way --------------
    # (What a product of a pair reads and writes of accumulators in the translation: the first of a
    # pair reads the accumulators both products add to and writes both destinations; the second
    # reads and writes the two destinations.)
    of_site = {}
    for i, s in sites.items():
        wrote = four(s['x']) | four(s['y'])
        of_site[i] = ((four(s['cx']) | four(s['cy'])) if s['half'] == 0 else wrote, wrote)
    registers = sorted({r for read, wrote in of_site.values() for r in read | wrote})
    def use(i, r):
        """What instruction i does with register r: 'q' (an accumulator of a pair there, read and
        written), 'c' (read as one, not written), 'p' (written whole as one, not read), 't' (read,
        or written in part), 'w' (written whole), None."""
        if i in of_site:
            read, wrote = of_site[i]
            return 'q' if r in read and r in wrote else 'c' if r in read else 'p' if r in wrote else None
        if flow.reads(i, r) or (flow.kind[i] == 'other' and src[i].mnemonic not in ('BSSY', 'BSYNC')) or flow.removes(i):
            return 't'
        if flow.writes(i, r):
            return 'w' if flow.all_of(i, r) else 't'
        return None
    bad = set()
    def refuse(r, at=None, what=''):
        bad.update(i for i, (read, wrote) in of_site.items() if r in read | wrote)
        out.dropped.append((at, 'R%d %s' % (r, what)))
    # (`forced`: the places entered several ways where a register is taken in T though it is next
    # used by a pair, because a way into it cannot be converted; found as the analysis goes.)
    forced, pending = set(), list(registers)
    while pending:
        r = pending.pop(0)
        again = set()
        out.before = {i: [c for c in cs if c[0] != r] for i, cs in out.before.items()}
        out.after = {i: [c for c in cs if c[0] != r] for i, cs in out.after.items()}
        uses = [use(i, r) for i in range(n)]
        # Backward: what the register is next used as, before each instruction.
        want = [DEAD] * n
        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                u = uses[i]
                new = WANTS_Q if u in ('q', 'c') else WANTS_T if u == 't' else DEAD if u in ('w', 'p') else max([want[s] for s in flow.succ[i]] or [DEAD])
                if new != want[i]:
                    want[i], changed = new, True
        mode = [Q if want[i] == WANTS_Q and (r, i) not in forced and i else T for i in range(n)]
        # Forward: the state before each instruction. A place entered several ways is in the state
        # it is next used in (the ways into it are converted), or zero when every way brings zero.
        def after(i, state):
            u = uses[i]
            if u in ('q', 'p'):
                return Q
            if u == 'c':
                return Z if state == Z else Q
            if u == 'w':
                return Z if zero_writer(src[i]) else T
            if u == 't':
                return Z if state == Z and not flow.writes(i, r) else T
            return state
        zero = [True] * n
        zero[0] = False
        state = [None] * n
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if flow.simple(i):
                    s = after(i - 1, state[i - 1])
                else:
                    z = zero[i] and all(state[p] is None or after(p, state[p]) == Z for p in flow.pred[i])
                    if not z and zero[i]:
                        zero[i] = False
                    s = Z if (z and flow.pred[i]) else mode[i]
                # Before the instruction itself: what it requires.
                u = uses[i]
                if u in ('q', 'c') and s == T:
                    s = Q
                elif u == 't' and s == Q:
                    s = T
                if s != state[i]:
                    state[i], changed = s, True
        # The conversions that this needs, each at a whole place.
        for i in range(n):
            u = uses[i]
            if flow.simple(i):
                came = after(i - 1, state[i - 1])
                if (u in ('q', 'c') and came == T) or (u == 't' and came == Q):
                    if not flow.whole[i]:
                        refuse(r, i, 'before an instruction that is not at a whole place')
                    out.before.setdefault(i, []).append((r, Q if u in ('q', 'c') else T))
                continue
            arrives = state[i] if u not in ('q', 'c', 't') else (Z if zero[i] and flow.pred[i] and all(after(p, state[p]) == Z for p in flow.pred[i]) else mode[i])
            if want[i] == DEAD or arrives == Z:
                continue
            if u in ('q', 'c') and arrives == T:
                # (A place entered several ways where the register is taken as Turing has it,
                # and whose own instruction is a product of a pair: converted there, once the
                # ways have met. Left out until 2026-11-18: an independent checker found it on
                # cuBLAS's kernels, whose loops are entered three ways at their first product.)
                if not flow.whole[i]:
                    refuse(r, i, 'before an instruction that is not at a whole place')
                out.before.setdefault(i, []).append((r, Q))
            for p in flow.pred[i]:
                came = after(p, state[p])
                if came in (Z, arrives):
                    continue
                kind = flow.kind[p]
                both = kind == 'cond' and all(want[s] != DEAD and mode[s] == arrives and not (zero[s] and state[s] == Z) for s in flow.succ[p])
                if kind == 'jump' or both:
                    place, ok = out.before, flow.whole[p]
                elif kind == 'plain' and i == p + 1:
                    place, ok = out.after, flow.goes_whole(p) or flow.lead[i]
                elif kind == 'cond' and i == p + 1 and flow.target(p) != i:
                    place, ok = out.after, flow.goes_whole(p)
                else:
                    place, ok = None, False
                if not ok and arrives == Q:
                    if SAY and (r, i) not in forced:
                        print('    R%d is taken as Turing has it at %d: the way from %d (%s, %s there) has no whole place for its conversion' % (r, i, p, kind, came), file=sys.stderr)
                    again.add((r, i))
                elif not ok:
                    refuse(r, p, 'on the way from a %s to %d' % (kind, i))
                elif (r, arrives) not in place.setdefault(p, []):
                    place[p].append((r, arrives))
        if again - forced:
            forced |= again
            pending.insert(0, r)
    out.before = {i: cs for i, cs in out.before.items() if cs}
    out.after = {i: cs for i, cs in out.after.items() if cs}
    if bad:
        out.bad = bad
        out.before, out.after = {}, {}
    return out


if __name__ == '__main__':
    import elf, machine as MC, translate as TR
    mc = MC.Machine()
    for path in sys.argv[1:]:
        data = open(path, 'rb').read()
        e = elf.Elf(data)
        for s in e.sections:
            if not (s.name.startswith('.text.') and s.type == 1) or len(s.data) % 16:
                continue
            name = s.name[6:]
            info = e.section('.nv.info.' + name)
            try:
                f = TR.Function(mc, name, s.data, s.info >> 24, TR.tables_of(info.data) if info is not None else {})
            except TR.Refused:
                continue
            products = sum(d.c.name == PRODUCT for d in f.src)
            if not products:
                continue
            p = plan(f)
            if p is None:
                print('%s %s: %d products, no plan' % (os.path.basename(path), name[:50], products))
                continue
            roles = {}
            for v in p.loads.values():
                roles[v['role']] = roles.get(v['role'], 0) + 1
            print('%s %s: %d products, %d in pairs, loads %s, extra rows for %s, kept %s' % (
                os.path.basename(path), name[:50], products, len(p.sites), roles, p.rows(), sorted(p.kept())))
            for i, c in sorted(p.before.items()):
                print('    before %5d %-8s %s' % (i, f.src[i].mnemonic, ' '.join('R%d>%s' % x for x in c)))
            for i, c in sorted(p.after.items()):
                print('    after  %5d %-8s %s' % (i, f.src[i].mnemonic, ' '.join('R%d>%s' % x for x in c)))
