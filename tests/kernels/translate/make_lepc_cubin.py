#!/usr/bin/env python3
"""make_lepc_cubin.py IN.cubin OUT.cubin: turn a direct call of k_lepc.cu into the pair cuDNN
9.24's sm_75 kernels use, a LEPC immediately before the call whose program counter the callee
returns through.

nvcc gives a direct call a MOV of its return address and the callee a RET.REL. Here the MOV
before the call becomes a LEPC of the same register, and every RET of the callee becomes a
RET.ABS of that register adding 0x20 (the address two instructions after the LEPC, which is the
instruction after the call). The MOV's register must not be written by the callee, or the
returned address would be lost: the script refuses then, and says so.

Checked in beside k_lepc.cu: the source of the cubin that `reference/compare.sh` and
`reference/controls.sh` use for the LEPC rules without NVIDIA's code.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'reference'))
import elf
import machine
import translate


def operand_field(mc, cls, operand):
    for kind, fields, data in mc.d75.plan(cls):
        if kind == 'operand' and data == operand:
            return fields[0]
    raise SystemExit('no operand %s in %s' % (operand, cls.name))


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    data = open(sys.argv[1], 'rb').read()
    e = elf.Elf(data)
    mc = machine.Machine()
    patched = 0
    for s in e.sections:
        if not s.name.startswith('.text.'):
            continue
        words = [int.from_bytes(s.data[at:at + 16], 'little') for at in range(0, len(s.data), 16)]
        decoded = [mc.decode(75, w) for w in words]
        call = next((k for k, d in enumerate(decoded)
                     if d is not None and d.c.name.startswith('call_rel')), None)
        if call is None:
            continue
        # The MOV of the return address immediately before the call: MOV Rr, the address after it.
        mov = decoded[call - 1] if call > 0 else None
        if mov is None or mov.mnemonic != 'MOV' or 'Sb' not in mov.ops:
            raise SystemExit('%s: no MOV of a return address before the call' % s.name)
        register, back = mov.ops['Rd'], mov.ops['Sb']
        if back != 16 * (call + 1):
            raise SystemExit('%s: the MOV holds 0x%x, not the address after the call (0x%x)'
                             % (s.name, back, 16 * (call + 1)))
        # The callee: the target of the call, up to the end of what is here.
        rel = translate.relative(mc, 75, decoded[call])
        if rel is None:
            raise SystemExit('%s: the call has no relative target' % s.name)
        off = translate.signed(rel[0], decoded[call].word) * rel[1]
        target = (16 * call + 16 + off) // 16
        if not 0 <= target < len(decoded):
            raise SystemExit('%s: the call goes outside what is here' % s.name)
        body = range(target, len(decoded))
        rets = [k for k in body if decoded[k] is not None and decoded[k].mnemonic == 'RET']
        if not rets:
            raise SystemExit('%s: the callee returns nowhere' % s.name)
        for k in body:
            d = decoded[k]
            if d is None:
                raise SystemExit('%s+0x%x does not decode' % (s.name, 16 * k))
            if any(l == 'R' and r == register for l, r in d.writes) and k not in rets:
                raise SystemExit('%s+0x%x writes R%d, which must hold the return address'
                                 % (s.name, 16 * k, register))
        # The LEPC of the same register in the MOV's place.
        lepc = mc.assemble('lepc_', Rd=register)
        if mc.decode(75, lepc) is None or mc.decode(75, lepc).mnemonic != 'LEPC':
            raise SystemExit('cannot make a LEPC of R%d' % register)
        words[call - 1] = lepc
        # A RET.ABS.NODEC of the register, adding 0x20, in each RET's place.
        abs_ret = 0x140fea0003e000000000002080007950        # RET.ABS.NODEC R128 0x20
        cls = mc.decode(75, abs_ret).c
        abs_ret = operand_field(mc, cls, 'Ra').put(abs_ret, register)
        for k in rets:
            words[k] = abs_ret
        patched += 1
        s.data = b''.join(w.to_bytes(16, 'little') for w in words)
    if patched != 1:
        raise SystemExit('expected one call with a return address; found %d' % patched)
    open(sys.argv[2], 'wb').write(e.write())


if __name__ == '__main__':
    main()
