# The Python reference for translating machine code to sm_70

Offline tools, written to find out what a trustworthy translator needs and to check the one the
layer has. The translator of the layer (`crates/vf-sass`) is `translate.py` carried over to
Rust, on tables that `generate.py` writes from the same descriptions; the two must give the same
bytes, map and refusals (`tests/sass/compare.sh`, which compares `MAP.lives` too, and
`tests/sass/fuzz.py`), so a change to what one translates is a change to both.
`docs/translator.md` describes the translation, `docs/validation.md` the checks.

They read the instruction descriptions that the
[denvdis](https://github.com/redplait/denvdis) project extracted from NVIDIA's disassembler
(`src/denvdis/data/sm70_*.txt`, `sm75_*.txt` and `src/denvdis/data12/sm80_1.txt`, fetched by `scripts/fetch-all.sh denvdis`, not in
this repository), and some use `toolkits/cuda-12.9.2/bin/nvdisasm` (fetched by
`scripts/fetch-all.sh redist`) as the judge of what an instruction word means on each
architecture. Run them with `python3 -I`.

## The translator

| File | What |
|---|---|
| `md.py` | The descriptions read into classes (format, opcode, fields); `Decoder`: the class of an instruction word and its operands |
| `elf.py` | The container read and written back (byte for byte when nothing changed) |
| `machine.py`, `translate.py IN OUT [MAP]` | The reference translator; with MAP, it writes `MAP.lives` beside it, for `validate.py` (as `vf translate` does). `VF_BORROW`, `VF_PAIRS`, `VF_COMPANIONS` (`0` by default, `1` opt-in), `VF_ALIGNED_ROWS` as in the layer; `VF_BORROW_SAY=1` says what it borrowed, `VF_PAIRS_SAY=1` where it gives pairs up (`docs/pairs.md`, section 12) |
| `pairing.py CUBIN...` | The plan of pairs of products on whole rows for each function (`docs/pairs.md`, section 11): which loads, which products, which conversions where; printed |
| `batch.py OUT_DIR CUBIN...` | The translator on many cubins in one process (what `tests/sass/compare.sh` compares with) |
| `ampere.py` | Checks and lowers supported sm80 instructions before the ordinary Turing-to-Volta translation. |
| `ampere_forms.py` | Projects the selected sm80 forms from denvdis `data12/` and checks the shared forms against the sm75 description used by the translator. |
| `generate.py` | Writes the tables of `crates/vf-tables/src/` (then `cargo fmt -p vf-tables`) |

## The checks

| File | What |
|---|---|
| `validate.py SM75 TRANSLATED MAP` | Every instruction of a translation against nvdisasm's reading of it, whichever translator made it (both write the `MAP.lives` it reads beside MAP); the schedule against the distances that native code keeps; the registers and predicates that the translation takes against the lives of the program's. It reads the original as sm_75: the sm_80 path (`ampere.py`) is not checked by it (`docs/validation.md`, section 8) |
| `qcheck.py`, `qflow.py` | What a plan of pairs has that the design does not allow, for the original function; the flow, the reaching definitions and the whole places it works from. Written apart from `pairing.py` |
| `controls.py [-u] [-c CONTROL]... SM75...` | Translations that are wrong on purpose, one fault each, which `validate.py` must object to. `tests/sass/controls.sh` runs every control in a process of its own, with the settings of the translator that each needs; `docs/validation.md`, section 3, has what comes of it |
| `check.py SM CUBIN...` | The decoder against nvdisasm on every distinct instruction of real code: one class, same mnemonic, same registers |
| `uses_check.py SM CUBIN...` | What each instruction reads and writes, by the tables, against the lives of registers that nvdisasm draws |
| `flow_check.py SM75...` | Where each instruction may lead: the translator's flow against the validator's. They must not differ |
| `pairs_model.py` | The arrangement of pairs of products (`docs/pairs.md`, section 2) checked on the host in exact integers |

## Where the tables come from

| File | What |
|---|---|
| `both.py OUT CUBIN...` | Which sm_75 instructions are, bit for bit, sm_70 instructions that read the same |
| `classes.py` | The sm_75 classes compared with the sm_70 classes of the same opcode, field by field |
| `remap.py TABLE` | Rules from classes with uniform registers or predicates to ordinary sm_70 classes, each confirmed by nvdisasm on the instructions of `TABLE` (of `both.py`); writes `rules-sm75.tsv` |
| `latency.py [-f] SM CUBIN...`, `merge.py`, `distances.py`, `compare.py` | The stall distance that real code keeps between a writer and a reader, by kind of instruction; `distances-sm70.tsv` is what native sm_70 code keeps |
| `probe.py` | Candidates per opcode (used by `check.py` and `latency.py`) |

## What native code does

Read from machine code that NVIDIA's compilers produced (PyTorch's, cuBLAS's and others): the
observations that the timing rules of `docs/translator.md` rest on.

| File | What |
|---|---|
| `idioms.py SM CUBIN...` | How code passes return addresses (as numbers: they must follow when code moves) |
| `sched.py CUBIN FUNCTION` | A function's instructions with their scheduling fields |
| `late.py SM CUBIN...` | What native code does between an instruction that reads its operands late and the next write of one of them: a wait for its read, for its result, or neither (and then how far) |
| `scoreboards.py SM CUBIN...` | Which instructions name a scoreboard for having read and for their result, and the stall of one whose scoreboard the very next instruction waits for (never less than two) |
| `pending.py SM CUBIN...` | How many instructions native code leaves pending on one scoreboard before a wait on it |
| `pairs.py SM CUBIN...` | Whether the operands of two registers or more start at an even register (always) |

To try the translator on a V100: `diffrun REFERENCE.cubin TRANSLATED.cubin` runs each kernel of
both files on the same inputs (`tests/gpu/run.sh` builds it from
`tests/kernels/translate/diffrun.c` into `bin/` of its output directory, `$VF_OUT` or else
`$VF_TEST_ROOT`).
