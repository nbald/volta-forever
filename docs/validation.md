# Validation: how a wrong translation would be found

The layer rewrites program text and translates machine code so that software built for later
GPUs runs on a Tesla V100. A wrong translation gives wrong numbers silently: the GPU runs what it
is given. This document says what evidence there is that the translations are right, how each
piece of it was obtained, and what none of it covers.

Unless another date is given, the counts are of 2026-11-20, with the layer and the reference
translator at their defaults; what ran on a GPU ran on one Tesla V100-SXM2-32GB with driver
580.178.04.

## 1. Layers of evidence

| Layer | What it can find | What it cannot find |
|---|---|---|
| **The independent validator** (`reference/validate.py`, section 2) | An instruction of a translation that is not what the original asks for; a register or a predicate that the translation takes where it holds something of the program; a wait that is missing or comes too close; a relocation left behind or carried wrong | A distance that is too short where no scoreboard is named (the table of distances is taken on trust); whatever only the hardware shows; anything in the translation of sm_80 machine code, which it does not read (section 8) |
| **Controls** (`reference/controls.py`, section 3) | A rule of the validator that lets a wrong translation through: each control is a translation made wrong on purpose in one way, which the validator must object to | The faults that nobody thought of; a fault in what the validator and the translator share (the descriptions of the instruction sets) |
| **The decoder against nvdisasm** (`reference/check.py`, section 5) | An instruction word given another class, mnemonic, register or predicate than NVIDIA's disassembler gives it | Fields that nvdisasm's text does not show as registers: constants, modifiers |
| **What instructions read and write, against nvdisasm** (`reference/uses_check.py`, section 5) | A register or a predicate that an instruction reads or writes by the tables and not by the chart of lives that nvdisasm draws, or the reverse | What the chart does not draw: calls, and code that nvdisasm reads in its older format |
| **The translator's flow against the validator's** (`reference/flow_check.py`, section 5) | A place where an instruction may lead for one of them and not for the other | A place that both miss |
| **The checker of pair plans** (`reference/qcheck.py`, `reference/qflow.py`, section 2) | A plan of pairs of products on whole rows that the design does not allow | The arithmetic of the arrangement, the schedule, what the hardware does |
| **Two translators, the same bytes** (`tests/sass/compare.sh`, section 7) | A difference between the reference translator in Python and the translator of the layer in Rust | A fault common to both: they share the tables and the rules |
| **The fuzzer** (`tests/sass/fuzz.py`, section 7) | A difference between the two translators on damaged cubins; a translator that ends improperly | A fault common to both |
| **Translated kernels against native builds, on the hardware** (`tests/gpu/run.sh`, `tests/kernels/translate/`, section 6) | A translated kernel whose results on a V100 differ from those of the same source compiled for sm_70 by NVIDIA's compiler | Only the kernels of the suite, on their inputs |
| **Rewritten instructions against host models, on the hardware** (`tests/kernels/tcrun.c`, section 6) | An instruction that Volta lacks whose replacement gives another result than its definition: on values where the answer is decided, and on random ones within a tolerance | Values and combinations that the kernels do not have |
| **Reference vectors from an RTX 5090** (`tests/kernels/blackwell/`, section 6) | A conversion of the 4-bit and scale formats that does not give the byte that the hardware gives; a block-scaled product further from the hardware's result than it is today | Forms and inputs that the vectors do not have |
| **Programs** (PyTorch's operations and whole models, llama.cpp, vLLM, SGLang, ZML: section 6) | A program that gives other results than the CPU, stops, or faults on the GPU | The kernels that these programs do not launch with these models and inputs |

## 2. The independent validator

`reference/validate.py` (5,100 lines of Python) reads an sm_75 cubin, its translation, and the
map of the translation: for each instruction of the result, which instruction of the original it
comes from and what was done to it. It was written to check a translation whichever translator
made it. The map is the translator's own, so it is checked first, and so is everything else that
the translator says about its work (which predicate stands for which, which registers it took:
the file `MAP.lives` beside the map, which both translators write).

It reads sm_75 code only. The translation of sm_80 machine code, which the layer has made since
0.1.2, is not checked by it: section 8 says what checks it instead.

What an instruction is, the validator takes from NVIDIA's disassembler: the original is read as
sm_75 and the result as sm_70 by nvdisasm, and the rules are on its text. The scheduling fields,
which nvdisasm does not print, are read from the words of the result, with the descriptions of
the instruction sets (the same ones the translator has) only for what kind an instruction is and
what it reads. The validator works out the control flow of each function and the lives of its
registers and predicates by itself.

### The map

- The map has every instruction of the result once and every instruction of the original; every
  function of the result is in the map, or is what a refused function becomes (a trap, with the
  exits).
- In the order of the result, the instructions come from the original in the order of the
  original.
- A kind of rewriting that the validator does not know is a problem.
- Whatever its kind, an instruction of the result that reads or writes memory has the guard of
  the instruction it comes from. (But what loads a uniform register from the frame before an
  instruction and stores it after: under no guard, "The frame" below.)

### Each instruction

**Same.** The instruction reads as the original, with a branch target moved to where its
instruction now is.

**Ordinary registers.** The instruction reads as the original with each uniform register replaced
by an ordinary one, always the same one, and the mnemonic without its U. An operand of two uniform
registers (as the destination and the addend of `UIMAD.WIDE` and `UIMAD.WIDE.X`) is printed as its
first register only: the ordinary register that stands for that one is even, and the uniform
register above it has the ordinary register above that one wherever it is named alone (since
2026-11-26, pull request #15). The translators place a pair so (`docs/translator.md`, "Placing
uniform registers and predicates"): this rule of the validator is what checks it. The control
`unpaired` gives the upper half another register, and `misruled` alters by one the immediate of an
instruction made so (section 3).

**Sequences.** What stands for an instruction that has no such twin (the upper half of a
product, a minimum or maximum, a stride, a constant at a uniform index) must be the instructions
that its recipe gives, on the operands of the original.

**Matrix instructions.** What stands for a matrix load or a tensor-core product of Turing must be
the sequence that its recipe gives, on registers of the translation, each for one thing. The
sequence goes by the lane of the thread, read into such a register when the function is entered,
and a product by four values made of the lane, computed there too: nothing else in the function
writes those registers, and no global function symbol points into the function elsewhere than at
its start. At level 3 (section 5 of `docs/translator.md`) a sequence takes registers of the
program that are free at its instruction, as the pool does at others: those are checked
instruction by instruction with every borrowed register, not for the whole function (until
2026-11-23 a function with such matrix loads and no product was objected to for that alone).

**Kept values.** A product may take the elements of its B operand from the registers where an
earlier product gathered them. Then that product must be the last that gathered, from the same
register of B into the registers this one reads; nothing from it on writes that register of B in
the original, and nothing after its gathering writes those registers in the result; no
instruction between them is entered otherwise than from the instruction before it, no branch of
the function leads to an unknown place, and no call is between them.

**Pairs of products on whole rows** (`docs/pairs.md`). The validator checks the sequences: a
register that holds a row is written by loads of rows only, and the loads with one destination
put the same rows in the same registers; the registers of a half of a pair are the rows of the
loads of its operands; a half has its other half, and adds to what its products add to; the
halves say on one scoreboard when their results are there; the conversions of the accumulators
wait for every scoreboard, fifteen cycles before their first shuffle, and what follows them
waits for them. Which products may be paired, which loads serve which, and the state of the
accumulators on every path are the plan: the validator gives it to `reference/qcheck.py`, a
checker written from the design apart from the planner (`reference/pairing.py`), with its own
analysis of the flow (`reference/qflow.py`), and reports what it objects to.

### Predicates

A uniform predicate may have several ordinary ones stand for it, one for each of its lives. The
translator says which, and what it says is checked against the lives of the program's
predicates, which the validator has from two sources: nvdisasm's own chart of lives (`-plr`),
completed across the calls within a function (nvdisasm takes every call to write every
predicate, which compiled code does not hold to), and a backward dataflow of its own on its own
control-flow graph, with `P2R` and `R2P` by their masks. nvdisasm refuses a function that
declares 255 registers; the validator gives it a copy that declares 254, the chart of predicates
being the same.

For each life: the ordinary predicate holds nothing of the program all along it (a free life),
or nothing of the program reads or writes it there and it is saved before the life and given
back after it (a saved life); no value goes from one life to another; an instruction uses the
predicate of the life it is in. A life may also be held in a register of the translation: then
the value is read back into a predicate before every instruction that reads it or is under its
guard, and written down after every instruction that writes it.

A predicate that some instruction of the function writes late (a comparison of doubles, an
instruction of a late kind: `docs/translator.md`, "Predicates written late") may stand in only
behind a wait: what writes or saves it begins with instructions that do nothing, the first of
which waits for the six scoreboards, and that last as long as the translator's rule asks. One
that is written late under no scoreboard may not stand in at all.

### Registers of the program

Where a function has no register to spare above its own, the translation takes registers of the
program where they hold nothing (`docs/translator.md`, "Registers").

**The whole truth.** Every register of the program that an added instruction writes is declared:
as a destination of the original instruction, as the home of a uniform register, as a register
that holds a predicate, as one taken for the time of an instruction, or as one put away for it.
A register written and not declared is a problem. (The register into which a matrix sequence
that takes everything at the instruction reads the lane too: until 2026-11-23 it was left out
with what is put around an instruction, and `missing_pool_line` passed unnoticed in
`k_stored.135.sm75.cubin`, where the lane goes into a register of the program. It is declared
taken there, or put away there, as the rules of room below ask: the store before the lane is
read into it, the load after. The first form of this check took the first declaration only,
and objected 1,443 times in 12 cubins of the held-out corpus to the lane of products put in
registers put away; the controls `room_lane_unsaid` and `room_lane_first` are for the second.)

**Free.** A declared register must hold nothing of the program where it is taken, by the
validator's own lives of registers: dead on the way in, dead on the way out, and neither read
nor written by the original instruction. No two uniform registers are declared in one home at
one instruction (two may have one register for home at different instructions), and a register
taken for an instruction is not a home declared alive there. A uniform register for which no
home is declared has its register in the whole function, and shares it with nothing.
Registers are borrowed in kernels only: a function that others call may be entered with its
caller's values in registers that it never names.

**The life of a home.** A register is the home of a uniform register for a life that the
translator declares, and serves other purposes outside it; this holds for a register of the
program and for one left under the kernel's own count alike. The validator works out the lives
of the uniform registers itself, with those of the program's registers: the declared life must
be the whole life (every instruction before or after which the uniform register holds something
that may still be read, and every instruction that reads or writes it), and the homes of one
uniform register are one register. And whatever is declared, nothing that the translation adds
writes a home at an instruction of its declared life, but what the instruction that writes
that uniform register became.

**A predicate held in a register.** The register that holds a uniform predicate between the
instructions that name it may be one of the program's: it then holds nothing of the program at
any instruction of the held life. It holds one predicate at an instruction (it may hold another
along other lives), and nothing that the translation adds writes it along a held life but what
writes the predicate down. Nothing but the save itself writes a register in which predicates
are saved.

**Waits.** The program may still be writing a register that the translation takes: the first
write of a borrowed register waits for the scoreboards on which the program writes it late. An
added instruction that writes a borrowed register late names a scoreboard, and what follows
waits for it; one that reads a borrowed register late says when it has read.

**The frame.** A function that has no register for its uniform values keeps them in its thread's
local memory. The attributes of the function are raised by the size of the frame; the stack
pointer is lowered by it, once; two values that make a pair are in adjacent words; each load
takes the word of its uniform register, from a base that is the constant the driver gives, and
the instruction right after the last load waits for it; every uniform register that the
original writes is stored, and the store says when it has read. A store is under no guard: an
instruction may write the predicate that guards it, and a store under that guard would go by
what the instruction left there. What an instruction under a guard writes must have been loaded
before it, so that the store puts back what was there when the instruction did not run. Where
the frame is addressed from the stack pointer, the validator follows the pointer by a dataflow
of its own (a branch through a register leads where nvdisasm's listing says), and every access
to the frame must be at a place where it knows how far the pointer is: where two ways lead to
one place with the pointer at two depths, or a branch may lead anywhere, it knows nothing.

A uniform predicate that has no register to be held in may be kept in a word of the frame too:
before an instruction that names it, it is loaded into a register of the translation and read
back into a predicate; after one that writes it, it is written down into such a register and
stored. Each of these moves must be beside its conversion, on its side of the instruction, take
the predicate's own word through the stack pointer, under no guard, with a register that was
borrowed or put away there and that the conversion names, once for each direction, and on a
scoreboard that what follows waits for.

**Room.** Where no register holds nothing, one that the original instruction neither reads nor
writes is put in the frame for the time of the sequence and brought back after it. The validator
checks the register (not R0 or R1, not read or written by the original, not a home alive there,
not one that holds a predicate), the instruction (not a branch, a call, a return, nor one that
writes the stack pointer), the store before any write and the load after every read and write,
each on a scoreboard that the next instruction waits for, each under no guard (where a guard of
the store did not hold, the register would not be put away, and would be brought back all the
same: until 2026-11-23 such a store passed), each register in a word of its own, the same word
both ways, and a frame large enough for the uniform values and the room. A register that the
translator declares put away at an instruction may be written there while it holds something:
the validator asks that its store and its load be there.

**What the frame's moves carry.** The rules of the two paragraphs above say what each load,
each store, each register put away and each register brought back is by itself: the word, the
register, the scoreboard. They would pass a store placed before the instruction that computes
what it stores, or a load into a register whose value the instruction still needs. So the
validator also follows the values through everything that an instruction became, in the order
of the code, with a name for each: what a register held before the sequence, what the frame
holds of a uniform register, what an instruction of the sequence made. A load gives its
register the value of its word, a store gives its word the value of its register, and so do
what puts a register away and what brings it back. Then:

- the register that stands for a uniform register of the frame at an instruction (the one its
  load writes, or the one its store reads where the instruction only writes it) holds nothing
  else that the instruction needs: it is not a register that the instruction itself reads or
  writes, not the home of a uniform register there, not a register that holds a predicate
  there, and not the register of another uniform register of the frame at that instruction;
- what the instruction became reads the register of a uniform register as loaded, or as the
  instruction made it; and in any other register, nothing that a move of the frame put there;
- a store is of a uniform register that the instruction writes, of the value that the
  instruction made in its register, and of the last one it makes there; once for each;
- a register put away ends with what it held;
- a load or a store of the frame, and what puts a register away or brings it back, takes its
  address from the stack pointer, or from a register that holds the top of the frame at that
  instruction: the register that it is loaded in, after the instruction that loads it;
- nothing that is put around an instruction, and no load of the frame, writes the home of a
  uniform register declared there, even where the instruction writes that uniform register:
  the instruction may still read it;
- the register that a move of a uniform predicate of the frame takes is not the register of a
  uniform register of the frame there, not one that the instruction reads or writes, not a
  home, not a register that holds a predicate, and not the register that the top of the frame
  is loaded in; what a load of a predicate puts in it is read by its conversion only, and a
  store of a predicate stores what its conversion wrote down (since 2026-11-22: section 7).

**Products that take everything at the instruction.** In a kernel with no register to spare, a
tensor-core product takes its sixteen registers for the time of the instruction. The validator
checks the sequence against its recipe, that every pair starts at an even register, that the
registers are distinct and hold nothing of the program, that the lane is read on a scoreboard
with the next instruction waiting, and the distances between the steps and after the last one.

### The schedule

- What an instruction of the original waited for, the first instruction of what it became waits
  for.
- A wait comes two cycles or more after the instruction that names its scoreboard (native code
  has no closer pair, and on a V100 a wait one cycle after does not wait).
- An instruction that reads its operands some time after it is issued (a memory access, a
  conversion, arithmetic on doubles) and reads a register that the translation gave it says on a
  scoreboard when it has read, or when its result is there, and the very next instruction waits
  for that. When the register is one of the program's that the translation borrows, read by an
  instruction that is not the program's own unchanged, what next writes it in the code that
  follows comes after a wait for that read (the homes of uniform registers are read late and
  written again only much later). (Until 2026-11-23 the borrowed ones were left out: at level 3 a
  store at the kept copy of its scaled index, in a borrowed register that the next copy writes,
  could go without the wait unseen; the control `nowait` changed two Triton cubins so and was
  objected to in neither, which an objection of a check of the whole function to the translation
  as it is had hidden.) Here and wherever a rule asks the next instruction to wait (after a load from the
  frame or a store to it, after a register put away or brought back, after a late write to a
  borrowed register, and after a late writer translated by a rule from a fixed-latency
  instruction under no scoreboard such as `UFLO` -> `FLO`), instructions that do nothing may stand between: the schedule puts some
  where it needs more time than a stall holds, a wait holds for everything after it, and an
  instruction that does nothing reads and writes no register.
- A predicate given back from a register of the translation comes nine cycles or more after the
  instructions before it, in straight-line code, that read that predicate.
- After the last step of a product, eight cycles pass before anything but another product reads
  its result (thirteen for accumulators in halves), and the steps of a product are as far apart
  as its recipe has them.
- A register that a memory instruction reads as data (what a store stores, an atomic operation
  combines, a shuffle exchanges; not the address of a memory access) is not written after it, in
  straight-line code, by anything but a memory instruction before a barrier or a wait for a
  scoreboard that it or a later memory instruction names (`data_overwritten`). Memory
  instructions go in order: native code writes such a register at once with one (`ptxas`, 4
  cycles after a store, with a load from shared memory or a matrix load), and with anything else
  only after such a barrier or wait: none otherwise among 44,052 such writes in 821 sm_75
  cubins, PyTorch's 475 and 346 of Triton (2026-11-23). Until that day the sequence of a matrix load wrote the lanes of its rows
  into its destinations first, the validator found nothing, and on a V100 a store of one of them
  just before the load stored the lane. A store under a guard (an atomic operation too) reads
  nothing in the lanes where its predicate does not hold: a write under the complement of that
  guard, on the same predicate, which nothing between them writes, runs in those lanes alone and
  is not objected to (`ptxas` writes so at once, `@P2 STS.64 [R6], R4` then `@!P2 IMAD.MOV.U32
  R5, RZ, RZ, RZ`: 1,270 writes in 14 cubins of the held-out corpus, of cuSOLVER, NCCL and
  vLLM); the register is still followed after it, for the lanes where the store ran. Not after a
  shuffle, which reads the register in other lanes. (Until 2026-11-23 such a write was objected
  to.) The controls `complement_rewritten` (a write of that predicate between them, in the
  program) and `complement_same` (the write under the store's own guard) are objected to.

### What the validator does not check

1. **The table of distances.** How long after an instruction of the translation that names no
   scoreboard its result may be read. The translator and the validator take the same table
   (`reference/distances-sm70.tsv`: what native sm_70 code keeps), and nothing else looks at it.
   It matters where the original named a scoreboard and what it became names none: an `R2UR`
   that became a `MOV`, a sequence for a matrix load whose last instruction names none.
2. **Late readers of borrowed registers inside a product that takes everything at the
   instruction.** The check of the schedule inside a matrix sequence follows the registers above
   the program's; in such a product the registers are among the program's.
3. **The lives of uniform registers have one source.** For predicates the validator has
   nvdisasm's chart and its own dataflow; for uniform registers its own dataflow only (it does
   not read the box of uniform registers of nvdisasm's chart, which nvdisasm leaves out of
   some functions that have them). `reference/uses_check.py` compares what each instruction
   reads and writes with that chart (section 5); nothing compares the lives.
4. **What a memory instruction reads as data, past a branch.** The rule above follows such a
   register in straight-line code only: not past a branch, nor from the end of a loop to its
   start.

## 3. Controls

A check that never objects proves nothing. `reference/controls.py` makes the reference translator
wrong on purpose, one way at a time, and counts what the validator finds in the result: 119
controls, each described at the head of that file. A control that changes translations and meets
no objection anywhere is a hole in the validator, or a control that tests nothing.

A control needs a translation that has what it takes away. `tests/sass/controls.sh` runs each
one in a process of its own, in five passes: at the default; with every function made to borrow
registers (`VF_BORROW=1`); with every kernel at level 3 (`VF_BORROW=8`); with room made
(`VF_BORROW=5`, the setting of the second translation); and with every frame addressed from
its top, which a register is loaded with (`VF_BORROW=3`), for the two controls on that
register. In each pass the translation as it is is validated first. On 39 cubins (the 16 test
cubins of `tests/kernels`, 17 of PyTorch's and 6 of the held-out corpus, chosen for having what
the controls need), 2026-11-23:

- **110 of the 113 controls are objected to**, each in every cubin whose translation it
  changes, but in 10 cases looked at one by one where what it changed stays right (below);
- 2 change nothing in any of these cubins, nor in any of PyTorch's 475: nothing at hand has
  what they take away (below);
- 1 is of another kind (`copy_after_write`, below);
- the translation as it is: 0 problems in the 39 cubins, at each of the five settings.

Eighty minutes with 24 processes.

Two of the four controls written since are in section 8 (`doublesoon`, `doubleacross`). The two
others came on 2026-11-26 with the rule on pairs of uniform registers (section 2, "Ordinary
registers"; pull request #15): `unpaired` gives the upper half of a 64-bit uniform operand
another register than the one above its lower half's, and `misruled` alters by one the immediate
of the first instruction of a function that a rule translates to ordinary registers. At the
default, one process a cubin, on 49 cubins (35 of the held-out corpus and of the 39 above, on
which `unpaired` stopped until it took the arguments of the translator by their names, and 14
others), `unpaired` changed the translation of 46 and was objected to in each, `misruled` changed
45 and was objected to in each.

A control may change a translation that stays right, and is then rightly not objected to there:
`alive_home` moves every home to the next register, which may be free too; `short_home` leaves
out of a declared life an instruction where, by the validator's own lives, the uniform register
holds nothing (the translator declares some lives longer than they are); `no_borrow_wait` takes
away a wait for the six scoreboards where no load of the program in flight writes the register;
`held_busy_l3`, in a kernel of cuDNN, takes registers that the translator holds busy only in a
routine that the kernel calls (at a return every register is alive for it), and that the
validator, which follows each call, finds free all along the life; and where a control makes the
translator's own attempt fail, the function is translated another
way or refused. `controls.py` prints `refused` for a cubin where all that a control changed is
functions that the translator then refused (they are traps: nothing of it reaches the validator),
and `=` where it changed neither the code, nor the map, nor the lives; neither counts as a
change. `tests/sass/controls.sh`, and `controls.py` run by itself, fail on every other cubin whose
translation a control changes and where the validator finds nothing, but for the 10 cases of the
list `RIGHT_THERE` of `controls.py`, each looked at. (Until 2026-11-23 one objection anywhere was
enough: the validator could have lost its objection to a control in every other cubin unnoticed.) The tables give, for each pass, in how
many cubins the control changed the translation and in how many of those the validator objected.
(`summary.tsv` of the run has the first objection to each, which is not always the one the
control was written for: a fault often breaks more than one rule.)

**The map, predicates, waits, matrix sequences (35)**

| Control | What is made wrong | Objected to (cubins objected to, of those it changes) |
|---|---|---|
| `unsaved` | A predicate that stands for a uniform one while it holds something of the program is not saved before nor given back after | default: 10 of 10 |
| `blind` | The stand-ins of uniform predicates are chosen as if no ordinary predicate held anything | default: 22 of 22 |
| `cut` | The lives of a uniform predicate are cut at every instruction: what one instruction writes, the next reads in what may be another predicate | default: 25 of 25 |
| `masks` | The P2R and R2P of the program are taken to read and write no predicate (the predicates are a mask, a number: the tables do not say which) | default: 2 of 2 |
| `nowait` | The instruction after one that reads late a register of the translation does not wait for it | default: 30 of 30 |
| `late_rule_nowait` | A late writer (`MIO_FAST` or `MIO_SLOW`) translated from a non-late instruction under no scoreboard (`UFLO` -> `FLO`) does not wait for its write scoreboard after the instruction | default: 1 of 1 (checked-in cubins) |
| `close` | A wait comes one cycle after the instruction that names its scoreboard | default: 35 of 35 |
| `returns` | A number that makes a return address is off by one instruction | default: 22 of 22 |
| `waitless` | What an instruction waited for, the first instruction of what it became does not wait for (ULDC lost its wait that way until a review found it) | default: 35 of 35 |
| `unlisted` | The translation as it is, and a map that leaves out one instruction of each function (what the map does not have, the check did not look at) | default: 38 of 38 |
| `swapped` | Two instructions of the result that follow each other are exchanged, and the map says so: each is what the map says, in another order | default: 38 of 38 |
| `batched` | An instruction with marks of register reuse that gets a stall which does not end its group keeps the marks, which then name a batch (as until 2026-11-17) | default: 1 of 1 |
| `unmoved` | The relocations on code stay at the offsets they had, where other instructions are now (as until 2026-11-17; cuFFT has such kernels) | default: 1 of 1 |
| `named` | The first relocation on code is made to name the function it is on, and the function is translated all the same: the place it names would have moved | default: 1 of 1 |
| `unguarded` | The loads that stand for a matrix load under a guard have no guard | default: 6 of 6; `=8`: 3 of 3 |
| `lanes` | In the sequence that stands for a matrix load or product, a value is taken from another lane, or another part of the lane's number | default: 13 of 13; `=8`: 9 of 9 |
| `unwaited` | In such a sequence, what reads the result of shuffles does not wait for them | default: 13 of 13; `=8`: 9 of 9 |
| `unordered` | Where a store before a matrix load reads, as nothing orders it, a destination that the sequence writes with arithmetic first: the lanes of the rows of a plain load go into its destinations, and the arithmetic of a transposed one does not wait for the shuffles before it (as until 2026-11-23: the store may store the lane) | default: 1 of 1; `=8`: 1 of 1 |
| `complement_rewritten` | In the ORIGINAL, where an instruction under the complement of the guard of a store writes a register that the store reads as data, with nothing waited for between them, an instruction between them that can write a predicate besides its result writes the store's predicate, and that program is translated as it is: the write then runs in lanes where the store may not have read the register yet | default: 1 of 1 |
| `complement_same` | In the ORIGINAL, such a write is put under the store's own guard, and that program is translated as it is | default: 1 of 1 |
| `soon` | The selections after the last step of a product come two cycles after it, with a wait for its scoreboard, and not eight (on a V100 they then read registers that are not written yet) | default: 11 of 11; `=8`: 7 of 7 |
| `partial` | R2P is taken to read its register only, not the predicates: two predicates given back one after the other are one cycle apart (as until 2026-11-18: the second R2P writes the first's predicate back as it was before), and a predicate that an instruction just wrote is not waited for | default: 1 of 1 |
| `boardless` | A predicate saved, or all of them for R2P, is read without waiting for the scoreboard that the instruction that wrote it announces it on | changes nothing in these cubins |
| `unmasked` | An R2P of the program is taken to write what its operands say, no predicate (as until 2026-11-18): a predicate that the translation saves right after it is saved as it was before | changes nothing in these cubins |
| `early` | A predicate saved in a register is given back as soon as the instruction before allows, and not nine cycles after the last one that read what stood in it (on a V100 that reader then goes by the value given back: PyTorch's f32 attention faulted, 2026-11-18) | default: 9 of 9 |
| `unheld` | A uniform predicate held in a register is not read back before the instructions that read it: they go by whatever their predicate held | default: 5 of 5 |
| `unwritten` | It is not written down in the register after the instructions that write it | default: 5 of 5 |
| `stale` | Every product after the first that gathers takes the elements of B where that one left them, whatever happened since: the register of B written, a way in from elsewhere, a call, another register of B gathered | default: 7 of 7 |
| `unkept` | Of what a function with a product keeps of the lane, the lane that holds what a thread gathers for the columns 4 to 7 is 8 further and not 16 | default: 7 of 7 |
| `slowno` | The instructions that do nothing, before what writes or saves a slow predicate, are left out | default: 6 of 6 |
| `slowshort` | They last 3 cycles where 13 are needed and 15 where 26 are | default: 4 of 4 |
| `slownowait` | The first of them does not wait for the six scoreboards | default: 6 of 6 |
| `slownever` | Slow predicates are taken like the others, with no wait: among them one that is written late under no scoreboard, which may never stand for a uniform predicate | default: 6 of 6 |
| `slowfree` | No wait before the instructions that write a uniform predicate whose stand-in, in a life where it is free, is slow | default: 6 of 6 |
| `slowsaved` | No wait before the first instruction of a life whose stand-in is slow and saved | default: 1 of 1 |
| `alive_scratch` | The predicate that a minimum or maximum computes into is one that holds something of the program there | default: 18 of 18 |

**Pairs of products on whole rows (13)**

| Control | What is made wrong | Objected to (cubins objected to, of those it changes) |
|---|---|---|
| `rowlane` | One of the values kept of the lane for the rows is computed with another shift | default: 5 of 5 |
| `otherrow` | A whole-row load takes its second half of k where the first is | default: 5 of 5 |
| `rowwait` | The loads of rows do not wait for the shuffles that bring their addresses | default: 5 of 5 |
| `rowread` | The loads of rows do not say when they have read their address | default: 5 of 5 |
| `halfswap` | A half of a pair writes the accumulators of its two products the other way round | default: 5 of 5 |
| `halfk` | The first product of a pair is done on the second half of k and the second on the first | default: 5 of 5 |
| `inplace` | The first product of a pair adds to the destinations, whatever the two products of the program add to (changes something only where a product does not add in place) | default: 1 of 1 |
| `stepfast` | The steps of a half of a pair are one cycle apart | default: 5 of 5 |
| `unsettled` | The conversions of accumulators start two cycles after the instruction that waits, not fifteen | default: 5 of 5 |
| `unsynced` | The instruction before conversions waits for the products' scoreboard only (as until 2026-11-18: a register that a load of the program was still filling was converted) | default: 5 of 5 |
| `unpermuted` | The conversions use the permutation of the other way | default: 5 of 5 |
| `unconverted` | The conversions of the first place that has some are left out: every sequence is as the design has it, and the plan is wrong (the checker of plans, qcheck.py, must object: nothing else looks at the states of the accumulators) | default: 5 of 5 |
| `notalso` | A load whose result in Turing's layout something else reads does not give it (the same: only the checker of plans knows who reads what) | default: 5 of 5 |

**Borrowed registers (10)**

| Control | What is made wrong | Objected to (cubins objected to, of those it changes) |
|---|---|---|
| `alive_pair` | The pair of registers given at an instruction is the lowest one of which a register holds something there | default: 1 of 1; `VF_BORROW=1`: 10 of 10 |
| `alive_home` | The home of each uniform register is the register after the one chosen | default: 2 of 2; `VF_BORROW=1`: 17 of 20 |
| `no_borrow_wait` | Where the first instruction of what an instruction became waits for the six scoreboards, it waits for what the original waited for only | default: 9 of 9; `VF_BORROW=1`: 26 of 28 |
| `no_late_borrow` | An instruction of the program that reads a borrowed register late does not say when it has read | default: 9 of 9; `VF_BORROW=1`: 20 of 20 |
| `late_writer` | A late writer (`MIO_FAST` or `MIO_SLOW`) of the translation that writes a borrowed register does not say when it writes (2026-11-27) | default: 1 of 1; `VF_BORROW=1`: 8 of 8 (checked-in cubins) |
| `written_pair` | The pair given at an instruction is one that the original instruction writes | default: 1 of 1; `VF_BORROW=1`: 9 of 10 |
| `shared_home` | What the homes take is forgotten once they are chosen: a register given at an instruction may be the home of a uniform register that is alive there | default: 2 of 2; `VF_BORROW=1`: 6 of 6 |
| `overlap_pool` | The same for the pairs only | default: 1 of 1; `VF_BORROW=1`: 5 of 5 |
| `short_home` | The declarations say less than the life of a uniform register in its home: an instruction of the life that neither reads nor writes it is left out (the translation is as it was: what the validator is told is wrong) | default: 1 of 2; `VF_BORROW=1`: 19 of 20 |
| `pool_in_home_unsaid` | As overlap_pool, and no line of the declarations says that the pair is taken where it is the home of a uniform register that is alive | default: 1 of 1; `VF_BORROW=1`: 5 of 5 |

**The frame (31)**

| Control | What is made wrong | Objected to (cubins objected to, of those it changes) |
|---|---|---|
| `no_lane_l3` | The S2R that reads the lane before a matrix sequence is replaced by a NOP | default: 5 of 5; `=8`: 8 of 8 |
| `odd_pair_l3` | Two uniform registers that make a pair are loaded from the frame into registers that begin on an odd one | default: 8 of 8; `=8`: 21 of 21; `=5`: 14 of 14 |
| `no_r1_lower` | The instruction that lowers the stack pointer by the frame is replaced by a NOP | default: 9 of 9; `=8`: 26 of 26; `=5`: 16 of 16 |
| `wrong_r1_lower` | It lowers the stack pointer by twice the frame | default: 9 of 9; `=8`: 26 of 26; `=5`: 16 of 16 |
| `wrong_slot_load` | The first load from the frame takes the word after its own | default: 9 of 9; `=8`: 25 of 25; `=5`: 15 of 15 |
| `no_store_l3` | Nothing is stored to the frame | default: 9 of 9; `=8`: 25 of 25; `=5`: 15 of 15 |
| `held_busy_l3` | The register that holds a uniform predicate between the instructions that name it is one where the program holds something at one of them | default: 1 of 2; `=8`: 3 of 4; `=5`: 3 of 3 |
| `unkept_holder_l3` | That register is not kept from the sequences: one of them may compute into it while it holds the predicate | `=8`: 3 of 3; `=5`: 2 of 2 |
| `shared_holder_l3` | The uniform predicates held in registers are all held in the first one's, whether or not they are held at the same instructions | `=8`: 3 of 3; `=5`: 1 of 1 |
| `home_unsaid` | The declarations leave out the home of one of two uniform registers that have one register for home at different instructions (the translation is as it was: the register then stands for that uniform register in the whole function, for all the validator is told) | `=5`: 1 of 1 |
| `guarded_store_l3` | The stores to the frame after an instruction under a guard are under that guard | default: 6 of 6; `=8`: 10 of 10; `=5`: 5 of 5 |
| `no_old_load_l3` | What an instruction under a guard writes is not loaded from the frame before it | default: 6 of 6; `=8`: 10 of 10; `=5`: 5 of 5 |
| `no_load_wait` | What follows the last load from the frame does not wait for it | default: 9 of 9; `=8`: 25 of 25; `=5`: 15 of 15 |
| `store_no_read_sb` | The stores to the frame do not say when they have read | default: 9 of 9; `=8`: 25 of 25; `=5`: 15 of 15 |
| `wrong_base_const` | The base of the frame is taken from another constant | `=3`: 18 of 18 |
| `missing_pool_line` | The last register of the program that the translator says it took at an instruction is not said | default: 9 of 9; `=8`: 30 of 30; `=5`: 20 of 20 |
| `room_lane_unsaid` | Where the lane is read into a register of the program that the translator says it took at the instruction, the declarations say that it puts that register away there instead, and nothing puts it away (the translation is as it was) | default: 5 of 5; `=8`: 8 of 8 |
| `no_attr_raise` | The attributes of the function are the original's, not raised by the frame | default: 9 of 9; `=8`: 26 of 26; `=5`: 16 of 16 |
| `no_borrow_wait_l3` | No instruction of a sequence that reads or writes the frame waits for the six scoreboards | default: 9 of 9; `=8`: 25 of 25; `=5`: 15 of 15 |
| `no_c3_wait` | Where the translation takes a register that the program writes late, the first instruction of the sequence does not wait for the six scoreboards | default: 7 of 7; `=8`: 21 of 21; `=5`: 16 of 16 |
| `store_before_l3` | The stores to the frame come before what the instruction became, with the waits that this order asks for: they store what was there, or what the register held | default: 9 of 9; `=8`: 25 of 25; `=5`: 15 of 15 |
| `store_other_l3` | Where an instruction only writes a uniform register of the frame, the register next to the one it computed into is stored | default: 9 of 9; `=8`: 25 of 25; `=5`: 15 of 15 |
| `stray_store_l3` | After an instruction that writes a uniform register of the frame, the same value is stored to another uniform register too, which the instruction does not write | default: 9 of 9; `=8`: 24 of 24; `=5`: 15 of 15 |
| `standin_home_l3` | A uniform register of the frame is loaded into the home of another uniform register that the instruction reads or writes, and read from there | default: 2 of 2; `=5`: 12 of 12 |
| `standin_operand_l3` | A uniform register of the frame is loaded into an ordinary register that the instruction itself reads, and read from there | default: 9 of 9; `=8`: 21 of 21; `=5`: 13 of 13 |
| `standin_shared_l3` | Two uniform registers of the frame that an instruction reads are loaded into one register | default: 6 of 6; `=8`: 22 of 22; `=5`: 12 of 12 |
| `base_after_use_l3` | The top of the frame is loaded after the first load or store that takes its address from it, with the waits that this order asks for: that access is addressed from what the register held before | `=3`: 18 of 18 |
| `up_standin_l3` | A uniform predicate kept in the frame is loaded, after the uniform registers of the frame that the instruction reads, into the register that stands for the first of them, and read back from there | default: 1 of 1; `=8`: 1 of 1; `=5`: 2 of 2 |
| `up_wrong_word_l3` | A uniform predicate kept in the frame is loaded from the word after its own | default: 2 of 2; `=8`: 1 of 1; `=5`: 2 of 2 |
| `up_no_store_l3` | What an instruction writes of a uniform predicate kept in the frame is written down in its temporary register and not stored | default: 2 of 2; `=8`: 1 of 1; `=5`: 2 of 2 |
| `up_unwaited_l3` | What reads back a uniform predicate loaded from the frame does not wait for the load | default: 2 of 2; `=8`: 1 of 1; `=5`: 2 of 2 |

**Room (18)**

| Control | What is made wrong | Objected to (cubins objected to, of those it changes) |
|---|---|---|
| `room_reads_orig` | The first register put away is one that the original instruction reads | default: 7 of 7; `=5`: 12 of 12 |
| `room_home_alive` | The first register put away is also declared the home of a uniform register | default: 7 of 7; `=5`: 13 of 13 |
| `room_no_back` | The first load that brings a register back is replaced by a NOP | default: 7 of 7; `=5`: 13 of 13 |
| `room_guarded_away` | The first store that puts a register away is under a guard: where it does not hold, the register is not put away, and it is brought back all the same | default: 7 of 7; `=5`: 13 of 13 |
| `room_lane_first` | Where the lane is read into a register put away at the instruction, it is read before the stores that put registers away, which wait for it: the register is put away with the lane in it, and brought back so | default: 1 of 1 |
| `room_back_wrong_word` | It brings the register back from the word after its own | default: 7 of 7; `=5`: 13 of 13 |
| `room_away_after_write` | The first store that puts a register away comes after the first instruction of the sequence | default: 7 of 7; `=5`: 13 of 13 |
| `room_same_word` | Two registers are put away in one word | default: 6 of 6; `=5`: 12 of 12 |
| `room_at_branch` | R2 is declared put away at the first branch or exit of the function, where nothing puts it away | default: 7 of 7; `=5`: 13 of 13 |
| `room_depth_wrong` | The first store that puts a register away is where it would be if the program had not moved the stack pointer | default: 1 of 1; `=5`: 3 of 3 |
| `room_frame_short` | The frame is declared with the size of the uniform values only | default: 2 of 2; `=5`: 1 of 1 |
| `room_missing_lives` | The registers put away are not declared | default: 7 of 7; `=5`: 13 of 13 |
| `room_no_back_wait` | What follows the last load that brings a register back does not wait for it | default: 7 of 7; `=5`: 13 of 13 |
| `room_scratch_busy` | At an instruction with room, the scratch register of a matrix load is R14, which the program uses and which is not put away | default: 2 of 2 |
| `copy_after_write` | In the ORIGINAL, the copy of a return address and the write that follows it are exchanged, and that program is translated as it is: the validator can no longer work out what the call reads and writes, and must not take registers for free where the translator did (no objection where the translator did not) | - |
| `room_back_first` | The registers put away are brought back right after they were put away, before the sequence that uses them, with the waits that this order asks for | default: 7 of 7; `=5`: 13 of 13 |
| `room_back_before_store` | A register put away that stands for a uniform register of the frame which the instruction writes is brought back before that register is stored | default: 2 of 2; `=5`: 6 of 6 |
| `room_away_after_load` | A register put away that a load from the frame writes is put away after the load: what is put away, and brought back, is the uniform register | default: 4 of 4; `=5`: 10 of 10 |

**Products at level 3 (7)**

| Control | What is made wrong | Objected to (cubins objected to, of those it changes) |
|---|---|---|
| `product_odd_before_l3` | `odd` is computed before the gathering, which still reads `half` in that register | default: 5 of 5; `=8`: 7 of 7 |
| `product_busy_take_l3` | The register of the A operand is declared taken at the product, where it holds the operand | default: 5 of 5; `=8`: 7 of 7 |
| `product_odd_pair_l3` | The first pair taken for a product begins on an odd register | default: 4 of 4; `=8`: 7 of 7 |
| `product_no_s2r_wait_l3` | The first instruction that reads the lane does not wait for the S2R that reads it | default: 5 of 5; `=8`: 7 of 7 |
| `product_soon_l3` | The selections come two cycles after the last step of a product, not eight (thirteen for halves) | default: 5 of 5; `=8`: 7 of 7 |
| `load_ur_wrong_slot_l3` | The first load of a uniform register before a matrix sequence takes the word after its own | default: 4 of 4; `=8`: 2 of 2 |
| `load_ur_no_wait_l3` | The last such load names no scoreboard, and what follows does not wait for it | default: 4 of 4; `=8`: 2 of 2 |

**What changes nothing.** `boardless` and `unmasked` each need a predicate that the translation
saves right after an instruction that writes it in a particular way (on a scoreboard; by an
`R2P` of the program): none of the code at hand has one. Until they are run on code that has
what they take away, the rules behind these two controls have not been seen to object.
`wrong_base_const` was a third until 2026-11-21. It and `base_after_use_l3` need a frame
addressed from a register that is loaded with its top. At the settings that the layer uses,
the translator does that only where it cannot follow the stack pointer: none of PyTorch's
kernels is one, and on the example known, a kernel of 18,000 instructions in vLLM's compiled
extension, a control had not ended after two hours of processor time and was stopped. So the
two are run at `VF_BORROW=3`, where every frame is addressed that way.

**Of another kind.** `copy_after_write` does not make the translation wrong: it changes the
program (the copy of a return address exchanged with the write that follows it) so that the
validator can no longer work out what a call reads and writes, and must then refuse the
registers that the translator would borrow across it. In the 85 of PyTorch's cubins that it
changes it meets no objection of that kind (four are objected to for the exchange itself,
which leaves a wait one cycle after its scoreboard): this control has shown nothing yet either.

**What the controls found in the validator.** Run each in a process of its own for the first
time on 2026-11-20, they showed three faults of the validator, corrected since: a register
declared put away at an instruction where nothing stored it or brought it back passed; a plan
of pairs with a half that had no other half ended the validator on an exception instead of an
objection; and in a function that borrows registers, each half of a pair was objected to for
writing the destination of its other product, which is what it must do. Controls find such
faults where someone thought of a control: the 106 above are the measure of what was thought
of. (Nine of them were written on 2026-11-20 for the rule on what the frame's moves carry,
after a reading by someone else had shown two wrong translations that passed: section 7.
They make the translator put its instructions in a wrong order, or take a wrong register,
and then schedule the result itself: the order or the register is the only fault.) Asked on
2026-11-23 to be objected to in every cubin whose translation they change, they showed one
more: with every store to the frame left out (`no_store_l3`), two functions of PyTorch's 0150 at
`VF_BORROW=5` passed, as their instructions that write uniform registers of the frame read
none, and nothing moved the frame around them: the rule that every uniform register the
original writes is stored was asked only where something did. It is asked of every instruction
of a function with a frame since. And three controls did something else than they say:
`no_load_wait` patched the code of the first function at the offsets of the others'
instructions, and looked at another scoreboard than the one whose wait it took away;
`no_c3_wait` took the declarations, the instructions and the registers written late of every
function at once; `missing_pool_line` left out the last line of the declarations even where it
names a register above the program's own, which the translation may write without saying so.
All three do what they say since.

**Rules without a control of their own.** Some things that the validator asks since
2026-11-20 have not been made wrong on purpose: that the depth of the stack pointer be known
where the frame is reached through it; that nothing but the save write a register in which
predicates are saved, whatever kind of instruction writes it; that the upper half of a
product read the register loaded from the frame for it, and no other; and, of what the
frame's moves carry, that a uniform register be loaded once and stored once for an
instruction, and not loaded after it was stored.

## 4. The corpora

Two corpora of sm_75 machine code are translated by the reference translator and checked by the
validator, cubin by cubin (`tests/sass/validate-corpus.sh`). The rules of the translator were
found on the first; the second was kept apart to test them. Triton's cubins at hand are checked
the same way (below).

**PyTorch.** The 475 cubins of `libtorch_cuda.so` of PyTorch 2.14.0+cu130 (the stock wheel;
`vf extract`): 23,693 functions, of which 29 are refused. At the default (2026-11-23, with the
translator and the rules of 0.1.3, and again with the rules as they stand): **475 of 475 cubins
at 0 problems, with no check left unmade**: 31,034,665 instructions of 23,664 functions (a
quarter of an hour with 44 processes). The counts that
follow are of 2026-11-20, when 23,663 functions were translated. Among what the validator counts as checked: 18,315 lives of uniform
predicates against nvdisasm's chart; 2,354 borrowed registers and 70,030 first writes of them
(54,443 behind a wait that the rule asks for); 52 homes of uniform registers whose declared
life is the whole life; 65 functions with a frame; 573 instructions with room made, in 33
functions; 119,486 instructions of matrix sequences; 1,920 products that take everything at
the instruction; 672 products that take what an earlier one gathered.

**The held-out corpus.** The sm_75 code of 42 libraries: those of CUDA 13.5, cuDNN, cuSPARSELt,
NCCL, NVSHMEM and the CUDA libraries of the PyTorch wheel, and vLLM's compiled extensions
(`tests/sass/heldout-extract.sh`): 3,108 cubins, none of which is byte for byte one of
PyTorch's. At the default (2026-11-23, with the translator and the rules of section 2 as they
stand): **3,105 of 3,108 cubins at 0 problems, with no check left unmade**: 52,551,415
instructions of 60,310 functions (an hour with 24 processes). The other three cubins are not
read at all by either translator (a section table that points outside the file; two with two
sections of code of one name). More functions are refused here than in PyTorch (`LIMITS.md`,
"Kernels refused"). None of this code was run. The first run after the sequence of a matrix
load was changed and the rules that came with it were added (2026-11-23, section 2) objected
2,713 times in 26 cubins, to two of those rules and for the rules' sake: 1,443 times in 12
cubins of cuDNN and vLLM to the lane of a product at level 3 read into a register put away at
its instruction (rule D took a register taken there, and not one put away: "The whole truth"),
and 1,270 times in 14 cubins of cuSOLVER, NCCL and vLLM to a write under the complement of the
guard of the store that reads the register (`data_overwritten`, which now leaves out these
1,270 writes). Looking at what makes the first right found a hole in the rules of room, closed
since: a register put away under a guard passed (control `room_guarded_away`).

**Triton.** The 346 sm_75 cubins that Triton made at run time in this site's runs, as found in
their caches (2026-11-23, with the translator and the rules of 0.1.3): **346 of 346 at 0
problems** (307,059 instructions of 334 functions). In 12 of these cubins, all variants of one
`tl.dot` kernel, the rule `data_overwritten` objects to the translation of 0.1.2 and earlier.
Then (2026-11-23, after 0.1.3) the cubins that Triton made for the corpus of section 6
("Triton"): **4,308 distinct cubins** of 20 kernels in 3,516 configurations, and the caches of
this site again (334 distinct cubins). The two translators give the same bytes for all of them
(`compare.sh`); the translator refuses 114 functions of the first (84 reductions whose layout
conversion is an `LDSM` at an address without register, `LDSM [RZ]`; 24 for registers, two of them
since 2026-11-23, for the registers that the block their cubin requires allows: the two cubins of
the gather of 16 warps of section 6; 6 for `UP2UR` with an immediate) and 7 of the second (6 for
registers, one for `STL` with a uniform register), and the layer runs the PTX kept in those cubins
instead. The validator: 334 of 334 at 0 problems (278,681 instructions of 327 functions), and
**4,308 of 4,308 at 0 problems** (2,876,544 instructions of 4,196 functions, the gather's two
translations among them, 10,865 instructions each: refused since) once a check of the whole
function was told that the registers
a matrix load at level 3 borrows are borrowed at its instruction alone: it objected for that
alone to two cubins, one `tl.dot` in float32 with 252 registers in two precisions, whose
translation was right on the V100 and is checked instruction by instruction by the rules that
check every borrowed register. Again with the rules as they stand (2026-11-23, after the changes
to `data_overwritten`, rule D and room of section 2): 4,308 of 4,308 at 0 problems (2,854,814
instructions of 4,194 functions: the gather's two are refused) and 334 of 334 (278,681 of 327).

This corpus showed two faults that PyTorch's code could not, the first time it was translated:
marks of register reuse that came to name a batch when the schedule was lengthened (225
instructions in 35 cubins of cuBLAS, cuSOLVER and cuDNN), and relocations on code that stayed
where they were when instructions moved (146 functions, 145 of them kernels of cuFFT). The
controls `batched`, `unmoved` and `named` are for these.

**The second translation.** A kernel that the driver refuses to launch for its registers is
launched from another translation of its module, made within each kernel's own count of
registers (section 6). Translated that way (`VF_BORROW=5`), PyTorch's 475 cubins are at 0
problems, with no check left unmade: 31,636,194 instructions of 23,651 functions, of which
4,920 keep uniform registers in a frame and 3,752 make room at 100,799 instructions; 17,187
homes whose declared life is the whole life of their uniform register. The held-out corpus
translated that way (2026-11-22, with the rules of that day): 3,105 of its 3,108 cubins at 0
problems, with no check left unmade
(50,625,696 instructions of 59,824 functions, of which 14,761 keep uniform registers in a
frame and 5,742 make room); the other three cubins are the ones that are not read.

Until 2026-11-20 the validator objected to 55 of PyTorch's cubins translated that way, 320
objections of six kinds. Each kind was put down to the rule or to the translation:

| What was objected to | Printed | What it was |
|---|---:|---|
| A register left under the kernel's count that is the home of a uniform register at some instructions and that sequences compute into at others | 203 | The rule, written for the usual translation, where such a register is one or the other in the whole function. Behind it, a hole: the life that the translator declares for a home was taken on trust. The validator now works out the lives of uniform registers itself (section 2, "The life of a home") |
| The upper half of a product with an operand loaded from the frame | 11 | The rule, which knew one register for a uniform operand in the whole function |
| Room made where the validator did not know how far the stack pointer was | 12 | The validator: its dataflow for the stack pointer did not follow jump tables. Behind it, two holes: an access to the frame through the stack pointer where its depth was not known passed without a word, and a place reached at two depths kept the first |
| A uniform predicate held in a register of the program, in a function without a frame | 6 | The rule, which allowed it with a frame only. And nothing asked that such a register be free along the held life: added |
| A minimum or maximum with an operand loaded from the frame | 4 | The rule on where the save of a predicate stands, which counted the load; and behind it the rule on what follows the last load, which looked at the wrong instruction |
| A population count that reads a register loaded from the frame, without the next instruction waiting until it has read | 3 | **The translation.** An instruction that reads the home of a uniform register late says when it has read, and the next one waits; a register loaded from the frame for one instruction had been left out of that. Corrected in both translators |

One more fault of the translation was found at the test settings: a texture load of the
program that reads a borrowed register is made to say when it has read, and lost its batch
on the way (two instructions of one cubin of cuDNN). Corrected, and that cubin is at 0 at
every setting.

The held-out corpus had never been translated that way. Its first run, with the rules as they
stood after PyTorch was at 0, had objections in 14 of its cubins (cuBLAS, cuBLASLt, cuSPARSE,
cuSPARSELt and cuDNN): 57 printed, of four kinds that PyTorch's code does not show, and all
four were rules:

| What was objected to | Printed | What it was |
|---|---:|---|
| An instruction that does nothing right after a store to the frame, and one right after a register brought back from it, that do not wait | 13 and 15 | The rules, which looked at the very next instruction. The schedule had put an instruction that does nothing there, for time; the instruction after it waits, as the general rule of the schedule already allowed |
| A register that holds one uniform predicate along some instructions and another along others | 20 | The rule, which looked a register up for the first predicate held in it anywhere in the function. It is looked up instruction by instruction now, and two predicates in one register at one instruction are objected to |
| Two uniform registers with one register for home, above the program's registers and under the kernel's count, at different instructions | 9 | The rule, which allowed that among the program's registers only |


The rule on what the frame's moves carry (section 2) came after all of this, from two readings
of the validator by someone else (section 7): the values that the moves carry first, then the
addresses that they are made at. The four runs were made again with each state of the rule
(2026-11-21, twice): the same counts, and no objection in either corpus at either setting.
The orders that it refuses are not ones that the translator makes.

"0 problems" is the validator's verdict by the rules of section 2, with the limits given there.
Section 8 says what it is not.

## 5. Against NVIDIA's disassembler

The translator and the validator both rest on descriptions of the two instruction sets that are
not NVIDIA's documentation (`docs/translator.md`, "The instruction tables"). Three scripts
compare what the project makes of them with what NVIDIA's own disassembler says of the same
code.

**The decoder** (`reference/check.py`). For every distinct instruction word of the cubins it is
given: the class found must be unique, its mnemonic the one that nvdisasm prints, and its
registers and predicates those of nvdisasm's text. On PyTorch's 475 cubins (2,935,225 distinct
words of sm_75) and on the 418 cubins of the last PyTorch wheel that has code for Volta
(3,166,930 distinct words of sm_70): one class and the right mnemonic for every word, and the
same registers and predicates for all but 871 and 885 words of two forms of `BMOV`, where
nvdisasm names a barrier register (`B0`) that the comparison does not read from the word.

**What each instruction reads and writes** (`reference/uses_check.py`). nvdisasm draws the lives
of the registers of a function: under each register, for every instruction, whether it reads it,
writes it or both. The schedule of a translation rests on the same knowledge: a register that
the tables do not say an instruction reads is a wait that is not inserted. On PyTorch's sm_75
code: 29,791,744 instructions compared, none differs (192,976 calls are left out: nvdisasm marks
there what the called code may change; 75,168 instructions have no line in the chart, all but
eight of them the NOPs that fill the end of a function). On the native sm_70 code of the
wheel named above: 30,743,200 compared, none differs. On the
held-out corpus the comparison was made once (2026-11-18): of 72.6 million instructions, 43,088
differ; they were put down to the cubins that nvdisasm reads in its older format, without its
analysis of lives (317 of them), and were not traced one by one.

**Where each instruction may lead** (`reference/flow_check.py`). The translator and the
validator each work out the control flow of a function by themselves, the first from the
descriptions and the jump tables of the container, the second from the mnemonics and the branch
targets that nvdisasm prints. An edge that the translator lacks is a path on which a predicate
it took for free may be read. On PyTorch's 475 cubins: 29,866,912 instructions, the same places
for both at all but 12, which are the 12 branches through a uniform register of that code
(`BRXU`): the translator does not read their tables, and refuses the ten functions that have
one.

**The lives of registers.** A register that the translation borrows must hold nothing of the
program. The validator works the lives of registers out itself and objects to a borrowed
register that it finds alive or written (section 2). The lives that the translator works from
were also compared with nvdisasm's chart on both corpora (2026-11-18, by a script that is not in
this repository): 29.8 million instructions of PyTorch's 475 cubins and 72.0 million of the
held-out corpus (nvdisasm draws no chart for 317 of its cubins). Where the translation of an
instruction needs two scratch registers, no even pair that the translator calls dead has a mark
in the chart: 7.9 million pairs in PyTorch, 33.0 million in the held-out corpus. Where the
translator says dead and the chart alive, the places are of three kinds: the chart's convention
for calls (it marks at a call every register that the called code reads or writes), branches
that are always taken, and writes that nothing reads: 17,801 places in PyTorch and 85,935 in
the held-out corpus where an instruction writes a register that is dead before it and after
it. That last kind is why a register counts as free at an instruction only if the instruction
does not write it.

A function that others call may be entered with values of its caller in registers that it never
names. Every function of PyTorch's 475 cubins is a kernel; 1,999 functions of the held-out
corpus are not (helpers such as `__cuda_sm70_warpsync`, which kernels of other files call). Both
translators refuse a function that no kernel of its file reaches when its translation needs
registers above those it declares.

## 6. On the hardware

Everything in this section ran on a V100 through the layer as it is built from this tree
(`tests/README.md` says how to run each suite and what it needs).

### Translated kernels against native builds

`tests/gpu/run.sh` compiles each source of `tests/kernels/translate/` for sm_70 with CUDA 12.9.2
and for sm_75 with CUDA 13.5.0. A test program loads both; the sm_75 cubin goes through the
layer, and each kernel must give what the sm_70 build gives on the same inputs. The native build
is compared with itself first: a kernel whose result depends on the order of its threads would
fail there, and would prove nothing.

Thirteen source files, 65 kernels: all the same as their native builds. They cover uniform
registers and predicates, calls, loops, minima and maxima, jump tables, memory instructions,
conversions, instructions on uniform registers under a guard, and `UIMAD` with an immediate,
`UIMAD.WIDE` (`k_uimad.cu`, since pull request #15), and `UFLO` (`k_uflo.cu`), one kind of rewriting each.

Two of the files run a second time from a cubin changed by hand into something that compilers
do not write, or write rarely, and that a translation must still get right:

- the seven memory kernels with another, equivalent schedule
  (`tests/kernels/translate/waits_moved.py`: where a load is followed by a `ULDC` and then by
  its first reader, the wait is put on the `ULDC`), because a translator that loses the wait of
  an instruction that becomes several gives wrong sums in two of them, and only with that
  schedule;
- the three guarded kernels with every uniform register kept in the thread's local memory
  (`VF_BORROW=8`), and with 28 of their guarded instructions giving a predicate into the one
  that guards them (`tests/kernels/translate/guard_written.py`; nothing reads that predicate
  after, so the program computes the same). Until 2026-11-20 the store that the translation
  adds after such an instruction was under the instruction's guard, went by the predicate as
  the instruction had left it, and did not happen: two of the three kernels gave wrong results
  on a V100 (18,441 of 65,536 bytes of one, every byte of the other). No compiled code at hand
  has such an instruction on a uniform register: of the 103.5 million instructions of the two
  corpora (section 4), 323 write the predicate that guards them, and none of those writes a
  uniform register.

### Rewritten instructions against host models

`tests/kernels/tcrun.c` loads a module through the layer, runs every kernel it knows in it and
compares each result register with a model of the instruction on the host, written from the
definitions of the PTX manual. The inputs are of two kinds: values on which the answer is
decided (halfway points of a rounding, the limits of a range, sums that are exact in any order),
which must give the model's result exactly, and random values, which must be within the
rounding of the sums.

| Module | Kernels | What |
|---|---|---|
| Tensor-core instructions of Turing, as PTX | 37 | every form of `mma` on FP16 and int8 that is rewritten, `ldmatrix`, `movmatrix`, with operands in registers and loaded from shared memory |
| the same with `VF_ALIGNED_ROWS=0` | 42 | the addresses of the rows are tested when the code runs: the kernels whose rows are not at multiples of 16 run too |
| Chains, as PTX | 7 | products chained on one accumulator |
| Products of Ampere, as PTX | 31, and 35 with `VF_ALIGNED_ROWS=0` | int8, bf16, tf32 and f64 in Ampere's shapes |
| FP8, as PTX | 19 | conversions both ways and products on E4M3 and E5M2 |
| FP4 and scales, as PTX | 10 | conversions of E2M1 and UE8M0 |
| bf16, as PTX | 15 | arithmetic, comparisons, the conversion from f64, atomic additions |
| Later instructions, as PTX | 3 | arithmetic on pairs of floats, cache policies |
| Tensor-core instructions of Turing, as machine code | 32 | the kernels of the first row compiled for sm_75 and translated: 28 of its 37 functions are (those with products on integers or `MOVM` are refused when the program asks for them) |
| Kept values, as machine code and as PTX | 5 and 5 | products that take what an earlier one gathered |
| Groups, as machine code and as PTX | 22 and 26 | eight products a group, accumulators chained along loops; 9 of 11 functions are translated |
| Pairs on whole rows, as machine code and as PTX | 34 and 37 | pairs of products in several shapes of loop; 11 of 12 functions are translated. Again with `VF_PAIRS=0`, each product by itself: 34 |

None is wrong. On values that are not exact, the tensor-core products of the layer and those of
later hardware do not round alike: `docs/emulations.md` says how far apart they are.

### Reference vectors from an RTX 5090

The conversions of E2M1 (with and without `.relu`) and of UE8M0 are compared with files that an
RTX 5090 computed (`tests/kernels/blackwell/ref/`): from these formats, on every value they
have; to them, on every float at which the result changes, with a float on each side. Every byte
is the same.

The block-scaled products are compared with vectors that an RTX 5090 computed
(`tests/kernels/blackwell/ref_scaled/`, measured 2026-11-18), and each form is held to its
present counts: a change that moves one result further from the hardware fails the test.

| Form | Results | Identical | One unit of the last place | Further |
|---|---:|---:|---:|---:|
| NVFP4 (`kind::mxf4nvf4`, scales in UE4M3) | 36,352 | 36,352 | 0 | 0 |
| the same, larger set | 406,912 | 405,861 | 1,051 | 0 |
| MXFP4 (`kind::mxf4`, scales in UE8M0) | 36,352 | 36,242 | 106 | 4 |
| the same, larger set | 394,112 | 371,635 | 21,636 | 841 |
| MXF8F6F4 (`kind::mxf8f6f4`, E4M3 values) | 36,352 | 24,272 | 8,059 | 4,021 |
| the same, larger set | 387,712 | 308,475 | 52,538 | 26,699 |

`docs/emulations.md` says where the differences come from (Volta's product adds four terms at a
time in f32, the later hardware in a wider accumulator).

### PyTorch

`tests/gpu/torch.sh` runs 925 operations of PyTorch 2.14.0+cu130, a stock wheel that has machine
code for sm_75 and later and no PTX: every kernel it launches on a V100 was translated. Each
case computes on the CPU and on the GPU from the same inputs and compares: equal for integers
and booleans, within a tolerance for floats. With cuBLAS 12.9.2 in front of the wheel's own
(`INSTALL.md`):

- **924 of 925 give the CPU's result**; the other one is not compared, because the CPU does
  not do it. None is refused, none is different. The cases are functions of one and of two
  tensors in f32, f64, f16 and the integer types, conversions between types, reductions, scans,
  sorting, indexing and copying, random numbers, pooling, normalisations, losses, convolutions
  (cuDNN), transforms (cuFFT) and matrix products (cuBLAS).
- **32 of 32 small whole models** (`tests/gpu/torch-models.sh`): an image classifier, a
  transformer encoder and a causal language model with heads of 64 and of 128, recurrent
  networks, an upsampling network; each in f32, under autocast (f16 on the GPU) and through a
  few steps of training.
- **Told 8.0** (`VF_CAPABILITY=8.0`, 2026-11-22): the same, 924 and the one, 32 of 32, set by
  hand and through the launcher. PyTorch is then still told 7.5, because its code is the
  translation of its sm_75 machine code (`docs/architecture.md`, section 4). Until 0.1.3 it was
  told 8.0 and gave 24 wrong attention cases and 28 wrong models with no error (section 7).

The workers of the two programs write their verdicts on a channel of their own, so that what a
kernel prints cannot hide or forge one, and a case that does not end in `VF_CASE_WAIT` seconds
is stopped and counted as a CRASH (`tests/README.md`). Until 2026-11-22 a verdict printed after
a kernel's message was lost from the count.

With `VF_TRANSLATE_DUMP` the layer writes every cubin that it gives its translator. In one
pass of the battery and one of the models it translated 157 distinct cubins. 122 are, byte for
byte, cubins of PyTorch's corpus of section 4. The other 35, of one function each, are kernels
of the wheel's other libraries, which the extraction of that corpus does not reach: translated
by the reference translator and validated, 76,763 instructions, 0 problems. On every cubin of
the dump the two translators give the same bytes (section 7).

### Triton

Triton makes its kernels while the program runs; told 7.5 (the default), it makes sm_75 machine
code, which the layer translates. Its code has instruction sequences that PyTorch's has not (the
defect of 0.1.2 in matrix loads was found in it, after the battery above had passed).
`tests/gpu/triton-diff.py` runs the kernels of `tests/gpu/triton_corpus.py`, a generator of 3,516
cases of 20 Triton kernels (2026-11-23): element-wise operations in the twelve types (integers of
8 to 64 bits signed and unsigned, float16, bfloat16, float32, float64) on small integers, normal
values, values with infinities, NaN, zeros and subnormals, and every bit pattern; conversions
between every pair of types, with the rounding toward zero and bit casts; reductions (sum,
maximum, minimum, arg-maximum, exclusive or, custom combinations) along each axis of tiles of
eight shapes and 1 to 16 warps; scans (cumulative sums and products, a maximum, a linear
recurrence); sorts, top-k, flips, gathers, histograms, transpositions, joins, interleavings;
`tl.dot` in float16, bfloat16, float32 (`ieee`, `tf32`, `tf32x3`) and float64 with 17 tile shapes,
1 to 8 warps, 1 to 4 stages, operands transposed or not, sizes that are and are not multiples of
the tiles, three epilogues; flash attention; masked, gathered, scattered, strided and reversed
memory, cache hints, block pointers, addresses beyond 2**31 elements; atomics (add, maximum,
minimum, and, or, exclusive or, exchange, compare-and-swap in a lock) with four orderings and two
scopes; loops with trip counts read from memory and loops that end with the data; grids up to
200,000 blocks and 65,535 in y; random numbers; inline PTX; 143 functions of libdevice. Each case
runs on the GPU through the translated machine code (5 runs on the same inputs, which must give the
same bits) and through the PTX that Triton made the same kernel from, retargeted by the layer
(the same compiled kernel and launcher, its module loaded from the PTX): the two must give the same
bits, and both what Triton's interpreter computes on the CPU, bit for bit (inputs that make sums
exact in any order) or within the error the case states (approximate functions, float32 division,
sums of 16-bit floats, which Triton rounds to their type).

On a Tesla V100-SXM2-32GB, driver 580.178.04, with the tree of 2026-11-23 (`fix/triton`): **no
case gives another result through the layer than through the other route or than its reference
allows, and none differs between its runs**, at the default and with `VF_BORROW=5` (what the
second translation at launch translates with), `VF_BORROW=1` and, for the products,
`VF_PAIRS=0`: the four give the same bits for every case. 86 cases are not compiled by Triton for
sm_75 (an int8 product, `div_rn` or `clamp` of integers, a rounding of float32 to bfloat16 toward
zero, a block too large for the shared memory); one, a gather in a sort kernel of 16 warps, was
not launched by Triton because the translation takes 137 registers where its cubin requires blocks
of 512 threads (a V100 allows 384 then: Triton reads that limit from the loaded kernel and stops
before the launch, so the second translation at launch never comes; the kernel takes 128 registers
on sm_75, all its blocks allow, and every level of the translator needs more): since the translator
keeps the block a kernel requires (later the same day), its machine code is refused for its
registers and the layer runs the PTX its cubin keeps, which gives the reference (`PASS-PTX`;
`tests/gpu/triton.sh` has the case: `ERROR` with the translator before, on the V100);
397 have no reference (the interpreter has no libdevice and no exchange of floats; subnormal
inputs, which Triton's instructions flush and numpy does not) and are compared between their runs
and the routes only, and 58 (98 with `VF_BORROW=1`) are refused as machine code and run as the
PTX the cubin keeps. The interpreter differs from Triton's code in places that the cases leave out
or compare otherwise (its bfloat16 conversion truncates, its conversion toward zero makes NaN of
what overflows, its sums of 16-bit floats are wider); Triton's own conversion to 8-bit floats for
sm_75 rounds halves away from zero and makes -0 of a float32 NaN, through both routes alike.

The kernels that vLLM 0.34 and SGLang 0.5.23 ship, launched by `tests/gpu/triton_vendor.py`
(`--vendor`, in the Python of each engine) with the shapes of small and 7B-70B models and with
lengths that are not multiples of the blocks, single tokens, and the largest values: 151 cases of
74 of vLLM's 1,012 `@triton.jit` functions (47 more reached as their helpers: fused MoE in its
plain, GPTQ/AWQ and int8 forms, unified, paged, prefix-prefill and decode attention, merging of
attention states, the KV-cache writer, sampling and rejection sampling, penalties, LoRA, Mamba and
SSD, the gated delta rule, AWQ, scaled products and quantization, the model runner's batch
kernels, batch-invariant operations, mrope) and 90 cases of 49 of SGLang's 1,034 (8 more as
helpers: decode, extend and prefill attention, MoE, its router, alignment and reductions, sampling,
KV-cache copies, norms, activations, the gated delta rule, Mamba, quantization, LoRA, EAGLE), each
5 times through each route (2026-11-23): vLLM 135 PASS, 10 PASS-PTX, 6 not compiled for sm_75 or
too large for the shared memory; SGLang 83 PASS, 5 PASS-PTX, 1 not compiled, and one not launched:
`fused_moe_router_cudacore_kernel` for 128 experts, 16 warps, whose usual translation takes 136
registers where blocks of 512 threads allow 128; its PTX route gives the reference. No case
differs between its runs or between the routes. Since the translator keeps the block a kernel
requires, the router is translated at level 3 within 128 registers (five uniform registers in the
thread's local memory; the validator finds nothing in it, and every control that changes it is
objected to): its four cases PASS on the machine-code route, 5 runs each, with the bits of the
PTX route (2026-11-23).

The test can fail: with the translator of 0.1.2, three of the float32 and bfloat16 `tl.dot` cases
give lane numbers for accumulators, other ones in each run, and three more stop with a misaligned
address (the fault seen when 0.1.3 was prepared). `tests/gpu/triton.sh` runs 180 of the cases,
among them those three, in four minutes.

### The second translation at launch

A V100 counts the registers of a thread in groups of eight, and a block of 1,024 threads may
have 64 registers a thread at most. The usual translation takes some kernels over that
(`docs/translator.md`, "Registers"). When the driver refuses a launch for it, the layer
translates the module again within each kernel's own count and launches the kernel from there.
`tests/kernels/fitrun.c` checks this with kernels made for it: 37 checks of 37 (each kernel's
output against the host's, by each way a program has to take a kernel and to launch it: from a
module or a library, by name or by enumeration, into an array larger than the library's number
of kernels too, through the module that the driver gives for a library, with the plain launch
entry or the one for the per-thread default stream; a kernel that needs more shared memory
than the default, allowed on the library's kernel and launched through the function of the
library's module; and a kernel of a module whose variable the program asked for by name is
refused, as it must be), and 17 of them fail with the mechanism turned off (`VF_FIT=0`). With
a build that does not give a library kernel's settings to the function, the check for it
fails, and with the build of 2026-11-20 the enumeration into a larger array ends in a fault
inside NVIDIA's driver. `tests/proxy/fit_api.sh` goes through the same ways and more without
a GPU, with a stand-in for the driver that refuses the launches of the first image: 39
checks, among them a launch whose second translation is being made while another thread asks
for data of the module, two threads that set the same thing on one kernel at once (the
stand-in takes the first setting and keeps its call from returning until the second has had
its turn: the kernel and its second image must end with the same value), and an unload of the
module from another thread while a setting or a launch of its second image is held (the
unload must wait for them).
In the battery, five kernels are launched that way (`avg_pool2d` and bilinear `interpolate` in
float32 and float64, `unfold` in float32), from three second translations, and their results are
the CPU's. What the validator finds in second translations: section 4.

### llama.cpp

`tests/gpu/llamacpp.sh` runs the official llama.cpp build for CUDA 13.5 (b12418) with a 0.5B
model, with flash attention and without, and compares with the same build on the CPU.
The next-token distributions over a text of a few thousand tokens, processed 512 at a time
(the kernels of batches: flash attention on tensor cores, quantized matrix products): a mean KL
divergence from the CPU's of 0.0017 with flash attention and 0.0017 without (the test's limit is
0.005), and the CPU's most likely token at 96.8 % and 97.5 % of the positions (at least 95 %).
Two right implementations differ that much: the native sm_70 build of the same llama.cpp, on
a text that the test had before, was at 0.0016 and 96.7 %. The 48 tokens generated from a
short prompt were the CPU's in both modes, which is reported and not required: any two
implementations disagree on a few most likely tokens in a hundred.

`tests/gpu/llamacpp-ops.sh test` runs llama.cpp's own backend test on the two operations that
its tensor-core and quantized kernels serve: 4,426 cases of flash attention and 2,688 of matrix
products, each compared with the CPU backend. None differs. (The script counted 4,428 and 2,690
until 2026-11-22: two summary lines of the program were taken for cases.)

With NVFP4 weights (`tests/gpu/llamacpp-nvfp4.sh`; Qwen3.8-27B, converted
to GGUF by llama.cpp's own script): the 48 tokens generated are the CPU's, with flash attention
and without; the next-token distributions have a mean KL divergence from the CPU's of 0.0005
and 0.0005, and the CPU's most likely token at 97.9 % and 98.4 % of the positions.

### vLLM, SGLang, ZML

**vLLM 0.34.0** (`tests/gpu/vllm.sh`, Qwen2.5-0.5B-Instruct, three prompts). `transformers` in
f32 on the GPU gives every token of the CPU; a kernel compiled by Triton gives the CPU's result
bit for bit; the engine's greedy text (f16) is the CPU's (f32) token for token; with sampling
(top-k 50, top-p 0.9), every token it samples is among the 50 most likely and within 0.9 of their
probability by the log-probabilities of its own logits, 150 of 150 (what the kernels of the
sampling do, not what the model computes); its HTTP server, with the repetition penalty that the
model's configuration asks for, gives the CPU's text made with that penalty.

**SGLang 0.5.23** (`tests/gpu/sglang.sh`, the same model and prompts), with its default
attention (FlashInfer): the engine's greedy text is the CPU's token for token, 150 of 150, and
its HTTP server's too. With `--attention-backend triton`: the same from the engine; the server
leaves the CPU's text at one token of 150, where the CPU's own lead over the next token is 0.021
in logit. SGLang makes the model f16 on a GPU that reports a capability under 8: its
log-probabilities are 1e-2 to 7e-2 from the CPU's f32.

**ZML** (`tests/gpu/zml.sh`, the upstream build with its own CUDA libraries, Qwen3.5-9B),
told that the GPU is of capability 7.5 and then 8.0: it answers one prompt with the expected
word. This is the weakest check of the five programs: no token is compared with a reference.

**The forwarding libraries** (`tests/gpu/forward.sh`): ten programs built against the headers of
CUDA 13.5, one for each of cuFFT, cuFFTW, cuRAND, cuSPARSE, cuSOLVER, cuSOLVERMg, cuBLASLt, NPP,
nvJPEG and NVRTC, run with the forwarding libraries in front of CUDA 12.9.2, their results
checked against the host's; first calls from 32 threads at once; and what a function that
CUDA 12.9.2 lacks answers.

### CUDA graph replay

On a Tesla V100-SXM2-32GB with driver 580.178.04 and VF 0.1.1, SGLang's FP16 Triton decode
attention ran with 16 query heads, two KV heads and head width 256. Four KV lengths
(23, 227, 288 and 512) and one or four partitions gave eight graphs. After capture, each
test replaced the query, filled the output with NaNs and replayed the graph. All 32,768
outputs were finite and agreed with the new CPU FP32 references within 0.00028 relative
L2 error. [Replay test](../tests/gpu/sglang-decode-graph.py)

The stock SGLang server used CUDA graphs for prefill and decode on Qwen3.5-35B-A3B-AWQ. The three distinct prompts each used 227 uncached input tokens and returned 64 tokens. [SGLang results](guides/benchmarks/sglang-default-operators.json)

A PyTorch FP32 `im2col` kernel translated with the default register policy used 69 registers. Its CUDA graph ran with 256 threads per block and returned error 701 at 1,024. With `VF_BORROW=9`, the same kernel used 64 registers and ran for three geometries at both block sizes on a V100. All 59,676 results matched the CPU reference bit for bit and all 384 surrounding guard values were intact. These runs measured correctness and block-size admission. [Register policy](architecture.md#translation-control) · [Graph API regression](../tests/proxy/graph_fit.sh)

### Through the launcher

Everything above sets the layer's environment by hand, the way `INSTALL.md` gives under "By
hand", with the forwarding cuBLAS alone in front of the libraries that a wheel carries. A user
puts `vf` in front of the command, and the launcher's table has every forwarding library whose
CUDA 12.9.2 library is there: all 22 after `vf fetch --all`, cuBLAS alone after `vf fetch`. So
the same tests were also run with the environment that the launcher gives (`VF_LAUNCHER=1`),
with all the CUDA 12.9.2 libraries and with what `vf fetch` downloads, and
`tests/gpu/launcher.sh` runs programs through the launcher with nothing else in the
environment:

- PyTorch's stock wheel: a product of matrices against the CPU, with the forwarding cuBLAS
  among the files that the process maps; llama.cpp's official build: text.
- A virtual environment made by `vf venv`, its Python started with no prefix: PyTorch runs on
  the GPU; the layer says that it is there, by its variable and by the functions of
  `libcuda.so.1`; the capability that PyTorch is told is 7.5, and what its builder of
  extensions would have `nvcc` compile for is `sm_75`. The same Python with the hook off is
  told 7.0 and would ask for `sm_70`, which CUDA 13's compiler refuses: a package that
  compiles GPU code when it is installed has to be installed through the layer.
- vLLM, installed with pip in a virtual environment: the five stages of its test, the HTTP
  server among them. SGLang, the same way, with its default attention: every stage, the
  engine's text and the server's the CPU's token for token.
- The 925 operations and the 32 models of PyTorch, with all the CUDA 12.9.2 libraries (the
  transforms then go to cuFFT 12.9.2 through its forwarding library): 924 the CPU's results
  and the one that is not compared, 32 of 32.
- The same with what `vf fetch` downloads, which was run for that (cuBLAS and nvJitLink from
  NVIDIA's servers, 1.0 GB): 924 and the one, 32 of 32. There the launcher's table has cuBLAS
  alone, and a process maps the forwarding cuBLAS, cuBLAS 12.9.2, and the wheel's own cuFFT,
  cuSOLVER and cuSPARSE. A transform, a solve, singular values, eigenvalues, a Cholesky
  factorisation and a sparse product, which go through those three, give the CPU's results
  (to 2e-5 in f32, 2e-12 in f64). The launcher first put every forwarding library in its
  table, whether its CUDA 12.9.2 library was there or not; with these libraries the first
  transform then stopped the program (`cannot load libcufft.so.11`). No test had run it with
  less than all the libraries: this is what trying the README's own commands found.
- The README's commands for pip, as they are written: `vf venv`, then `pip install torch`
  from PyTorch's index for CUDA 13 in that environment, pip itself started through the layer.
  On 2026-11-21 that brought PyTorch 2.14.1, a later one than the 2.14.0 of everything above.
  The environment's Python, started on a V100 with no prefix: the products, the transform,
  the factorisations and the sparse product give the CPU's results, with all the CUDA 12.9.2
  libraries and with cuBLAS alone; and the 925 operations (924 and the one) and the 32 models
  on that wheel too. With the hook off (`VF_VENV=0`) the same Python is told capability 7.0,
  and its first operation on the GPU fails with `cudaErrorNoKernelImageForDevice`: what the
  wheel does on a V100 without the layer.

## 7. The two translators, the fuzzer, and readings by others

### Two translators

The reference translator (`reference/translate.py`) and the translator of the layer
(`crates/vf-sass`) must give the same bytes, the same map and the same refusals for the same
reasons on every input, and the same file of lives beside the map (`MAP.lives`, since
2026-11-22). `tests/sass/compare.sh` checks it. On 2026-11-23, with the translators of 0.1.3:
the 14 test cubins (five of them for sm_80) and PyTorch's 475, 489 of 489 the same (23,727
functions, 30 refused); on 2026-11-22, before the change of the sequence of a matrix load, the
same at `VF_BORROW=8`. Before (2026-11-20): the same again when
they are translated as the second translation at launch does (`VF_BORROW=5`: 42 refused); of
the 157 distinct cubins that the PyTorch battery and the models made the layer translate, 122
are among those 475, and the other 35 are the same at both settings; the held-out corpus, 3,108
of 3,108 (70,978 functions, 10,986 refused).

The validator checks what the reference translator writes; the comparison is what carries its
verdict over to the translator that runs in the layer.

### The fuzzer

`tests/sass/fuzz.py` damages cubins (instructions replaced by others from real code, a field
changed, bits flipped, the schedule changed, instructions exchanged, doubled or removed,
attributes, register counts, relocations and headers changed) and gives each one to both
translators, which must agree on everything; where the reference translator stops on an
exception, the other must still end properly. With the build of this tree: 200,000 damaged
cubins in four runs, made from 60 small ones (the test kernels, small cubins of PyTorch, twelve
of cuFFT that have relocations on their code, a few of cuDNN and cuBLAS), with 3,942
instructions of 247 classes to splice in: 194,197 the same from both translators, 5,803 where
the reference translator stops and the other refuses the file, none different, none that
ended badly.

### Readings by others

The translator, the validator and the tests have one author. Seven readings of the code by
others (2026-11-17 to 22) found faults that every check above had let through:

- An instruction that becomes several must keep its wait. 949 `ULDC` of the translation of
  PyTorch had lost theirs, and with the schedule arranged for it two test kernels gave wrong
  sums on a V100; the battery had never shown it. The control `waitless` and the test
  `translated-waits-moved` come from there.
- The validator did not look at an instruction that the map left out, and two instructions
  exchanged passed. The rules of the map and the controls `unlisted` and `swapped` come from
  there.
- The analysis that lets a pair of products do without part of a matrix load did not count the
  predicates that an `R2P` writes under a mask, and the checker of plans had the same omission.
- A predicate saved after a branch was read without a wait for the instruction that wrote it.
- In the rewriting of PTX: a 256-bit load done in two halves overwrote its own address; atomic
  additions on bf16 lost their memory order and scope; the type of a register declared again in
  a block was not restored at its end; what was known of a register outlived its scope.
- In the forwarding libraries: more words of the stack were copied than the arguments take.
- What the translation stores after an instruction went by the instruction's guard, and an
  instruction may write its own guard. No compiled code at hand does; with test kernels
  changed by hand to do it, two gave wrong results on a V100 (section 6). The test
  `translated-guard-written` and the controls `guarded_store_l3` and `no_old_load_l3` come
  from there.
- The second translation at launch was reached by one of the two exported names of the call
  that asks for a variable (so that a module whose variable the program had asked for could
  be launched from a second image, with its own copy of the variable), by two of the four
  launch entries, and by one of the three ways to take a kernel of a library.
  `tests/proxy/fit_api.sh` and seven checks of `fitrun` come from there.
- In the rewriting of PTX: a register is followed by its name, and the assembler takes
  `%d00` for `%d0` of a family `%d<4>`; a statement that read the accumulator of a chain of
  products under that spelling was given the register as it was before the chain. The first
  correction then renamed too much: a variable declared in a block under such a name was
  taken for the register, and a statement that took its address read the register (found by
  the next reading; such a function is now refused).
- The validator itself, asked to be read for what it would let through: it passed a store to
  the frame placed before the instruction that computes what is stored, and a load from the
  frame into the home of a uniform register that the instruction still read (two
  translations made wrong by hand, which a model of the instructions runs to 123 for 124 and
  to 22 for 18). Each rule looked at one instruction of a sequence, none at what the moves
  carry from one to the next. The rule "What the frame's moves carry" (section 2) and nine
  controls come from there. Neither wrong order is one that the translator makes.
- In the proxy: an enumeration of a library's kernels read as many handles as the program's
  array could hold, not as many as the driver had written; with an array larger than the
  library's number of kernels and anything in the places after them, NVIDIA's driver
  faulted on what it was given (seen on a V100 with the test that now passes). What the
  program sets on a library's kernel was not given to the second translation of the same
  kernel taken through the library's module. And a launch whose second translation was
  being made when the program asked for data of the module still started from it.
- In the rewriting of PTX again: the statements of a body are read as text, and a comment
  inside a statement was read with it. A family of registers declared with a comment beside
  it was not seen, its member written with zeros kept that spelling, and the translation of a
  chain of products stored 0 where the source stores 32, with no refusal. The comments of a
  body are now written over before anything reads it.
- The validator again: the rule on what the frame's moves carry followed the values and not
  the addresses. A store moved before the load of the base it is addressed from passed, and
  wrote the word of another uniform register (a model runs it to 123 for 124). The clause on
  addresses, the control `base_after_use_l3` and a fifth pass of the controls come from
  there. The translator does not make that order.
- In the proxy: the driver was called before the lock under which a setting is remembered and
  given to the second kernel. Of two settings of one kernel at once, the one that the driver
  took first could be recorded last: the kernel and its second translation were left with
  different values after both calls had returned.
- In the launcher: `--map FILE` copied the lines of the file into a variable, where a
  relative path is no longer relative to the file, and dropped a last line without an end of
  line. And the launcher's own test suite ended with status 1 when all its checks had passed.
- A review of the whole tree (2026-11-22), with runs on V100s. In the proxy: told 8.0 or more
  with `VF_CAPABILITY`, PyTorch was given the translation of its sm_75 code, in which the
  kernels that it chooses for 8.0 are stubs that write nothing: 24 attention cases of the
  battery and 28 of the 32 models were wrong with no error, and told 9.0, kernels stopped the
  GPU. No test ran PyTorch told 8.0. A caller is now told no later generation than the code it
  is given (`docs/architecture.md`, section 4), and the battery and the models give the CPU's
  results told 8.0 (section 6). The test programs of PyTorch lost a verdict that a kernel's
  message came before (the battery counted 13 wrong cases where 24 were), could end with
  status 0 on a wrong model, and waited for ever on a worker that filled its error pipe. The
  validator did not follow the moves of a uniform predicate kept in the frame: a load of the
  predicate into a register that the instruction reads passed with 0 problems (two
  translations of a PyTorch cubin made wrong by hand); the rule of section 2 now follows them,
  and the four controls `up_*` come from there. The translator does not make such a
  translation. The review also found that the translation of sm_80 machine code has no
  independent check (section 8), and faults of the proxy, the launcher, the installer and the
  parsers that are not faults of a translation (`CHANGELOG.md`).

Each was reproduced and corrected, except the gap of the sm_80 translation, which section 8
states. They are listed because they measure the checks: the
validator, the comparison of the two translators and the programs had all passed.

## 8. What is not covered

**The table of distances.** Where an instruction of a translation names no scoreboard, what
reads its result comes after it by a distance from `reference/distances-sm70.tsv`: the least
that native sm_70 code compiled by NVIDIA keeps between such a writer and such a reader
(`docs/translator.md`, "Timing rules"). The translator and the validator take the same table,
and nothing checks that a distance is long enough. Too short, the reader gets what the register
held before, and nothing faults.

**The descriptions of the instruction sets.** The translator and the validator decode with the
same descriptions. A wrong entry would be wrong for both. Section 5 covers what nvdisasm prints
(the class, the mnemonic, the registers and predicates, what is read and written), not the
fields that it does not print as such.

**The hardware's behaviour.** The timing rules that were measured (the time after the last step
of a product, what `R2P` reads and writes, the predicates written late) were measured on one
V100 with one driver. Another V100 is assumed to behave the same.

**In the validator itself**: the three points at the end of section 2, the rules that section
3 lists as having no control of their own, and the controls that it lists as changing nothing
in the code at hand.

**The translation of sm_80 machine code.** Since 0.1.2 the layer also translates a subset of
stripped sm_80 executables (`crates/vf-sass/src/translate/ampere.rs`, `reference/ampere.py`;
`LIMITS.md`, "Kernels refused"). This path is **not independently validated**: the validator
reads its sources with the descriptions of sm_75, and on the reference translator's own
translation of the five sm_80 test cubins it reports problems that are not faults (it can
neither pass nor usefully fail such a translation); `validate-corpus.sh` and `controls.sh`
take sm_75 cubins only, no control exists for an Ampere lowering, and no corpus of sm_80 code
is validated. What checks it: its own rules, written twice by one author (`compare.sh`: the
two translators agree on the five sm_80 test cubins), unit tests, five cases of
`tests/gpu/run.sh` (`cubin-sm80-*`: each kernel's result on a V100), and the engines that were
run by hand with it (SGLang and FlashInfer, `docs/guides/sglang.md`). A review of 2026-11-22
added two checks of its own, which found nothing wrong: the 221 forms that the path accepts,
read by NVIDIA's disassembler as sm_80 and as sm_75 on 453 instruction words of real code
and 4,420 drawn at random, with no mismatch; and 101 kernels written for it, built for sm_80
without PTX and for sm_70, of which 28 are translated and give their sm_70 builds' results
on a V100 at seven block sizes (73 are refused). Of 31,207 functions of 1,193 sm_80 cubins of
PyTorch, cuBLAS, cuSOLVER and cuFFT, it translated 900 (2.9 %); some of its lowerings
(`ATOM_ADD32`, `SHARED_INC`, `COPY128`, `COMMIT`, `PACK`) are reached only from code that Triton
emits, and neither of those checks ran them.

On 2026-11-25 the path was widened, and the widening showed what this gap costs. Whole classes
of sm_80 were taken (`docs/translator.md`, section 1): 680 classes whose every bit reads the
same in the two descriptions, where the path took 221 forms. `reference/ampere_check.py` checks
that claim against NVIDIA's disassembler on real code: for every distinct word of sm_80 cubins
that passes as it is (scheduler controls apart), nvdisasm must print the same text for it as
sm_80 and as sm_75, and each description must find one class, the same one, with the same
operands. With every whole class taken: 791,674 distinct words of vLLM 0.34.0's sm_80 code and
3,178,811 of PyTorch 2.14.0's, of which 620,563 and 2,688,238 passed as they are, and none
failed (the check was seen to fail: with the class beside `F2FP` taken whole, as a first
version of the table had it, a word of `F2FP` to bf16 passes as that class, nvdisasm reads it
as sm_80 and not as sm_75, and the check says so; classes kept for some values only no longer
vouch for their neighbours). The two translators agreed on all 544 cubins. And the validator
of that day, given a function as the sm_80 path normalizes it (sm_75 words) and its
translation, found 0 problems. **All of that passed, and three kernels were wrong on the
GPU**: PyTorch's double-precision erfc, erfinv and exp2, which it compiles for sm_80 while it
runs when it is told 8.0, gave other results than the CPU in `tests/gpu/torch.sh` (erfc and
erfinv in every element, exp2 in 67 of 2,145), where the three had been refused before.

**The cause, shown from the code; confirmed on the GPU: not yet.** A review of the same day
found it in the schedule, not in the instructions. An instruction of double precision (DADD,
DMUL, DFMA, DSETP: Volta's `fma64lite` pipe) announces its result on a scoreboard, and its
reader waits for it; the translator, and the validator, took that wait to be enough. NVIDIA's
compiler does not: in its own code for sm_70 a reader of such a result never comes closer than
a fixed distance, scoreboard or not: 8 cycles to an instruction of the same pipe (54,085
pairs of 108 cubins of cuSPARSE 12.9 and of a corpus of double-precision kernels built for
sm_70, none below, two thirds at 8), 9 to any other, and a predicate of DSETP 9 as data, 8 to
an instruction of double precision and 15 as a guard or to a branch; its tables for sm_70 and
sm_75 say 9 and 15, and code for sm_75 keeps the same distances. Code for sm_80 keeps 6, 6 to
7 and 14, and the sm_80 path, which schedules every instruction anew, lengthened only the
distances from writers without a scoreboard. The three wrong translations have 165, 192 and 33
reads closer than Volta's own code has any; every kernel that ran right (the float32 ones of
the same functions, the test kernels, vLLM's norm) has none. Of 2,060,115 waited pairs of the
sm_80 corpora (cuSPARSE, PyTorch, vLLM), 314,815 are under Volta's floor, all but 11 from
these four instructions (8 from an atomic operation, read 11 cycles later where 32 native
pairs have 13 or more, and 3 from I2F, whose "floor" native sm_75 code itself goes under 63
times: neither is taken for a rule, and no form that the path takes has such a pair). How each
kernel failed fits: erfc reads the seed of its reciprocal 6 cycles after the DADD that makes
its operand, in every element; erfinv branches on a DSETP 14 cycles before, into the path of
values outside the domain; exp2 has few such reads.

What was done:

- **The schedule** (`reference/machine.py`, `Machine.waited`; `crates/vf-sass/src/distance.rs`,
  `schedule.rs`): a reader of what DADD, DMUL, DFMA or DSETP writes comes no earlier than
  Volta's own code has one (8, 9, 15), whatever scoreboard is named and waited for, wherever
  one of the two instructions is the translation's (in an sm_80 function, every one). With it,
  the three kernels translated as on that day have no read under the floor (165, 192 and 33
  without).
- **Across a branch** (2026-11-25, found by the review): the schedule and the validator's rule
  both saw straight-line code only, and a reader at a branch's target was neither kept away nor
  looked at. With every whole class taken (a variant of each translator made for the
  measurement), over 253 sm_80 cubins (the linalg library's 65, vLLM's 69, every fourth of
  PyTorch's), 24 readers were under the floor on a path through a branch (23 `DFMA` to `DMUL`
  at 6 or 7 cycles in linalg's reciprocals, one `DMUL` to a store at 7). Now, in code for
  sm_80, a result of double precision is as old at the end of straight-line code as any reader
  needs (9 cycles, a predicate 15), and the validator follows the next instruction and the
  targets of the code's branches (not a branch through a register, a BSYNC or a return): 0 of
  189,426 readers in the same 253 translations, by the validator and by the review's own
  tool; the two translators give the same bytes for all 253. Native sm_75 code has no reader
  under the floor on any such path (PyTorch's 475 cubins), so nothing is added to its
  translation: the 1,107 cubins of the corpora are translated to the same bytes as before.
  Control `doubleacross` (the reader where a branch goes is brought to 6 cycles on the path
  through the branch): objected to in PyTorch's 0434, 0467 and 0469, and in none of them by
  the rule as it was.
- **The validator** (`double_read_early`): the same distances, written there on their own, for
  every translation. Given the three functions as normalized: 420, 448 and 66 problems in the
  translations that were wrong, none in those made with the rule; none in the translations
  of 79 of PyTorch's 475 sm_75 cubins (every sixth; 418,387 such readers looked at; the whole
  corpus was not run again). Control `doublesoon` (`reference/controls.py`: the reader after
  an instruction of double precision 6 cycles after it; objected to 237 times in a cubin of
  double-precision kernels built for sm_75, and in PyTorch's 0097).
- **No form of double precision passes** all the same: a word passes by its class only in a
  form that ran right on a V100 (`reference/ampere-evidence.txt`: 74 forms of 43 classes), and
  the generator refuses a form of these four instructions, until the measurement that decides
  is made: the three kernels as translated then, and the same with only their stall counts
  raised to the floor, on a V100 against the CPU (prepared by the review, not run: it runs code
  known to be wrong, and waits for the owner's word). Until then the rule above is a rule that
  no translated code exercises on a GPU.

With the evidence list the three kernels are refused again: the two translators agree on the
same corpora (519 functions of PyTorch's and vLLM's sm_80 code translated, where 452 were),
`ampere_check.py` passes on them, and `torch.sh` told 8.0 gives 867 right, 57 refused, none
different (864 and 60 before: the float32 erfc and exp2 and the float16 erfc are now given, and
are the CPU's within the battery's tolerances). The translation of sm_80 machine code is
**still not independently validated**: the validator does not read sm_80 originals (the
normalized function was given to it by a script of the day), and it has no rule for the forms
that the path rewrites. The day's lesson is that agreement of descriptions, of translators and
of the validator did not replace results on the hardware: a form enters the list with a
kernel that ran right.

**Triton's machine code.** Triton makes its kernels while the program runs, for its shapes and
settings; told 7.5 (the default), it gives them as sm_75 machine code, which the layer
translates. The corpus of section 6 ("Triton") is 3,516 configurations of 20 kernels written for
the test and 241 cases of 123 kernels of vLLM and SGLang (of their 2,046 `@triton.jit`
functions); the validator has checked the 4,308 cubins of the first and the 334 of this site's runs
(section 4). Other programs, shapes and settings make other kernels, which neither saw. Until 2026-11-23 the
translation of variants of one `tl.dot` kernel stored wrong values on a V100, and the validator
found nothing.

**Kernels that nothing runs.** The battery and the programs launch the kernels that their cases
ask for; the others of the 23,664 functions translated in PyTorch's library are checked by the
validator only. None of the kernels of the held-out corpus was run.

**Programs and models.** vLLM and SGLang were run with Qwen2.5-0.5B-Instruct and three prompts,
llama.cpp with Qwen2.5-0.5B-Instruct and with Qwen3.8-27B, ZML with Qwen3.5-9B. Another model,
another batch or another option may launch a kernel that no test ran, and with FlashInfer one of
the five kernels that are refused (`LIMITS.md`). ZML's check compares no token with a reference.

**What was not tried** is listed in `LIMITS.md`: translated kernels on several GPUs at once
in one process, MPS, later drivers, other operating systems. Machine code for sm_86 and later
that comes without PTX, without code for sm_75 and without code for sm_80 is not translated.

**One author.** The validator was written by the author of the translator, and so were the
tests and the choice of what to test. The readings of section 7 are the only ones by others, and
each of them found something.

**What "0 problems" is.** Over a corpus, it says that every instruction of every translated
function reads, for NVIDIA's disassembler, as what the original asks for by the validator's
rules; that the registers and predicates taken hold nothing of the program by the validator's
own lives; that the waits are there and not too close. It does not say that the code gives the
right result on a V100: the validator runs nothing. The tests on the hardware say that, for the
kernels they run.
