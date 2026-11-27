# Test suites

What is here: the tests on a GPU (`tests/gpu/`), from single kernels to whole engines; the
tests of the machine-code translator, which need no GPU (`tests/sass/`); a test of the
forwarding libraries' stack handling (`tests/forward/`); the launcher tests
(`tests/launcher/`); the tests of `VF_TRITON_TC` on the host (`tests/python/`); and the kernels,
programs and reference vectors that they use (`tests/kernels/`). `docs/validation.md` says what each of them can and cannot show.

## Settings

Every test script reads `tests/site.env` when it exists (see `tests/site.env.example`) and uses
`VF_TEST_ROOT` as the root of its working files. Each setting is an environment variable with a
default; a variable already set in the environment wins.

| Variable | Default | Used by |
|---|---|---|
| `VF_TEST_ROOT` | `$XDG_CACHE_HOME/volta-forever/tests` | all scripts |
| `VF_OUT` | `$VF_TEST_ROOT` | GPU scripts: build outputs and logs |
| `VF_PROXY` | `dist/lib` of this tree | GPU scripts: the `libcuda.so.1` to test |
| `VF_LAUNCHER` | (not set) | torch, torch-models, vLLM, SGLang: `1` takes the layer's environment from the launcher (`vf env` of this tree) instead of setting it by hand |
| `VF_JIT_CACHE` | `$VF_TEST_ROOT/jit-cache` | llama.cpp, ZML: compiled kernels |
| `VF_TORCH` | `$VF_TEST_ROOT/torch-cu130/site` | torch, vLLM, SGLang: PyTorch 2.14.0+cu130 |
| `VF_TORCH_NATIVE` | `$VF_TEST_ROOT/torch-cu126/site` | torch-bench, torch-models: PyTorch with sm_70 code |
| `VF_NATIVE_LLAMA` | `$VF_TEST_ROOT/llama-b12418-sm70/build/bin` | llamacpp-bench, llamacpp-ops: native sm_70 build |
| `VF_AUDIT_MAP_FILE` | `$VF_OUT/torch-audit.map`, written when missing | torch, vLLM, SGLang: the map that puts the forwarding libraries in front of the cuBLAS that the PyTorch wheel carries (`INSTALL.md`, way B) |
| `VF_VLLM_SITE` | `$VF_TEST_ROOT/vllm/site` | vLLM: vLLM 0.34.0 and its dependencies |
| `VF_VLLM_MODEL` | `$VF_TEST_ROOT/models/Qwen2.5-0.5B-Instruct` | vLLM: the test model |
| `VF_VLLM_VENV` | (none) | vLLM: a virtual environment, instead of the directories above |
| `VF_SGLANG_SITE` | `$VF_TEST_ROOT/sglang/site` | SGLang: SGLang 0.5.23 and its dependencies |
| `VF_SGLANG_VENV` | (none) | SGLang: a virtual environment, instead of the directories above |
| `VF_MODEL` | `$VF_TEST_ROOT/models/Qwen2.5-0.5B-Instruct` | SGLang: the test model |
| `VF_ZML` | `$VF_TEST_ROOT/zml/build` | ZML: the build of `tests/gpu/zml/build-zml.sh` |
| `VF_ZML_MODEL` | `$VF_TEST_ROOT/models/Qwen3.5-9B-AWQ-fp16-text` | ZML: the model (9 GB) |
| `VF_VLLM_EXTRA` | `$VF_TEST_ROOT/vllm-extra/site` | vLLM: packages that the sites above lack (loguru and others) |
| `VF_VLLM_DEPS` | `$VF_TEST_ROOT/vllm-deps/site` | vLLM: what the sites above lack of what vLLM declares |
| `VF_CUTLASS_PREFIX` | `$VF_TEST_ROOT/cutlass-prefix` | SGLang: the CUTLASS DSL |
| `VF_PYTHON_DEV` | `$VF_TEST_ROOT/python-dev` | vLLM, SGLang: Python 3.12 headers |
| `VF_SERVER_WAIT` | `1800` | vLLM, SGLang: seconds to wait for the HTTP server to answer |
| `VF_ENGINE_WAIT` | `1800` | SGLang: seconds that an engine stage may take |
| `VF_CASE_WAIT` | `600` | torch, torch-models: seconds that a case may take before it is stopped and counted as a CRASH |
| `TORCH` | `$VF_TEST_ROOT/torch-cubins` | SASS: PyTorch cubins for the corpus |

A setting of one script alone is described in the header of that script (for example
`VF_ZML_CUDNN` of `zml.sh`, `VF_BENCH` of `llamacpp-bench.sh`, `VF_TOOL` and `JOBS` of the
scripts of `tests/sass/`).

`run.sh`, `forward.sh`, `forward-bf16.sh` and `triton.sh`, whose cases set the layer's settings that
they need, unset those left in the environment (the settings of `INSTALL.md`, "Environment
variables", that change what the layer does, not those that say where something is, such as
`VF_PROXY_CACHE`) and print one line that names them, except `VF_CAPABILITY` for `run.sh` (a release
is checked with it at 8.0; its cases of what a program is told by default run without it); the other
scripts take the layer's settings from the environment (as `torch.sh` is run at `VF_CAPABILITY=8.0`
or with `VF_SAFE=1`).

Where the GPU scripts write:

- Under `$VF_OUT`: their builds and logs (`bin/`, `logs/`; `translate/` of `run.sh`, `bf16-test/`
  of `forward-bf16.sh`, `tbo/` of `llamacpp-ops.sh`, `zml/` of `zml.sh`), the maps of the forwarding
  libraries (`torch-audit.map`, `forward-audit.map`), and for the scripts that run whole programs a
  directory of their own, which is those programs' `HOME` and `TMPDIR` and holds their caches
  (Triton's, Hugging Face's, FlashInfer's, SGLang's): `vllm-cache/` of `vllm.sh`, `sglang-cache/` of
  `sglang.sh`, `triton-test/` of `triton.sh`, `vllm-topk-cache/` of `vllm-topk-topp-run.sh`, and
  `launcher-caches/` of `launcher.sh` (the `HOME` of its programs and their caches: the layer's, the
  driver's and PyTorch's). `profile-torch.sh` writes under its `$VF_OUT` (default
  `$VF_TEST_ROOT/profile`) only: `logs/`, `tmp/` (its `TMPDIR`) and `cache/` (the layer's, the
  driver's and PyTorch's caches).
- The kernels that the driver compiles from PTX, and those that PyTorch compiles: in
  `$VF_OUT/nv-cache` and `$VF_OUT/torch-kernels` when `$VF_OUT` exists at the start and
  `CUDA_CACHE_PATH` and `PYTORCH_KERNEL_CACHE_PATH` are not set (`tests/gpu/cache.sh`); llama.cpp's
  and ZML's in `$VF_JIT_CACHE`, `vllm.sh`'s and `sglang.sh`'s in their directory.
- Under `$TMPDIR` (`/tmp` when it is not set), each in a directory of its own made by the run and
  removed at its end: the layer's cache (`tests/gpu/cache.sh`, unless `VF_PROXY_CACHE` is given, `0`
  to turn it off), which must be a local directory of whoever runs (the layer refuses a cache whose
  files belong to another user, as they can on a shared file system); the caches of the cases of
  `run.sh` that test the layer's cache; and the sockets of vLLM's processes in `vllm.sh` (unless
  `VLLM_RPC_BASE_PATH` is given). `vllm.sh`, `sglang.sh`, `triton.sh` and `vllm-topk-topp-run.sh` set
  `TMPDIR` to their directory under `$VF_OUT` before they make the layer's cache, which is then there
  (one for each pass of `vllm-topk-topp-run.sh`); the sockets of `vllm.sh` stay under the `TMPDIR`
  that it was given.
- What the programs of the other scripts write by themselves under `HOME` (PyTorch's and Triton's
  caches, for one) is where those programs put it.

## What PASS means

- `run.sh`: the test kernels, loaded as PTX and as machine code through the proxy (retargeting,
  translation, refusal, the cache, the second translation at launch, the CUDA runtime), give
  what the native sm_70 build of the same source, the host models or the reference vectors of
  an RTX 5090 say. 234 cases (229 at 0.1.6, 200 at 0.1.5, 165 at 0.1.4), including nonconsecutive MMA fragments, a
  partial warp reduction, a zero-filled asynchronous copy, sm_80 machine code translated, refused lookups
  (error 218 and a NULL handle, through a module, a library and the driver's linker), files of PTX,
  relocatable sm_75 machine code and a fat binary given to the driver's linker with `cuLinkAddFile`
  and `cuLinkAddFile_v2`, adapted, linked and run (`link-file-*`, `DRVRUN_LINK` of `drvrun.c`), a program
  with a refused kernel under the CUDA 12 and 13 runtimes in both loading modes
  (`CUDA_MODULE_LOADING`), a refused kernel launched with a handle taken from NVIDIA's driver
  itself (218 from every launch function and graph node, and no line more with "Xid" in the
  kernel's log, where it can be read: `run.sh` reads it with `dmesg`, else, where Ubuntu's
  `kernel.dmesg_restrict=1` keeps that from users, with `journalctl -k` (readable by the groups
  `adm` and `systemd-journal`), else with `sudo -n dmesg` (sudo without a password only: it never
  asks for one), and says in each case's log which one it read; where none can be read, the last
  line of the output says "Xid lines could not be checked (the kernel's log could not be read:
  ...)"; `tests/scripts/kernel-log.sh` tests the three with stand-ins), a kernel whose code is a trap and nothing else,
  from PTX and from sm_75 machine code, given at the lookup and its launch refused with 218, the
  context going on (`nocode-*`, with the same check of the kernel's log; the control against
  NVIDIA's driver, `test_nocode IMAGE native` of `tests/kernels/test_nocode.c`, stops the context
  with an Xid line and is not run by the suite), the words of fp16 products of machine code
  translated in pairs and each by itself, and with `VF_SAFE=1` (`pair-bits`), CUB's sort, sum and
  scan of a program whose code is PTX for sm_75 only, of its sm_75 machine code and of its sm_70
  build, and told 8.0 (`cub-*`: the PTX version of its kernels is that of its PTX, 7.5, where it was
  7.0 and CUB answered "invalid device function" until 2026-11-24), lookup tables
  asked for by name, answered as NVIDIA's driver answers them, except where the layer refuses one
  (`tables.py`), and a fat binary whose sm_80 machine code has a kernel that its sm_75 code has not
  (`sections-*`, `tests/kernels/test_sections.c`): told 8.0, that kernel is given from the sm_80
  code, translated, in a module of the layer's own, through a module, a file and a library, with
  one line, and stores what the sm_80 code stores; told 7.5 its lookup fails with 500 as the
  driver answers and nothing is said; with `VF_TOLD_CODE=0` it fails told 8.0 too, with the line
  that says which code has it; with `VF_TRANSLATE=first` the kernel runs from the whole sm_80
  code. And what such kernels compute (`told-*`, `tests/kernels/k_told.cu`, `test_told.c`):
  sums of pairs of bf16 and pairs of bf16 nearest to f32 (two instructions of sm_80 that the
  layer rewrites), 16,384 each against the host's exact rounding, and the quotients of a
  division whose slow path is a function inside the kernel, also through the library's module,
  after the image is loaded again and in safe mode; a kernel that reads a variable of the image,
  one that adds a `__constant__` variable, one that stores the address of one (a word of its
  own bank 0) and one that is not translated are not given, with the reason for each. What needs
  no GPU of this is in `tests/proxy/told.py`, against the stand-in driver of `launch_guard.py`,
  which has the kernels of the machine code it is given: a function is given only in the context
  of the module asked (the program's, or the one that the driver gives for a library), from a
  module loaded there; an unload from another thread while the layer loads leaves nothing
  loaded; what was loaded goes with the module or library, and stays when the driver refuses
  the unload; nothing is loaded with `VF_TOLD_CODE=0`, nor for an image of relocatable code
  (`k_told_rdc.cu`), loaded or linked (on the driver itself: `told-8.0-rdc`, `test_told_rdc.c`,
  the two units linked by `cuLinkComplete`, the lookup of the kernel that only the sm_80 code
  has failing with 500 and nothing said of that code). Three changes of `told.rs` each fail it (the context
  not checked, the library's module not checked, the unload during the load not looked for); and kernels that hold, or call through a table, an instruction of the
  multimem family (`reported-*`, `tests/kernels/test_reported.c`): given with the instruction
  replaced by a reported trap, said in one line for their image, and right on every path that does
  not reach it (one with its multimem instructions under a guard that is false among them), through
  a module, a library and with `VF_LAZY=0`, and in a module with comments before its header, while the same call that may reach
  `barrier.cluster.arrive` is refused, and with `VF_REPORTED_TRAPS=0` all of them are, with the same
  check of the kernel's log; and what the driver answers about multicast, without the layer and
  through it (`multicast-*`: the attribute is 0, `cuMulticastCreate` an error), and what the layer
  says of it to a program without a multimem instruction (`multicast-quiet`: nothing at the default
  level; `multicast-said`: the answer at `VF_LOG_LEVEL=2`).
- `nccl.sh [cuda13|cuda12] [--bench]`: NCCL's collectives through PyTorch on the first two V100s
  visible, one process for each (`nccl-collectives.py`): all-reduce (sum in int32, int64,
  float64, float32, float16 and bfloat16, max and min in those but int64), broadcast, all-gather, reduce-scatter, reduce,
  send/receive and all-to-all on 1 to 4,194,304 elements, every value compared with the CPU's
  exactly (floats that every order of addition sums exactly), first as NCCL chooses, then with each
  algorithm and protocol pinned (Ring and Tree; Simple, LL and LL128), NCCL's lines saying what ran;
  with the wheel's CUDA 13 NCCL (`cuda13`, given with reported traps: a refused kernel of its
  collectives or an assertion of the layer fails it) or the CUDA 12 build put in its place
  (`cuda12`); `--bench` times an all-reduce of 64 MiB and one of 8 bytes.
- `multi-gpu.sh`: on systems with at least two GPUs, two threads in one process load
  `k_basic` concurrently via both `cuModuleLoadData` and `cuLibraryLoadData` on separate primary
  contexts, capture and launch per-GPU CUDA graphs simultaneously, and run a cross-device
  `cuMemcpyPeerAsync` pipeline against native sm_70, retargeted sm_75 PTX and translated sm_75
  cubins under `LAZY` and `EAGER` loading, comparing every case with the host's exact values
  (skips when fewer than two GPUs are visible).
- `forward.sh`: for each of ten libraries, a program built against CUDA 13.5 calls a few
  functions through the forwarding library and gets the host's result, five of them again under
  `CUDA_MODULE_LOADING=EAGER` (cuBLAS is exercised by `run.sh`, under both loading modes).
- `forward-bf16.sh`: the opt-in bf16 cuBLAS variants on a V100, with a test build of their
  runtime: the skinny path's results bit for bit against cuBLAS 12.9 where it serves, and the
  inputs that tell a fused multiply-add chain, a multiply-and-add chain and the epilogue's orders
  apart (`bf16-arith.c`); the tensor-core path within 32 units of 2^-24 sum |a b| of fp64,
  deterministic; every forwarded call with 12.9's bits; what a program's graphs, capture modes,
  threads and contexts see (`bf16-graphs.c`, `bf16-threads.c`, `bf16-life.c`, `bf16-reserve.c`;
  a graph captured through the variant behaves step for step as with 12.9 alone and replays 12.9's
  bits, with tensor-core-sized calls, forwarded, and with skinny ones, `bf16-graphs-skinny`, served
  by kernel nodes of the runtime's module, which replay after every handle is destroyed);
  cuBLASLt calls that keep 12.9's behaviour: an explicit algorithm other than 12.9's heuristic one,
  and a workspace size without a workspace (`bf16-lt.c`); the tensor-core path's fallback in the TN
  layout of linear layers, bit for bit against a host model, also with failing rows and columns spread
  over the chunks of a call, there against cuBLAS 12.9 too (`bf16-fallback.c`); verify mode, every
  skinny call of each kind (entry points, layouts, output types, C read, scalars on the device, an
  unaligned C) and a tensor-core call's fallback elements against 12.9, eager and replayed from a graph,
  silent when nothing differs, failing with a fat binary that changes one element of the skinny kernel's
  (`bf16-verify.c`), and bf16-graphs' cases with the skinny path's calls captured under it; verify mode
  with two threads on one stream, on the legacy stream, on the per-thread default stream and on a stream
  each, none of their calls reported as differing, and failing without the stream's lock (the control
  `verify-unlocked`; `bf16-verify-threads.c`); the runtime told
  the real capability under the layer's settings, under both module loading modes
  (`CUDA_MODULE_LOADING=EAGER` and `LAZY`);
  the checked-in fat binary rebuilt when the CUDA 12.9.2 compiler is there (`bf16-kernels`); the
  tensor-core path's special values against 12.9 (`bf16-s1.c`) and its acceptance test
  (`bf16-accept.c`: 12.9's classes, determinism, an error bound and a bias bound against the exact
  product for every call it serves at k >= 768, in fp32 and bf16 output, nothing served below and every
  TN call from there on served, failing when the gate is moved to 512 or 256 (on other families than the two judged from 1024),
  with `VF_ACCEPT_STRICT=1` (its four documented exceptions removed: lognpos and advpos then fail at the
  gate) and with the path off; with `VF_SAFE=1`, every result 12.9's); the path's layout
  rule, tall rule and wide rule (`bf16-rules.c`: a fine-tuning step's input and weight gradients, NN and
  NT, an output layer up to the tall rule's n, a wide down projection past the wide rule's n and a long
  product of many rows at the tall rule's long-k m with k past its long-k bound forwarded with 12.9's
  bits, the step's TN products, the output layer one column more and at 128 columns, the wide down
  projection at the wide rule's n and the long product at the long-k bound or one step of m under it
  served; with each rule removed by a control, it fails, and a bound that the runtime does not report
  is a failed check, not a crash: `VF_RULES_NO_BOUND=5`, a control); the runtime's counts in the
  layer's `VF_SUMMARY` file (`bf16-summary.c`, whose child process opens the layer and the variant
  with `dlopen` in the order of its case: killed with SIGKILL after its calls, its file holds the lines
  that the runtime's fill gives, with the counts of the calls made; at exit with the layer loaded first,
  the lines given live to the layer's last write, and with the runtime loaded first, its last report
  kept, both the lines of the exit report; children made by `fork()` that do nothing with the layer and
  write nothing; the variant and every reference to the runtime given back by `dlclose` while the layer's
  thread writes, the runtime staying, pinned; and its controls: no registration, no `VF_SUMMARY`, the
  runtime not pinned, unloaded with its last report kept, neither pinned nor removed, the child ending
  by SIGSEGV, and a child that does not end, before it loads anything or instead of exiting
  (`VF_BF16_SUMMARY_HANG=start`, `exit`), whose process group is killed at the case's limit with no
  process of it left; no wait of it is unbounded, the child running in a process group of its own and
  the case ending within `VF_BF16_SUMMARY_LIMIT`, 240 s by default, its checks of the file made on lines
  from their start); every case with a time limit, 900 s (`bf16-summary`'s 300 s), past which
  timeout(1) ends it and it fails; with `VF_BF16_SANITIZE` set, under compute-sanitizer, without a
  time limit. `bf16-skinny.c` is the measurement behind the
  skinny path's rule and bounds (12.9's kernel per configuration; path, bits over the m rows of each
  column, writes outside them, error and time per shape), `bf16-cost.c` the one behind the
  documented host times, fallback cost and same-sign error, and `bf16-layout-cost.c` the one behind the
  layout rule, the tall rule and the wide rule (the time of single products per layout, shape and share
  of failing columns, through the variants and through 12.9); all run by hand.
- `tests/forward/bf16.sh` (no GPU, in the CI): the variants generated from stand-in libraries, and
  their runtime against stand-ins for the driver and cuBLAS that record every call (H1-H7, H9 of
  the design review), the paths' rule as a table of shapes across its boundaries in n, m k and k, with
  the products of a fine-tuning step, the tall rule's edges and the classes it serves (rows that hold
  whatever the gate is), the runtime's lines in the layer's `VF_SUMMARY` file (mode `summary`: under a
  stand-in that has the layer's `vf_summary_add`, `stub_cuda.c` built with `-DSTUB_LAYER`, the runtime
  registers its fill once, at its first handle, which gives the exit report's lines with the counts as
  they are, and at the runtime's end what the exit report printed; under the plain stand-in nothing is
  registered, and the exit report prints the lines in the text of before, which a test build's control,
  `report-format`, changes; mode `summary-refused`: the stand-in refusing the registration with a code,
  `STUB_LAYER_ADD`, the runtime is not registered, serves its calls, and says so once at
  `VF_LOG_LEVEL=2` with the code, not for `ECHILD` and not at level 1), each control making its test
  fail.
- `torch.sh`: every PyTorch operation on the GPU, computed by translated kernels, gives what
  the CPU gives (within the tolerance of different rounding).
- `llamacpp.sh`: the next-token distributions of a text are those of the CPU's (mean KL
  divergence at most 0.005, same most likely token at least 95 %).
- `llamacpp-bench.sh`, `llamacpp-ops.sh`: speed and correctness against a native sm_70 build.
- `vllm.sh`, `sglang.sh`: the tokens generated by the engine are those of the CPU's (plain
  greedy), or the layer says which kernel it cannot give; in stage c2 of `vllm.sh`, every token
  sampled is among the top k and within the top p of the engine's own log-probabilities. A
  stage's verdict is the last `verdict:` line that its program prints, and counts only when the
  program then ends by itself, with status 0, within its time (`tests/gpu/verdict.sh`). Stage e
  of `vllm.sh` runs the model in its own bfloat16, told 8.0 with `VF_CAPABILITY_ALL=1`, without
  compilation: vLLM 0.34 stops there at its first forward (`LIMITS.md`, "The capability told
  with `VF_CAPABILITY`"), which the stage says as LIMIT, with the layer's line that names the
  kernel, without changing the exit status; the same stop without that line is a FAIL.
- `zml.sh`: the program answers with the right word and reports a speed.
- `triton.sh`: each of the 180 cases of the quick subset of the Triton corpus gives on the V100
  what the same kernel gives on the CPU under Triton's interpreter (or the case's PyTorch
  computation), bit for bit or within the tolerance the case states; its five runs give the same
  bits; and the layer's two routes, the translated sm_75 machine code and the retargeted PTX of
  the same kernel, give the same bits. All of them on the same inputs: made once by the step that
  makes the references and saved with them, and named by their digest in the report. A case
  without a reference (the interpreter cannot run it) is compared between its runs and the routes
  only (`SKIP-REF`); one whose machine code the layer refuses and runs as the PTX kept in the
  cubin is `PASS-PTX`.
- `compare.sh`: the Rust translator and the Python reference give the same bytes, the same
  instruction map, the same `MAP.lives` and the same refusals for every cubin, sm_80 test
  cubins included (at 0.1.6: 492 of 492, the 17 test cubins and PyTorch's 475; 491 of 491 at 0.1.4
  and 0.1.5).
- `python3 -I reference/ampere_check.py DIR...` (not a step of the release check: it needs sm_80
  cubins, `vf extract FILE DIR 80`, and NVIDIA's disassembler): every distinct word of sm_80 code
  that the translator passes as it is reads the same to nvdisasm as sm_80 and as sm_75, and is of
  the same class with the same operands in both descriptions (0 fail on the sm_80 code of PyTorch
  2.14.0 and vLLM 0.34.0, 2026-11-25). It does not show that the translation runs right: an
  instruction of sm_80 passes by its class only in a form listed in
  `reference/ampere-evidence.txt`, with the kernel that ran right on a V100.
- `validate-corpus.sh`: the independent validator finds 0 problems over the corpus (of sm_75
  code: the validator does not read sm_80 code, `docs/validation.md`, section 8).
- `controls.sh`: the validator objects to translations that are made wrong on purpose (the
  controls of `reference/controls.py`, each in a process of its own), in every cubin whose
  translation a control changes but the cases looked at one by one (at 0.1.4: 113 controls, 110
  objected to; `docs/validation.md`, section 3, says what the other three are).
- `heldout-validate.sh`: the same, on a corpus kept apart from the one the rules were found on.
- `fuzz.py`: on damaged cubins, the two translators agree on what they make of every mutant.

## First run

The first run of a test that launches CUDA kernels through the proxy compiles them (the
driver's PTX compiler turns the retargeted PTX into machine code for the GPU). This takes
minutes (llama.cpp: 3 to 5 minutes for the small model; vLLM and SGLang: longer). The compiled
code is kept in `$VF_JIT_CACHE` and reused on later runs.

## Suites

### GPU tests (`tests/gpu/`)

Need a Tesla V100, the layer built (`scripts/build.sh`), the forwarding libraries built
(`scripts/build-forwarders.sh`), and the CUDA toolkits fetched (`scripts/fetch-all.sh`).

| Script | What it runs | Compared with | Time | What it needs |
|---|---|---|---|---|
| `run.sh` | test kernels through the proxy | native sm_70 builds, host models, RTX 5090 reference vectors | a minute or two | toolkits |
| `multi-gpu.sh` | one-process dual-GPU concurrency (`cuModuleLoadData`, `cuLibraryLoadData`, CUDA graphs) & peer copy | host exact computation | a few seconds | 2+ GPUs |
| `nccl.sh` | NCCL's collectives through PyTorch, one process for each of two GPUs, with the wheel's CUDA 13 NCCL (`cuda13`, its multicast instructions given as reported traps) or the CUDA 12 build (`cuda12`), as NCCL chooses and with each algorithm and protocol pinned; `--bench` times two all-reduces | the CPU, exactly | 85 s (`cuda13` once the driver's cache holds NCCL's code: its first run compiles it for about 15 minutes in each process), 40 s (`cuda12`) | 2 V100s, PyTorch 2.14 (`VF_TORCH`), the CUDA 12 NCCL for `cuda12` (`VF_NCCL12`) |
| `forward.sh` | forwarding libraries | host computation | a minute | toolkits |
| `torch.sh` | 925 PyTorch operations | the CPU | 3 min | PyTorch 2.14.0+cu130 wheel |
| `torch-models.sh` | 32 small whole models | the CPU | 4 min | PyTorch 2.14.0+cu130 wheel |
| `torch-bench.sh` | 89 PyTorch operations (speed) | PyTorch with sm_70 code | 12 min | both PyTorch wheels |
| `llamacpp.sh` | llama.cpp, official build | the CPU | 2 min | `scripts/fetch-all.sh demos` |
| `llamacpp-bench.sh` | llama.cpp (speed) | native sm_70 build | 10 min | demos, native build |
| `llamacpp-ops.sh` | llama.cpp's backend test; with `test` and `layer-only`, also the cases that run through the layer only (a flash-attention case that native sm_70 builds abort on, faulting the GPU: never run natively) | native sm_70 build (`perf`) or the CPU (`test`, `layer-only`) | 6 to 20 min | demos, native build, sources |
| `llamacpp-nvfp4.sh` | llama.cpp with NVFP4 weights | the CPU, native build | depends on the model | demos, native build, an NVFP4 GGUF model |
| `vllm.sh` | vLLM's engine and server | the CPU (greedy) | 10 min | vLLM 0.34.0, PyTorch, model, Python headers |
| `sglang.sh` | SGLang's engine and server | the CPU (greedy) | 10 min | SGLang 0.5.23, PyTorch, model, Python headers |
| `zml.sh` | ZML's LLM example | the word "Paris"; reports a speed | 5 min and the time to read the model twice | ZML build, a 9B model |
| `profile-torch.sh` | PyTorch with the layer's profile | the profile has the right columns | 15 min | both PyTorch wheels |
| `launcher.sh` | PyTorch and llama.cpp through the launcher (`vf`), a virtual environment made by `vf venv` whose Python is started with no prefix, and `VF_SAFE=1` with `VF_CUBLAS_BF16_TC=1` | a matmul against the CPU (within 1e-3); the forwarding cuBLAS in `/proc/self/maps`; text from llama.cpp; the layer says that it is there, by its variable and by the functions of `libcuda.so.1`; with safe mode, what `vf doctor` says holds: the bf16 cuBLAS variants (`bf16-safe.c`) give cuBLAS 12.9's bits with no call on their tensor-core path, and other bits without `VF_SAFE` | 10 min | PyTorch 2.14.0+cu130 wheel, demos |
| `vllm-topk-topp-run.sh` | vLLM's top-k and top-p sampling kernel (Triton), 18 cases, in two passes, each with a proxy cache of its own: at `VF_BORROW=1` (levels 2, 1, 0), then at `VF_BORROW=5` (level 3 with room made, as the second translation at launch translates) | the same kernel run on the CPU by Triton's interpreter | not timed | vLLM, PyTorch, Python headers; the GPU chosen by `CUDA_VISIBLE_DEVICES` |
| `triton-tc.sh` | Triton's matrix products with `VF_TRITON_TC` off, on, off again and on again, through the launcher of this tree, in four processes that share one cache of Triton: GEMMs of float16 (four configurations, one whose pipelined sm_80 kernel needs more than 96 KiB of shared memory), bfloat16, int8 and float32; with `VF_VLLM_VENV`, vLLM's `fused_moe_kernel` and `kernel_unified_attention` in float16 and bfloat16, and `torch.compile` of a matrix product with Inductor's Triton template alone | float64 on the CPU; on against off: what PyTorch, Triton and NVML are told (7.5 both), the target of each kernel's PTX (sm_80 on for the three types, sm_75 for float32 with the bits of off), the time (less on), the bits of the second off and on against the first | 15 min | PyTorch 2.14.0+cu130 with its Triton (or `VF_VLLM_VENV`), Python headers, the layer built with the forwarding and CUDA 12.9.2 libraries the launcher finds |
| `triton.sh` | 180 Triton kernels and configurations (the quick subset of `triton_corpus.py`, run by `triton-diff.py`): element-wise operations in every type, conversions, reductions, scans, sorts, `tl.dot` in several types, precisions, tiles, warps and stages, attention, gathers, atomics, loops, grids, libdevice; each through the translated machine code (5 runs) and through the retargeted PTX of the same kernel (2 runs) | Triton's interpreter on the CPU, or a PyTorch computation on the CPU; the runs and the two routes with each other | 4 min (2 of them for the interpreter) | PyTorch 2.14.0+cu130 with its Triton, numpy (`VF_TRITON_EXTRA`), Python headers; or `VF_TRITON_PYTHON`. `VF_TRITON_TC` puts `python/` first in `PYTHONPATH` (the kernels of its types are then compiled for sm_80, and their verdicts say so); `VF_TRITON_VENDOR=1` adds the kernels of vLLM or SGLang (`triton_vendor.py`) |
| `sglang-decode-graph.py` | SGLang's FP16 decode attention captured in CUDA graphs and replayed with new inputs, run through the layer with SGLang and PyTorch on `PYTHONPATH` | the CPU in FP32 | not timed | SGLang, PyTorch |

(Times on one V100 with everything already compiled and in the file cache.)

`tests/kernels/launch_rate.c` measures the time of a kernel launch through a `libcuda.so.1`
(NVIDIA's or the layer's), with and without a module with a refused kernel loaded; its header
says how to build and run it.

`run.sh` ends with what the layer says after a kernel faults (`tests/kernels/test_fault.c`). By
default only the control runs: the same kernels without the one that faults, after which nothing
is said (`fault-none`, `fault-none-silent`). The four cases in which a kernel writes through a null
pointer on purpose (`fault-candidates`, `fault-exact` under `VF_FAULT_TRACE=1`, `fault-graph`,
`fault-copy`), and the one in which a kernel reaches a reported trap (`reported-fault`: the
driver's line for the assertion, error 710, the layer's report naming the kernel with what it was
given with), end their CUDA context and add lines with "Xid" to the
kernel's log each: they run only with `VF_TEST_FAULTS=1` (exactly), which no release check and no
suite sets, and each log says how many such lines the kernel's log had before and after the case.

The whole Triton corpus (3,516 cases; `triton_corpus.py` says how cases are made and checked) is
run by hand, the reference on any machine and the GPU part on the V100, in as many processes as
wanted (`--shard I/N`; a case that stops its process is counted as a CRASH and the next process
goes on from the case after it):

```sh
TRITON_INTERPRET=1 python3 tests/gpu/triton-diff.py interp --level full --out REF --jobs 24
python3 tests/gpu/triton-diff.py gpu --route sass --ref REF --out sass.jsonl --level full --runs 5
python3 tests/gpu/triton-diff.py gpu --route ptx --ref REF --out ptx.jsonl --level full --runs 5
python3 tests/gpu/triton-diff.py report --level full sass.jsonl ptx.jsonl
```

(the `gpu` commands through the layer, with `VF_LOG_LEVEL=2` and `VF_LOG` set, so that each case
says which route its modules took; `interp` saves each case's inputs in `REF/inputs`, made in one
thread, and the `gpu` commands take them from there, so that a reference made on another machine
was computed on the inputs the GPU is given (made again, PyTorch's tanh, which is MKL's, does not
give the same last bits on every machine, nor in every process: `ld-erfinv-f64` once had other
inputs in one of the two processes of the routes, a block of 2,048 values that one thread computed
with MKL's AVX2 code of low accuracy); `triton-diff.py inputs --out DIR` makes the inputs alone and
prints the digest of all of them; `--vendor` adds the kernels of vLLM and SGLang of
`triton_vendor.py`, run with the Python of each engine's environment: 151 cases with vLLM, 90 with
SGLang; Triton builds its C launcher on the node, so `CPATH` must name Python 3.12 headers there).

`torch-ops.py` and `torch-models.py` run their cases in worker processes, which write their
verdicts on a descriptor or to a file of their own (what kernels print on standard output cannot
hide one); a case that has not ended after `$VF_CASE_WAIT` seconds (default 600) is stopped and
counted as a CRASH. `python3 tests/gpu/torch-workers.py` checks that on the host, with fake
workers (no GPU, no PyTorch, 15 seconds): kernels that print, a worker that fills its standard
error or does not end, a line that is not a verdict, fewer verdicts than cases, a second verdict
for a case, a verdict for a case that the worker did not start or did not name.
`python3 tests/gpu/vllm-engine-verdicts.py` checks, with a stand-in for vLLM (no GPU, no vLLM),
the verdict lines of `vllm-engine.py` (stages c and c2 of `vllm.sh`): fewer results than prompts,
or none, or a reference with fewer texts, do not pass, nor a sampled token whose log-probability is
NaN or above 0; and the verdict of a stage of `vllm.sh` and `sglang.sh` (`tests/gpu/verdict.sh`): a
program that prints its verdict and then fails, or does not end within its time, does not pass;
the LIMIT of stage e is said only for a program that ended with an error after the layer's line of
a kernel that only the code for the capability told has, and refused nothing.

### Launcher tests (`tests/launcher/`)

No GPU needed. Exercises the launcher (`vf`) on fake installations (both the archive and
source-tree layouts): environment variables, path order, map entries, user priority, options,
what is done when only some of the CUDA 12.9.2 libraries are there (a forwarding library with
nothing to forward to is not used), argument passing, exit status, symlinks, its long name,
`doctor` (and what it says of the layer's cache directory: one that its group can write is
refused by the layer, said with `chmod 700`), `fetch` command line, the
subcommands of the offline tool, help and error cases, the bf16 cuBLAS variants that
`VF_CUBLAS_BF16_TC` and `VF_CUBLAS_BF16_SKINNY` select (both names mapped or none, what is
refused, what `doctor` says); safe mode (`VF_SAFE`: what `env` prints, what `doctor` says, what
reaches a command, the libraries that the bf16 switches select left as they are); and `vf venv` with the Python of the
machine: an environment whose Python runs through the launcher with no prefix (arguments,
exit status, standard input, a child process), what it refuses, how it is turned off.

| Script | What it does | Time | What it needs |
|---|---|---|---|
| `launcher.sh` | 117 checks; run it under `dash` and under `bash --posix` | seconds | the launcher (`scripts/vf`); python3 3.10 or later with `venv` for the checks of `vf venv` |
| `install.py` | `install.sh` on local releases: an upgrade that keeps the CUDA 12.9.2 libraries and the `vf venv` environments of the release before, installs stopped half-way, two at once, what killed ones left, plain HTTP refused, redirections from HTTPS (through curl and through wget); under `dash`, or `bash --posix` with `VF_TEST_SHELL=bash` | seconds | dash, flock, python3 3.10 or later with `venv`; openssl for the redirections (skipped without it) |
| `fetch-redist.py` | `scripts/fetch-redist.py` on local archives: retries after an interruption, concurrent fetches, runtime-only installs, the NCCL wheel, a failure or a signal half-way, a component killed while it is moved into place (the journal, which the next fetch completes), what an interrupted fetch left, a manifest over plain HTTP | seconds | python3, tar |

Run the last two with `python3 tests/launcher/install.py` and `python3 tests/launcher/fetch-redist.py`.

### Tests of `VF_TRITON_TC` (`tests/python/`)

No GPU needed. `python3 tests/python/triton_tc.py` (87 cases, 80 without a Triton; the layer
built, for the marker of its `libcuda.so.1`; python3 3.9 or later with `venv`; a short `TMPDIR`,
for the socket of `multiprocessing`'s forkserver) runs the launcher on an installation of the
archive's layout made for the test, with a stand-in for the offline tool and a stand-in for Triton
(a package `triton` whose `CUDABackend` has the methods of Triton 3.3 to 3.9 and makes cubins with
a table of sections): the `PYTHONPATH` that the launcher gives, its default and `VF_SAFE`; the
import hook in a Python it starts, in that Python's children, in workers of `multiprocessing`
(spawn, fork, forkserver), in a `vf venv` environment started with no prefix; two copies of
`python/` on the path; the `sitecustomize` of the user, of the distribution and of an import finder
still run, and `importlib.reload` of it; nothing under `-I`, `-E`, `-S`, nothing written into
`python/`; which kernels are compiled for sm_80 and which keep Triton's stages (the guards: a stage
replaced by a hook, several CTAs, an unknown size of shared memory; the skinny products of
`SLOWER`); fewer stages for the shared memory, and the stages used in the metadata; every failure
of the sm_80 compilation (an error, a crash or a hang of the compiler in a helper, a helper killed
or stopped from outside, one killed while its `ptxas` runs, a helper that cannot import Triton,
imports it slowly or never, one that ends on its own as a request reaches it, an executable that
is not the interpreter, another Triton in the
helper, too much shared memory, a cubin without its PTX, a kernel that the tool refuses or makes
trap) leaves the kernel to Triton, and what Triton's cache keeps of it; the helpers of a process
tree (threads, a `fork()` once the program has helpers, processes it starts, `multiprocessing`'s
workers, a helper started by a thread that ended, a `fork()` while another thread holds the
module's lock) and their end (idle, with the program killed during a compilation and its `ptxas`,
when the program ends during one, by its own bound when it cannot see its program end); what a
helper runs of the program (`-I -S`, its environment, `sys.path`, working directory, session,
descriptors, no bytecode where the program writes none); its temporary files (in the temporary
directory where Triton writes, removed, nothing of the places' directory removed; a FIFO or a link
where the cubin is, its directory renamed and replaced during a compilation); the places of the
helpers (2 places for 6 processes, places held without end: one wait per process, also for its
threads that wait together, which then share its helper without a place, an idle helper
that gives its place up, 100 processes at once, lock files that cannot be used: no wait, the
directory of Triton's cache when the first eligible kernel is compiled, that directory replaced by
a link after it was opened); a Triton changed on disk under the program; a subreaper program, an
`exec()`, a `fork()` after the program closed its descriptors, an abort tried again once, much
written before a crash, no process that can be started; what a request may ask (each value: `ptxas` options, libraries, an IR,
plain values without a line end, a path of 7 MB, JSON nested too deep, a message over 64 MiB) and
what an answer must be (not an ELF, for another SM, a kept PTX that is not the answer's, a changed
metadata); the bounds of each wait (a stopped helper, the program stopped during a compilation, a
program with no descriptor left, one that closes its descriptors, a compilation whose `ptxas`
outlives it, a send to a closed peer with SIGPIPE at its default); `ptxas` ended from outside or
out of disk; 1,100 descriptors; the mark in the backend's hash and in `triton_key` (with the
layer's settings), and its absence in a process that can use no helper; when nothing is done (not
a Volta, `VF_CAPABILITY`, `VF_SAFE=1`, no layer, a Triton whose methods differ); and, read in the
process, the rule of `SLOWER`, the checks of an answer's cubin and kept PTX, the messages, the
values of a request. With `VF_TRITON_PYTHON` (a Python with Triton; with `PYTHONPATH` naming
another Triton, that one), 7 more through the launcher of this tree and its tool:
`tests/python/triton_tc_compile.py` compiles kernels for 7.5 (float16, bfloat16 and int8 products
made for sm_80, each cubin the same as that of the helper's `compile80` run in the process, for six
configurations; float32 and no product the same cubin as off; the cache used off, on, off, on, and
with `VF_TRANSLATE=first` or `VF_PAIRS=0`; a failure of the tool that leaves nothing in the cache;
fewer stages for a block of 128x256x64; `TRITON_DISABLE_LINE_INFO=1` keeps sm_75; a skinny
bfloat16 product with B stored K x N at 7.5, with B stored N x K at sm_80). They were run with
Triton 3.3.1, 3.4.0, 3.5.1, 3.6.0, 3.7.1, 3.8.0 and 3.9.0 (2026-11-24, CPython 3.12; each installed
alone, without its dependencies, in a directory given by `PYTHONPATH`; with 3.3.1 two are skipped:
Triton alone compiles no product of halves for 7.5), and the 67 others with CPython 3.9 to 3.14
and PyPy 3.10 and 3.11.

### Tests of the machine-code translator (`tests/sass/`)

No GPU needed. They need the reference translator (`reference/`), the instruction descriptions
that it reads (fetched by `scripts/fetch-all.sh denvdis`), NVIDIA's disassembler for the
validator (in `toolkits/`), and the `vf` tool built (`cargo build --release -p vf-cli`).

| Script | What it does | Time | What it needs |
|---|---|---|---|
| `compare.sh` | the Rust translator against the Python reference: bytes, maps, lives, refusals | seconds for the test kernels; a minute for PyTorch's 475 cubins with 48 processes | `reference/`, `vf` tool, cubins |
| `validate-corpus.sh` | the independent validator over a corpus | a quarter of an hour for PyTorch's 475 cubins with 48 processes | `reference/`, `toolkits/`, cubins |
| `controls.sh` | every control of the validator, in five passes | 40 minutes with 64 processes for the cubins it names | `reference/`, `toolkits/`, cubins |
| `heldout-extract.sh` | extract the cubins of other libraries | minutes | `vf` tool, the libraries |
| `heldout-validate.sh` | translate and validate the held-out corpus, keeping why each function is refused | an hour with 40 processes | `reference/`, `toolkits/`, the corpus |
| `fuzz.py` | damaged cubins: the two translators must agree | two and a half minutes for 20,000 mutants with 20 processes | `reference/`, `vf` tool, small cubins |

### The second translation, by every way to it (`tests/proxy/`)

No GPU needed. A stand-in for NVIDIA's driver library, built by the test from C, refuses the
launches of the first image that is loaded; the proxy translates the test cubin and loads it
through the stand-in.

| Script | What it does | Time | What it needs |
|---|---|---|---|
| `fit_api.sh` | The second translation at launch by every way a program can come to it: a variable or a surface reference asked for by name (both exported names, and what `cuGetProcAddress` gives) excludes it; the launch entries for the per-thread default stream, a library's kernels taken by enumeration, and the functions of the module that the driver gives for a library are launched from it; a library and that module are one image; an enumeration into an array larger than the library's number of kernels reads only what the driver wrote; what is set on a library's kernel is given to the function of the same name in the library's module, for the device of its context, unless the function has a setting of its own, and when it is set later too; a launch whose second translation was being made when the program asked for data is refused; two settings of one kernel made at once leave it and its second image with the same value; an unload of the module waits for the settings given to its second image, for the lookup of its kernel there and for the launch from there; a second translation made for a module that goes meanwhile is not taken for the module loaded next with the same handle, and is unloaded when the program unloaded the module, not when its context was destroyed. 46 checks | seconds | the proxy built, `tests/kernels/k_fit_test.135.sm75.cubin`, gcc |
| `tests/proxy/graph_fit.sh [PROXY_LIBRARY]` | CUDA graph register pressure with a stand-in driver reading the actual translated ELF. Eight recording, update and capture paths under the usual, bounded and block-size-aware policies | seconds | the proxy, the same cubin and a C compiler |
| `marker.sh` | The functions by which a program can tell that the layer is there (`volta_forever_abi`, `volta_forever_version`): found by `dlsym` in the layer's `libcuda.so.1`, with the interface number 1 and the workspace's version; and that the library names no path of the crates as the build host keeps them (`/.cargo/registry/`, `CARGO_HOME`'s when it is set) | seconds | the proxy built, gcc |
| `lifecycle.sh` | What the layer keeps over the life of a process, and what it does at its end: a child made by `fork()` while another thread of the parent waits in a write of the log ends when it calls `exit()`; what the layer keeps of the modules of a context (adapted images, kernels, originals kept for a second translation, second translations) is dropped when the context is destroyed, released for the last time or reset, by both names of each call, and not otherwise (a release that is not the last, another context, a library, a module that another thread loads once the driver destroyed the context, in a context made at its address or with the handle of one of its modules, and a retain of the primary context by another thread meanwhile); while a library restored from the cache is adapted for a kernel whose code the cache does not have, a kernel of another library is given at once, the same kernel asked for by another thread is the same, and another kernel of the library asked for meanwhile is given from the same adaptation; a write to the cache that fails leaves no temporary file, and the first use of the cache removes one left an hour before; a file whose name is not UTF-8 is adapted as any other. Each case in a process of its own | seconds | the proxy built, gcc, `tests/kernels/k_fit_test.135.sm75.cubin` |

Twenty-one more host regressions exercise the built libraries directly:

```sh
python3 tests/proxy/library_entries.py dist/lib/libcuda.so.1
python3 tests/proxy/driver_capability.py dist/lib/libcuda.so.1
python3 tests/proxy/nvml_capability.py dist/lib/libvf_audit.so
python3 tests/proxy/link_refusals.py dist/lib/libcuda.so.1
python3 tests/proxy/nccl_remedy.py dist/lib/libcuda.so.1
python3 tests/proxy/jitlink_objects.py dist/lib/libnvJitLink.so.13 toolkits/cuda-12.9.2
python3 tests/proxy/jitlink_fork.py dist/lib/libnvJitLink.so.13
python3 tests/proxy/readonly_tables.py dist/lib/libcuda.so.1
python3 tests/proxy/launch_guard.py dist/lib/libcuda.so.1
tests/proxy/guard_stress.sh
python3 tests/proxy/safe_mode.py dist/lib/libcuda.so.1
python3 tests/proxy/log_lines.py dist/lib/libcuda.so.1
python3 tests/proxy/summary_file.py dist/lib/libcuda.so.1
python3 tests/proxy/nocode_launch.py dist/lib/libcuda.so.1
python3 tests/proxy/told.py dist/lib/libcuda.so.1
python3 tests/proxy/profile_failure.py dist/lib/libcuda.so.1
python3 tests/proxy/fault.py dist/lib/libcuda.so.1
python3 tests/proxy/driver_calls.py dist/lib/libcuda.so.1
python3 tests/proxy/ptx_version.py dist/lib/libcuda.so.1
python3 tests/proxy/parallel_threads.py dist/lib/libcuda.so.1
python3 tests/proxy/summary_fill.py dist/lib/libcuda.so.1
python3 tests/proxy/reported_traps.py dist/lib/libcuda.so.1
```

The first checks filtered kernel counts and enumeration with a stand-in driver, including
deferred loads and the lifetime of replacement images, and that a lookup that fails leaves NULL
in the handle (refused by the layer, by the driver's compiler when a deferred library is loaded
whole for it, or of a name the library does not have). The second checks the compute capability that the proxy gives (device attributes,
`cuDeviceComputeCapability`, the attributes of a function and of a kernel) to callers whose
code is for different GPUs, on a stand-in driver with a V100 and three later cards, with and
without the launcher's marker and `VF_CAPABILITY`; it needs gcc and objcopy. The third checks
the capability that NVML gives through the audit library, by `ctypes` (as `pynvml` asks) and
to a library linked against it; it needs gcc. The fourth checks, with another stand-in, that
a kernel refused in an input of the driver's linker is refused in the image the link makes,
and in a copy of it, that a file of PTX, a cubin or a fat binary given with `cuLinkAddFile` or
`cuLinkAddFile_v2` reaches the linker as what the layer makes of the same bytes in memory, its
refused kernel refused in that image with the line of a refused kernel, while a file of another
type (a host object, a library of them), one that cannot be read, a null path and a file of which
the driver refuses what the layer made go to the driver's call (and the program gets its answer),
that a kernel that the note of a cubin names (as the layer's nvJitLink
writes it) is refused wherever the cubin is loaded (each loading function, a file, a fat binary,
a wrapper, an input of the driver's linker), and that what is known of a link or a module made
with a handle that the driver gives again, while the destruction or unload of the one before
returns, is kept. The fifth
checks that a refused kernel of NCCL makes the layer say once, at the default log level, that
NCCL's CUDA 12 build is the remedy. The sixth links a native sm_70 device function from both a
host object and a static library with the CUDA 12.9 toolkit, reads a file whose name is not
UTF-8 by that name (not the one its text names), and checks that PTX in which every kernel
traps is refused as text and in a fat binary alike (and Turing machine code whose every kernel
the translator refuses), and that with one kernel that runs both are linked, into a cubin that
names the kernel that traps in a note, which a link of that cubin keeps. The seventh
checks, with a stand-in for nvJitLink 12.9.2, that a child made by `fork()` while another
thread of the parent waits in a write of the adapter's log ends when it calls `exit()`; it
needs a C compiler. The eighth checks, with another stand-in that keeps the text of what it is
given, that a deferred library gives a kernel that only reads a table of its module from a
sub-module that holds a copy of the table, and then refuses the table by every name the program
could ask for it with (error 801; a name that the module does not have stays 500), while a
variable that kernels write is given; and that a table
the program was given first keeps the kernels that read it in the whole module, also one that
another thread asks for while the whole module is compiled for the table (the stand-in holds that
compilation). Each case in a
process of its own, a second time from the proxy cache; it needs a C compiler. The ninth checks,
with another stand-in that keeps what NVIDIA's driver keeps by handle and that the test also calls
directly (as a program takes a handle through a private interface of the driver, or from a table
of `cuGetExportTable` as cuFFT does), that a refused kernel of a module or of a library, and its
function in a context, is not launched by any launch function, not made a kernel node of a graph
or given to one, and not asked about (218, and the driver is not called); that the kernel that
runs beside it is; that the driver is not asked what a handle is while nothing is refused, and
once only for a handle found allowed, also when the program asks for the function of a kernel
before each launch; that a handle that the driver gives again after an unload or
the end of a context (destroyed, or a green context) is taken for what it is now; and that a
library's refused kernel is launched on a later GPU, unless the library is deferred. Each case in a
process of its own, with deferred libraries and without, and with `VF_FIT=0`; it needs a C
compiler. The tenth runs the same refusal under threads, against a stand-in that threads can share
and that gives the handles of what is gone again, last freed first, to any thread
(`guard_mt_mock_driver.c`; `guard_stress.c` says what each scenario checks): no refused kernel
reaches the driver and no kernel that runs is refused while threads load and unload modules, and
make contexts, take a library's functions in them from the driver and destroy them (or green
contexts) while the driver gives their handles to the next thread; a function named as a refused
kernel elsewhere runs. Each scenario in a process of its own, those with threads for
`VF_GUARD_STRESS_SECONDS` seconds each (default 1, the CI's; a long run takes 60 or more); it needs
gcc.
The eleventh checks safe mode (`VF_SAFE=1`), with a stand-in that keeps every image it is given: each switch that safe
mode turns off (`VF_BF16_TC_GROUPS`, `VF_BF16_TC_TILES`, `VF_BF16_COLUMN=tree`, `VF_BF16_ROW=tree`,
`VF_TRANSLATE=first` and `force`) changes what the driver is given, and with `VF_SAFE=1` the driver
is given the default's image, with one line at level 1 that names the switch; machine code with
pairs of products (`k_pair_bits.135.sm75.cubin`) is given translated as with `VF_PAIRS=0`, and the
line says `VF_PAIRS in machine code`, while the pairs of the PTX lowering stay; `0`, an empty value,
`true`, ` 1` and the like leave the switch in use (each but `0` said once); the `ordered` modes,
`VF_PAIRS=0` and `VF_ALIGNED_ROWS=0` are left as they are; and the proxy's cache keeps what a run
with safe mode and one without make under keys of their own, in either order, for translations of
machine code (one the same in both modes, one with pairs) and for a library compiled kernel by
kernel. Each case in a process of its own; it
needs a C compiler. The twelfth checks,
with the stand-in of the fourth, that from `VF_LOG_LEVEL=2` the lines about an image name it (by
the file of `cuModuleLoad`, by the size and XXH64 digest of an image that `cuModuleLoadData` or
`cuLibraryLoadData` is given), so that two kernels of one name in two images are told apart, and
that the default level prints none of them; and that the lines of a kernel with emulated
instructions and of a refused kernel are said once for each kernel, its name and its code, or for
a refused kernel its name and why (a second kernel of a name, of other code or refused otherwise,
said, from level 2 with its image's digest; the same image loaded again not said again), within
`VF_EMULATION_WARNINGS`, and that past 4,096 kernels of one name from as many images the next are
neither listed nor counted (what is kept of them is bounded); it needs a C compiler. The
thirteenth checks, with the same stand-in, that with `VF_SUMMARY` the report at exit is in its file
while the process runs, rewritten when it changes, and still there, as it last was, once the
process is killed with SIGKILL; that it is written at exit (with `VF_STATS`'s counts); that no
thread is started without the variable; and that children made by `fork()` while the thread
writes end and write no report of their own; and that no link put where that file or the profile
(`VF_PROFILE`) goes, or where the file was written first, leads either into the file it names,
both being their owner's only; it needs a C compiler. The fourteenth checks, with the stand-in of
the ninth and its calls, that a kernel without code (its body in the image is a trap and nothing
else) of a module or of a library, and its function in a context, is given at the lookup with
nothing said and answered about, while no launch function runs it and no kernel node of a graph is
made or given with it (218, and the driver is not called), with one line at the default level that
names it and its image; that the kernels beside it that trap behind a test or after a store, and
the empty one, run; and that its handle, given again by the driver to a kernel that is code, is
launched. Each case in a process of its own, with deferred libraries and without; it needs a C
compiler. The fifteenth checks, with the same stand-in, whose synchronisation returns what the test
sets, that with `VF_PROFILE` the first synchronisation after a launch that returns an error is said
at once, with the kernel it followed and its sizes, and only that one; it needs a C compiler.
The sixteenth checks, with a stand-in that shows a fault as NVIDIA's driver does (the launch of `write_null` returns 0, the next call that
waits returns 700, and every call that takes the context after it), what the layer says after a
fault: once, at the default level, the launches of the context that no synchronisation covers, the
last first, with what the layer made of each image and its size and digest, and again in the report
at exit (`VF_SUMMARY`); not those of another context, nor those made before a synchronisation of
their stream (another stream's, the legacy stream's, the thread's own default stream's), and those
of every thread, by thread; a CUDA graph's launch with the kernel nodes captured in it; a fault
that only a copy met, found at the next routed call (a launch), and not said when the context is
destroyed first or when the program exits without another routed call (the layer asks the driver
nothing at exit); that a program whose signal handler calls `exit()` during a call of the driver
that holds a lock of the driver's ends (with a stand-in that raises the signal while it holds one),
and that the report at exit made then has the fault reported before; under `VF_FAULT_TRACE=1`
(only that value) the launch whose synchronisation met the fault named, alone when nothing else
is uncovered (with a launch made during a capture elsewhere, which was not waited for), with what
is not ruled out, and its launch returning the error, and
nothing waited for while a graph is being captured, nor after an end of a capture that failed on a
stream that joined it, while another capture is under way; that the report ends while another thread
destroys contexts that have modules (120 threads' launches looked up meanwhile); that the waits of
`VF_PROFILE` are the program's synchronisations for the report; that a fault of a context met by
a wait for its stream while another is current is said of the stream's context (and one met on a
stream whose handle outlived its context, given to a stream of another context, of a stream of a
context since destroyed), and one met by a wait for an event, or with no context current, of a context that the layer cannot tell, with the
launches of every context, and again of its context later; that faults of six contexts, each met
first by a wait for an event, are said once and then again for each one with launches made since,
four times at most, and once more after a context is destroyed; that with `VF_FAULT_REPORT=0`
(only that value) nothing is said or waited for and a lookup of a synchronisation leads to the
driver; and nothing said when nothing faults; that an uncorrectable ECC error (214) is said to be the hardware's, with what ran then; that
the report at exit keeps the caveats of its line (launches a thread's record no longer holds); that
with no launch known nothing is said of where the kernel was launched; and that a capture during which the program retains and releases the primary context and
destroys another context, or at whose end the program exits, is not invalidated. Each case in a
process of its own; it needs a C compiler. The seventeenth records the calls that reach the driver
(a recorder between the layer and the stand-ins of `lifecycle.c` and `fault.py`, which exports every
function of NVIDIA's library and gives its own function for each pointer that a lookup gives) in the
cases of those two in which nothing faults, and of `driver_calls.c`, which makes its calls through
the pointers of `cuGetProcAddress_v2` as the CUDA runtime does, and by the exported names (the same
calls, but for the lookups), with the report of
faults and with `VF_FAULT_REPORT=0`, and checks that they are the same, thread by thread, but for
the two getters that the report calls without a fault, which it names: `cuCtxGetCurrent` after a
synchronisation that succeeded, `cuStreamGetCaptureInfo_v2` while a capture is under way (any
other call fails the test, and so does one of these that no case shows); given a second
`libcuda.so.1` (of 0.1.5), that the calls with `VF_FAULT_REPORT=0` are the very same as with it, on
both paths. It needs a C compiler. The eighteenth checks, with the stand-in of the ninth, which
answers the questions about a function or a kernel as NVIDIA's driver does (the architecture of the
device for its machine code, the latest `.target` of its PTX that the device reaches for its PTX),
that code the layer retargeted from PTX for sm_75 or sm_80 is said to be compiled from that PTX
(`CU_FUNC_ATTRIBUTE_PTX_VERSION`), at most the capability that the caller is told, and its machine
code for that capability: through a module loaded from memory and from a file, a library loaded from
memory (deferred, its kernels compiled alone, and whole) and from a file, its kernel asked about as a
kernel and as a function, the kernel's function, the function of the library's module in a context,
and a kernel asked for after that module; set up from the layer's cache; for a program whose code is
PTX for sm_75 only, the CUDA 13 runtime and a library with code for the V100 (told the truth: the
driver's answers), by default, with `VF_CAPABILITY=8.0` (and `VF_CAPABILITY_ALL=1`) and 7.0, under
the launcher and not; on a later card, the driver's answers for the program's own code, and the
PTX's architecture for the layer's code there (of a deferred library, or of one loaded from PTX
text). Until 2026-11-24 the PTX was said to be for sm_70, which made CUB's algorithms answer
"invalid device function". Each setting in a process of its own; it needs gcc and objcopy.
The nineteenth checks the fallback of the parallel lowering and analysis of PTX (`VF_THREADS`) to
the calling thread when no worker thread can be started, on a module of 20 kernels and more than
128 KiB: that `vf-translate adapt` writes the same bytes with `VF_THREADS=1`, with `VF_THREADS=4` and
with `VF_THREADS=4` under `RLIMIT_NPROC=1` (this part is skipped, without a line, when
`dist/bin/vf-translate` is not there); and that, with the stand-in of the fourth under
`RLIMIT_NPROC=1` and `VF_THREADS=4`, `cuModuleLoadData` and `cuLibraryLoadData` of that module
retarget it and give its kernel, with no internal error and nothing loaded as is (what the driver is
given is not compared). It needs a C compiler; run as root, `RLIMIT_NPROC` refuses no thread, and
the fallback is not reached.
The twentieth checks the lines that another library puts in
`VF_SUMMARY`'s file through the layer's `vf_summary_add` and `vf_summary_remove`, with the stand-in of
the thirteenth and a library of the test's (`summary_fill.c`) whose functions give its lines: the
return codes (a name that begins with `volta-forever`, in any case, refused), the four slots and a
name that takes its slot back, the functions found on the layer's handle and not on the driver's; a
library's lines after the layer's, with its name in front, written again when they change and kept
after SIGKILL, every check of the file made on whole lines from their start; functions that fill their whole buffer with control
bytes, bytes that are not UTF-8 and valid UTF-8 between them, and return more than its size, leave
the file valid UTF-8 without control characters, each line with a known prefix, the text read up to
the buffer's end and no further; a removal that waits for the call under way (of 300 ms), calls the
function once more, keeps its lines marked as its last report, and calls nothing after; a function
that calls the layer back, refused, not left waiting; functions that break the contract: one that never
returns (a fork from another thread does not wait for it, and the last write waits 2 s for it, then is made
without the libraries' lines and says so, 2 s of the monotonic clock with the time of day an hour ahead
under `clock_ahead.c`), one that forks at each write, one that calls `exit(3)` on the
layer's thread (the process ends with that status, its last write made, and its library's removal in its
destructor returns 0); after the last write, a removal returning `EBUSY` at once while a call is under way
(the function that never returns, removed by its library's destructor; one that runs 3 s, whose library
unmaps the page it reads when its removal returns 0: the page stays, the call ends well, and the function
after it in the slots is not called; a removal's own call of the function, stalled 3 s after its look at
the last write by `stall.c`, the destructor's removal of the same function then returning `EBUSY`; the
layer's thread stalled between its announcement of a call and the call, the same); a function that forks
on the layer's thread, its child returning
from it: the child ends at once and writes no file, nothing of the program's running there (its exit
handler runs in the parent only, and a line left in a buffer of the C library is written once); the
layer's thread blocking every signal but those
of a fault (its `SigBlk`), the thread that loaded the layer keeping its mask, and a fault there reaching
the program's crash handler (Python's faulthandler); children made by `fork()` while the thread
writes and holds the lock, in which both functions do nothing, nothing is called, and which end and
write nothing; without `VF_SUMMARY` no thread and no call, and without the thread (it cannot be
started under `RLIMIT_NPROC`) the lines at exit; at exit, the library loaded after the layer called
by the layer's last write and its removal after it calling nothing, loaded before it removed first
with its last report written; a library unloaded by `dlclose` while the thread runs, which removes
itself in its destructor (its last report stays, the process lives on), and one pinned, which stays,
its lines written as they change; and, the control, one unloaded without removing itself, into which
the thread calls (the process ends by SIGSEGV). Each case in a process of its own, killed with its
children when a wait of the test passes 60 s; it needs a C compiler (`summary_fill.c`,
`link_mock_driver.c`, `clock_ahead.c`, a `clock_gettime` put in front of the C library's, and `stall.c`,
a `calloc` and a `getpid` that stall one thread once).
The twenty-first checks, with the stand-in of the fourth and devices whose
multicast it chooses, that a kernel of NCCL's name that calls a function holding a multimem
instruction is given with a reported trap, said in one line for its image (the kernel itself from
level 2) and without the line about NCCL's CUDA 12 build, when no GPU of the process can take part
in multicast; refused (218, with that line) when the second GPU can, when the driver does not
answer the attribute, with `VF_REPORTED_TRAPS=0`, and in both loads of a process where the first
question failed; that the driver's answer is said from level 2, and at the default level only once
after a refusal that it causes (nothing for a kernel given, nor for a program without a multimem
instruction, whatever the answer); that before the stand-in's slow load of a module of 17 MiB, and before its slow
completion of a link of such PTX by the program, the layer says, once, what the driver compiles
(naming NCCL's CUDA 12 build for NCCL's kernels only); and that a cubin
that the driver's linker made from such PTX, given as the relocatable fat binary of a wrapper, has
its kernel said. It needs a C compiler.

### Forwarding stack test (`tests/forward/`)

No GPU needed. Builds and tests the status-renumbering trampolines of the forwarding
libraries against a mock library.

| Script | What it does | Time | What it needs |
|---|---|---|---|
| `stack.sh` | stack-word copying correctness | seconds | Cargo (`vf-fwdgen` crate) |
| `package-refusals.sh` | what `scripts/package.sh` refuses to pack: a built tree with a forwarding library missing, with one older than the sources, with the report of a run for one library only, with some but not all of the three files of the bf16 cuBLAS variants (`forward/bf16tc`); and that it packs the three in `forward/bf16tc` when they are all there | seconds | a built `dist/` (`scripts/build.sh`, `scripts/build-forwarders.sh`) |

### Build scripts (`tests/scripts/`)

No GPU needed.

| Script | What it does | Time | What it needs |
|---|---|---|---|
| `nvcc-host.sh` | `scripts/nvcc-host.sh`, through which every script and test that runs nvcc goes: on the machine as it is, nothing added and nothing said where nvcc compiles a probe as it is, and on a newer system (Ubuntu 26.04 for CUDA 12.9.2), what it adds builds a program that calls `rsqrt`, `sinpi` and `cospi`, which prints the right values (said skipped, `newer-system`, on a machine where nvcc compiles the probe as it is); with a C library that declares them as glibc 2.43 does (a `bits/mathcalls.h` of the test's, on any machine), the program does not compile without `scripts/nvcc-host` and does with what the helper adds; with a default host compiler that says it is GCC 99, a `g++-N` of `PATH` is chosen (skipped without one) | seconds | `toolkits/cuda-12.9.2` (and `cuda-13.5.0` when it is there), gcc |
| `kernel-log.sh` | `tests/gpu/kernel-log.sh`, where `tests/gpu/run.sh` reads the kernel's log to count its Xid lines, with stand-ins for `dmesg`, `journalctl` and `sudo` first in `PATH`: `dmesg` when it can be read; else `journalctl -k` when it gives a line (it gives none, with status 0, to a user outside the groups `adm` and `systemd-journal`); else `sudo -n dmesg`, sudo never run without `-n`; else no source, and nothing counted | seconds | bash |
| `target-dir.sh` | Where the build scripts have cargo build a checkout: two checkouts in two directories of their own (`scripts/target-dir.sh`), each naming its checkout in its `CHECKOUT` (`--list` says ORPHAN of one whose checkout is gone), each script of a checkout in the same one, `CARGO_TARGET_DIR` when it is set; `scripts/build.sh` remaps the paths of the crates (of `CARGO_HOME`, else of `~/.cargo`, whatever `HOME` is), of the build directory and of the checkout; with `--build` (as the CI and the release check run it), two copies of the tree, the second's `crates/vf-fwdgen/bf16tc/runtime.c` changed and all its files older than the first's outputs, each built by `scripts/build-forwarders.sh`: each generator embeds its own `runtime.c` (with one directory for every checkout, the second had the first's) | seconds | bash, git; cargo and gcc for `--build` |
| `untouched-home.sh` | Not a test but how the CI and the release check run the unit tests: `tests/scripts/untouched-home.sh COMMAND` runs COMMAND with `HOME` and the XDG directories (`XDG_CACHE_HOME` and the others) in an empty directory made for it, `VF_PROXY_CACHE` unset (`CARGO_HOME` and `RUSTUP_HOME` where they were), lists what COMMAND left there and fails when it left anything: what a test leaves there, it leaves in the home of whoever runs it (one kept a translation in the layer's cache, `~/.cache/volta-forever/proxy`, at each run) | as COMMAND takes | bash |

### What the CI runs

On every push and pull request, `.github/workflows/ci.yml` runs the formatting and lint
checks and the unit tests (`cargo test`, 538 at 0.1.6, in an empty home of their own that they must
leave empty: `tests/scripts/untouched-home.sh`); `scripts/build.sh`; `fit_api.sh`, `marker.sh`,
`graph_fit.sh`, `lifecycle.sh`, `library_entries.py`, `readonly_tables.py`, `driver_capability.py`,
`link_refusals.py`, `nccl_remedy.py`, `launch_guard.py`, `fault.py`, `driver_calls.py`, `guard_stress.sh`, `jitlink_fork.py`, `safe_mode.py`,
`log_lines.py`, `summary_file.py`, `summary_fill.py`, `nocode_launch.py`, `told.py`, `profile_failure.py`, `ptx_version.py`,
`parallel_threads.py`, `reported_traps.py` and `nvml_capability.py` of `tests/proxy/`; `tests/forward/stack.sh` and
`tests/forward/bf16.sh`; `tests/scripts/kernel-log.sh`; `launcher.sh` under `dash` and under `bash --posix`;
`tests/python/triton_tc.py` (without a Triton);
`tests/gpu/torch-workers.py` and `tests/gpu/vllm-engine-verdicts.py`; `install.py` under `dash`
and with `VF_TEST_SHELL=bash`, and `fetch-redist.py`; `tests/sass/compare.sh` on the test
kernels; then `install.sh` on five distributions. It does not run `jitlink_objects.py` and
`tests/scripts/nvcc-host.sh`, which need the CUDA 12.9.2 toolkit, nor
`tests/forward/package-refusals.sh`, which needs the forwarding libraries built from NVIDIA's
toolkits (`scripts/fetch-all.sh redist`, `scripts/build-forwarders.sh`): it fetches neither.
The tests on a GPU and the SASS tests over whole corpora are run by hand.

### Checking a release

`tests/release-check.sh` runs, one after the other, what needs no GPU, and ends with a table (each
step, PASS, FAIL or SKIP, its time, and the log of a step that failed) and the exit status 1 when a
step failed, 2 when none failed but one could not run here (what it needs is missing: the CUDA
12.9.2 toolkit, the forwarding libraries, the instruction descriptions), 0 otherwise:

- the CI's host steps (formatting, lints with warnings as errors, the unit tests, `scripts/build.sh`
  and every host suite of the list above); the host suites that the CI does not run, when what
  they need is here (`jitlink_objects.py`, `package-refusals.sh`, `scripts/build-bf16-kernels.sh
  --check`, `tests/scripts/nvcc-host.sh`); and `compare.sh`, on the test kernels and on PyTorch's
  cubins when `TORCH` names them (about ten minutes in all with `JOBS=6`, eight of them for `compare.sh`);
- with `--full`, then the validator over the corpora: `validate-corpus.sh` on `TORCH`,
  `heldout-validate.sh` on `HELDOUT` and `controls.sh` (hours; a step whose corpus is missing is
  skipped);
- last, `executable`: every script that a step or a command of the GPU part runs directly has the
  mode 100755 in git's index, which is what a checkout gives it.

The GPU part is printed, not run: the commands of `tests/gpu/`, from the tree, with the settings a
release is checked with (`run.sh` at the default capability and at `VF_CAPABILITY=8.0`;
`forward.sh`, `forward-bf16.sh`; `torch.sh` and `torch-models.sh` at the default, at
`VF_CAPABILITY=8.0` and at `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1`; `triton.sh`,
`vllm-topk-topp-run.sh`, `llamacpp.sh`, `launcher.sh`; `vllm.sh` and `sglang.sh` with
`VF_VLLM_VENV` and `VF_SGLANG_VENV` when they are set; `zml.sh`), then how to run them on an
installed archive instead of the tree: the GPU scripts take the tree's `dist/` (the forwarding
libraries of `dist/forward`, the launcher that `launcher.sh` runs), so the archive's directories
and launcher have to stand in a copy of the tree, with a new `VF_OUT`. With
`VF_RELEASE_GPU_RUN='PREFIX'`, each command of the tree is also run, as `PREFIX 'COMMAND'` (a
runner of jobs that takes a shell command line and holds a GPU for it), one after the other, and
put in the table; `VF_RELEASE_GPU_TREE` is the path of the tree where the command runs, and
`VF_GPU` the GPU's index there (by default the first of `CUDA_VISIBLE_DEVICES`, else 0). Every step
writes its output to `$VF_OUT/release-check/`.

## Preparing the dependencies

### Toolkits and test models

```sh
scripts/fetch-all.sh redist cudnn   # CUDA 12.9.2 and 13.5.0, cuDNN, from NVIDIA's servers (tens of GB)
scripts/fetch-all.sh demos          # the official llama.cpp build, two small models
scripts/fetch-all.sh denvdis        # the instruction descriptions, for reference/
scripts/fetch-all.sh src            # sources that some tests build (llama.cpp's backend test)
```

### PyTorch 2.14.0+cu130 (stock wheel, sm_75 and later)

```sh
python3 -m pip install --target $VF_TEST_ROOT/torch-cu130/site \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple torch==2.14.0
```

### PyTorch 2.14.0+cu126 (native sm_70 code, for benchmarks)

```sh
python3 -m pip install --target $VF_TEST_ROOT/torch-cu126/site \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple torch==2.14.0
```

### Native llama.cpp build for sm_70

A build of the same llama.cpp sources for sm_70 with CUDA 12.9.2, for speed comparisons. nvcc
is given what `scripts/nvcc-host.sh` finds that it needs on this machine (nothing on most); the
toolkit's paths are absolute (CMake 4.2 does not find nvcc from relative ones).

```sh
export NVCC_PREPEND_FLAGS="$(scripts/nvcc-host.sh --args toolkits/cuda-12.9.2/bin/nvcc | tr '\n' ' ')"
cmake -S src/llama.cpp-b12418 -B $VF_TEST_ROOT/llama-b12418-sm70/build -G Ninja \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCUDAToolkit_ROOT=$PWD/toolkits/cuda-12.9.2 \
    -DCMAKE_CUDA_COMPILER=$PWD/toolkits/cuda-12.9.2/bin/nvcc \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=ON \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
ninja -C $VF_TEST_ROOT/llama-b12418-sm70/build llama-completion llama-bench llama-perplexity
```

Run it with `LD_LIBRARY_PATH=$VF_TEST_ROOT/llama-b12418-sm70/build/bin:toolkits/cuda-12.9.2/lib`.

### vLLM

The simplest way:

```sh
python3 -m venv $VF_TEST_ROOT/vllm-venv
$VF_TEST_ROOT/vllm-venv/bin/pip install vllm==0.34.0
```

Then set `VF_VLLM_VENV=$VF_TEST_ROOT/vllm-venv`.

### SGLang

```sh
python3 -m venv $VF_TEST_ROOT/sglang-venv
$VF_TEST_ROOT/sglang-venv/bin/pip install sglang==0.5.23
```

Then set `VF_SGLANG_VENV=$VF_TEST_ROOT/sglang-venv`.

### ZML

Build with `tests/gpu/zml/build-zml.sh` and set `VF_ZML` to the build directory. The model
(`VF_ZML_MODEL`) is a 9 GB checkpoint that must be provided; the script stops and says what
to set when it is not found.

### PyTorch cubins (for `tests/sass/`)

Extract the cubins from the PyTorch wheel:

```sh
vf extract $VF_TEST_ROOT/torch-cu130/site/torch/lib/libtorch_cuda.so \
    $VF_TEST_ROOT/torch-cubins 75
```

### Instruction descriptions (for `tests/sass/`)

```sh
scripts/fetch-all.sh denvdis
```

## Files outside `tests/` that the suites depend on

- `scripts/build.sh`, `scripts/build-forwarders.sh`: build the layer and the forwarders
- `scripts/fetch-all.sh`: fetch the CUDA toolkits, test models and instruction descriptions
- `reference/`: the Python reference translator, validator and batch translator
- `toolkits/`: the CUDA 12.9.2 and 13.5.0 redistributable libraries (fetched)
- `demos/`: the official llama.cpp build and test models (fetched)
- `src/`: llama.cpp's sources for `llamacpp-ops.sh`, the instruction descriptions for `reference/` (fetched)
- `dist/`: the built layer (`dist/lib`) and forwarding libraries (`dist/forward`)
- `crates/vf-fatbin/tests/data/`: fat binary test files used by `run.sh`
- `tests/gpu/zml/build-zml.sh`: builds ZML for `zml.sh`

## Testing an installed archive

The GPU scripts test this tree: `run.sh` and `torch.sh` take the proxy from `VF_PROXY`, but the
forwarding libraries from this tree's `dist/forward` (the forwarded-runtime cases of `run.sh`, the
map that `torch.sh` writes), and `launcher.sh` runs this tree's `scripts/vf`, which uses `dist/`.
So `VF_PROXY=ARCHIVE/lib` alone does not test a release archive. What 0.1.4 and 0.1.5 were
tested with (`tests/release-check.sh` prints the commands of a copy of the tree):

- `install.sh` with `VF_INSTALL_BASE=file://DIR` (the archive and `SHA256SUMS`), `VF_PREFIX`,
  `VF_BIN` and `VF_INSTALL_FETCH=0`; then the release's `cuda-12.9.2/lib` given the CUDA 12.9.2
  libraries (`vf fetch`, or a link to `toolkits/cuda-12.9.2/lib`). REL below is
  `VF_PREFIX/volta-forever-VERSION-linux-x86_64`.
- `run.sh` with `VF_PROXY=REL/lib`, from a copy in which `$W/dist/forward` is `REL/forward` (and
  `W` the tree, when the copy is elsewhere).
- `torch.sh` with `VF_PROXY=REL/lib` and `VF_AUDIT_MAP_FILE` a map whose `libcublas.so.13` and
  `libcublasLt.so.13` are those of `REL/forward` (the 12 entries as in `tests/gpu/env.sh`).
- `launcher.sh` from a copy in which `vf` is `REL/bin/vf`; and `REL/bin/vf version`.
- For 0.1.5 also `vllm.sh`, with `VF_PROXY=REL/lib` and the map of `torch.sh`, from the copy.
