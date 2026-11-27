# volta-forever: installation

A compatibility layer that lets CUDA 13 programs (built for sm_75 and later, with no code for
Tesla V100) run on V100s with the 580 driver.

## Requirements

- **GPU**: NVIDIA Tesla V100 (compute capability 7.0)
- **Driver**: NVIDIA 580.178.04 (tested; the last branch with Volta support)
- **OS**: Linux x86_64, glibc 2.34 or later. The installation was run in fresh systems of
  Ubuntu 22.04, 24.04 and 26.04 and of Debian 12 and 13 (2026-11-21; no GPU in them: what was
  seen is that everything installs, that the binaries start, and that the downloads are
  made). An outside tester ran it with a V100 on Rocky Linux 9.9, Ubuntu 26.04, Debian 13 and
  Fedora 45 (`docs/reports/`); RHEL 9 and the other versions of Fedora from 36 have the C library
  it takes, and were not tried.
- **On a fresh Debian or Ubuntu**, what the commands below use and the system does not have
  yet: `curl` (for the installer), `python3` and `xz-utils` (for `vf fetch`), `python3-venv`
  (for `vf venv`). The installer and the launcher say which one is missing; they install
  nothing themselves.
- **CUDA 12.9.2 runtime libraries**: not included; `vf fetch` downloads them (below)

## 1. Install

With the installer:

```sh
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
```

`install.sh` is at the root of the repository; read it before you run it, or download it and
run it as a file. What it does, in order:

1. checks the machine: Linux, x86-64, glibc 2.34 or later, and the tools it uses (`tar`,
   `gzip`, `sha256sum`, `mktemp`, `flock`, `curl` or `wget`);
2. downloads the archive of its version and the file `SHA256SUMS` from the project's releases,
   over HTTPS only, redirections included, and stops if the archive's SHA-256 is not the one
   that the file gives;
3. unpacks the archive under `~/.local/share/volta-forever` (`$XDG_DATA_HOME/volta-forever`
   when that is set; `/opt/volta-forever` for root) and links `vf` and `volta-forever` in
   `~/.local/bin` (`/usr/local/bin` for root). The CUDA 12.9.2 libraries that an earlier
   installation downloaded are kept, and what an installer that was killed left there more
   than a day before is removed;
4. runs `vf fetch` (section 2: NVIDIA's cuBLAS, from NVIDIA's servers), when `python3` and `xz`
   are there; when one is not, it says which and leaves that step to you;
5. runs `vf doctor`, and says so if the directory of the commands is not on your `PATH`.

It installs no package of the system, asks for no password and writes nowhere else.

| Setting (environment) | |
|---|---|
| `VF_VERSION` | the version to install (default: the one the script was published with) |
| `VF_PREFIX`, `VF_BIN` | where versions are unpacked, and where the commands are linked |
| `VF_INSTALL_FETCH=0` | do not run `vf fetch` |
| `VF_INSTALL_BASE` | where the archive and `SHA256SUMS` are taken from (a mirror, a directory served locally): an `https://` URL, or a `file://` one; `VF_INSTALL_ALLOW_HTTP=1` takes an `http://` one too |

Reinstalling or switching releases keeps existing `vf venv` environments pinned to their installed launcher: with such an environment activated, `vf version` and `vf doctor` say when it runs another release than theirs, and `vf venv DIR` makes it run through the release of that `vf`. Releases share one `cuda-12.9.2` directory under `VF_PREFIX`. When upgrading an older installation, this path points to its existing CUDA directory so manual `vf fetch` commands share the same downloads. Keep that directory when removing individual old versions.

To remove it: delete the directory of `VF_PREFIX` and the two links.

By hand, from a release's archive:

```sh
tar -xzf volta-forever-VERSION-linux-x86_64.tar.gz      # or: tar --zstd -xf ...tar.zst
export PATH=$PWD/volta-forever-VERSION-linux-x86_64/bin:$PATH
```

`bin/vf` is the launcher: everything below goes through it. `bin/volta-forever` is a link to
it, the same program under its long name (in the fish shell, `vf` may already be the command
of VirtualFish). It finds the rest of the archive from its own place, so it may also be called
by its full path, or through a link from a directory of the `PATH`.

## 2. Download the CUDA 12.9.2 libraries

The layer does not redistribute NVIDIA's libraries. The launcher downloads the ones your
program needs from NVIDIA's public redistributable archives, checks them (SHA-256) and unpacks
them into `cuda-12.9.2/` of the archive's directory, where it looks for them afterwards:

```sh
vf fetch           # cuBLAS, cuBLASLt, nvJitLink: 1.0 GB to download, 0.95 GB on disk
vf fetch --only libnccl  # multi-GPU collectives: 337 MB to download, 421 MB on disk
vf fetch --all     # also cuFFT, cuSOLVER, cuSPARSE, cuRAND, NPP, nvJPEG, NVRTC and NCCL
```

NCCL comes from NVIDIA's `nvidia-nccl-cu12` wheel. VF verifies its SHA-256 and installs the shared library and its license into the same tree. [NVIDIA NCCL on PyPI](https://pypi.org/project/nvidia-nccl-cu12/2.32.6/)

It needs `python3` and GNU `tar` (and uses `aria2c` when there is one). It unpacks the shared
libraries and NVIDIA's licence, nothing else, and removes each archive once it is unpacked.
Each archive, and the NCCL library, is unpacked in a staging directory and moved into place: a
fetch that fails or is stopped (Ctrl-C, SIGTERM, SIGHUP) while it unpacks leaves the files that
were there before, and the next one removes what it left; once it has begun to move a component
into place, it finishes before it stops. While it moves one, or when it was killed (SIGKILL, a
power cut) or failed meanwhile, a journal (`.vf-publishing`) is in the directory: the launcher
refuses that directory (its files may be of two versions), `vf doctor` says so, and the next
fetch completes the move. It can be run again: what is unpacked already is not
downloaded again. `--only REGEX` chooses the
packages of NVIDIA's manifest by name, `--to DIR` another directory (then tell the launcher
where: `--cuda12 DIR/lib`, or `VF_CUDA12=DIR/lib`).

**cuBLAS alone, or all of them.** cuBLAS is the one that a program cannot keep from CUDA 13:
the translator refuses some kernels of its CUDA 13 build, and the operations that need them
fail. With cuBLAS alone (`vf fetch`), a program's other CUDA libraries are its own CUDA 13
builds, whose kernels the layer translates when the program loads them. With all of them
(`vf fetch --all`), each is served by its CUDA 12.9.2 build, which has code for Volta: nothing
of it is translated, and nothing of it waits for a translation at the first run. What was run
with which:

| Program | CUDA 12.9.2 libraries behind it |
|---|---|
| PyTorch (stock wheel, sm_75 machine code only) | cuBLAS alone: the 925 operations and the 32 models of `LIMITS.md` (the transforms through the wheel's own cuFFT), and factorisations of `torch.linalg` and a sparse product through the wheel's own cuSOLVER and cuSPARSE. All of them: the 925 operations and the 32 models again |
| llama.cpp (official CUDA 13 build) | cuBLAS |
| vLLM, SGLang (pip) | cuBLAS alone (environment set by hand); all of them (through the launcher) |
| ZML source example (bundled CUDA libraries opened by absolute path) | cuBLAS, cuFFT, cuSOLVER, NVRTC |

If you already have the CUDA 12.9 toolkit installed, the launcher finds it in
`/usr/local/cuda-12.9/lib64`; elsewhere, name its library directory with `--cuda12` or
`VF_CUDA12`. The launcher refuses a directory whose path has a `:` (the lists it passes on are
separated by colons): name it by a link without one. `vf doctor` says which directory it took
and why. Section 4 has the command that `fetch` runs, for those who would rather run it
themselves.

## 3. Run a program

Put `vf` in front of the command:

```sh
vf python3 -c "import torch; print(torch.cuda.get_device_name())"
vf llama-completion -m model.gguf -p "Hello" -n 48 -ngl 99
```

The launcher sets the environment for that command and runs it: nothing stays in the shell,
and the exit status is the command's. `vf doctor` says what it finds and what is missing (the
layer's files, the card, the driver, the C library, which CUDA 12.9.2 directory it took and
why, the library that each forwarding library calls, the CUDA 12 NCCL, and the
`LD_LIBRARY_PATH` and `LD_AUDIT` that a command would get), and ends with a status other than
0 when something required is missing (of the CUDA 12.9.2 libraries, cuBLAS is required; it
names the others that are not there in one line).

**Python: pip and virtual environments.** pip looks at neither the card nor the CUDA of the
system: install the ordinary wheels for CUDA 13 (`pip install torch --index-url
https://download.pytorch.org/whl/cu130`). They bring CUDA 13 libraries of their own, which the
layer replaces when the program runs. A virtual environment can be made to run through the
layer by itself:

```sh
vf venv .venv                 # python3 -m venv .venv, then every Python of it runs through the layer
.venv/bin/pip install ...     # no prefix: pip, python and the scripts that pip puts in .venv/bin
.venv/bin/python train.py
```

`vf venv DIR` makes the environment when DIR is not one yet (`--python PYTHON` names the
Python to make it with, and what follows `--` goes to `python -m venv`), or takes one that is
there (made by `python -m venv` or by `uv venv`: a directory with a `pyvenv.cfg`). When Python
cannot make the environment (on Debian and Ubuntu, without the package `python3-venv`), it
says what Python says and removes what the attempt left, so that the same command works once
the package is there. It puts a module in the environment's
`site-packages` that every Python of the environment imports when it starts, and that starts
Python again through the launcher when the layer's environment is not set yet: activated or
not, started by a service or by an editor. So a package that compiles GPU code when it is
installed, and asks the GPU what to compile for, is told what a program is told (7.5), and not
`sm_70`, which CUDA 13's compiler refuses. `VF_VENV=0` in front of a command starts that Python
as it is; `vf venv --off DIR` takes the module out again. It needs Python 3.10 or later, and
it is for Python: another program run from the environment's shell still wants `vf` in front.

For a whole shell session:

```sh
eval "$(vf env)"
```

(`env` prints `export` lines; applied twice, it does not repeat the directories.)

| | |
|---|---|
| `vf [OPTIONS] [--] COMMAND [ARGS...]` | run COMMAND through the layer (`run` may be written before the options; needed for a command named like one of the words below) |
| `vf env [OPTIONS]` | print the environment as `export` lines |
| `vf venv [--python PYTHON] DIR [-- OPTIONS]`, `vf venv --off DIR` | a virtual environment of Python that runs through the layer by itself |
| `vf fetch [--all \| --only REGEX] [--to DIR]` | download the CUDA 12.9.2 libraries |
| `vf ptxas ARGS...` | assemble file-backed PTX for the V100 using the CUDA 12.9 assembler |
| `vf doctor` | check the installation |
| `vf inspect`, `extract`, `adapt`, `translate` | the offline tool (`bin/vf-translate`; "Tools", below) |
| `vf version`, `help` | |
| `--cuda12 DIR` | the directory of the CUDA 12.9.2 libraries (else `VF_CUDA12`, then `cuda-12.9.2/lib` of the installation, then in a source tree `toolkits/cuda-12.9.2/lib`, then `/usr/local/cuda-12.9/lib64`); a path with a `:` is refused |
| `--map FILE` | substitutions of your own, which come before the launcher's (the format of section 4, B) |
| `--no-audit` | do not set `LD_AUDIT` (a program that finds all its libraries through the loader does without it); an audit library of the layer that `LD_AUDIT` already names (after `eval "$(vf env)"`) is taken out of it |

**What the launcher sets.**

- `LD_LIBRARY_PATH`: the layer's `lib` (the proxy `libcuda.so.1`, the nvJitLink adapter), its
  `forward` (the libraries with CUDA 13's names), the CUDA 12.9.2 directory, then what was there.
- `LD_AUDIT` = the audit library, then those that were in `LD_AUDIT` (an audit library of the
  layer is not named twice): they see the names that it chose, and have the last word. And
  `VF_AUDIT_MAP` = its table of substitutions: for each forwarding library whose CUDA 12.9.2
  library is there, its name to the forwarding library, and the name of the CUDA 12.9.2
  library that it calls to that library; `libnvJitLink.so.13` to the layer's nvJitLink
  adapter; `libnccl.so.2` to NCCL's CUDA 12 build when it was fetched. So a program that
  carries CUDA 13 libraries of its own and finds them by a recorded path (a PyTorch wheel) or
  opens them by absolute path gets the forwarding ones instead. Entries that you already have
  in `VF_AUDIT_MAP` stay first, and the launcher run under itself (`vf vf ...`, or after
  `eval "$(vf env)"`) does not add its own again.
- `VOLTA_FOREVER` (the version) and `VOLTA_FOREVER_HOME` (the archive's directory): how a
  script can tell that it runs through the launcher.
- For the command it runs (not in what `vf env` prints): `VF_CUDA12`, the CUDA 12.9.2
  directory it took, and, for a Bazel program, `RUNFILES_DIR` (below).
- With `VF_JIT_LINEINFO=1`, `NVCC_APPEND_FLAGS` gains `-lineinfo` (after what it already holds;
  not twice) and `CUTE_DSL_LINEINFO=1` is set unless it is set already. This is for a program told
  8.0 or later (`VF_CAPABILITY`) that compiles CUDA while it runs (PyTorch extensions, FlashInfer,
  TileLang through nvcc; NVIDIA's CuTe DSL, CUTLASS kernels written in Python, without it): such
  compilers target the capability they are told and keep machine code only, of which the layer can
  translate little when it is for sm_80 or later. With line information they also keep each
  cubin's PTX, which the layer compiles again for sm_70. The machine code is the same; the cubins
  grow by 60 to 120 %. NVRTC (PyTorch's jiterator, CuPy) reads neither. Caches keyed without the
  environment (PyTorch extensions, FlashInfer, TileLang, ccache) keep what they built before:
  clear them once. `vf doctor` suggests it when `VF_CAPABILITY` is 8.0 or later. It is not set by
  default because the launcher cannot see a compiler's command line: next to a `-G` there,
  `-lineinfo` makes nvcc warn twice, and fail under `-Werror all-warnings`. It is left out when
  `NVCC_APPEND_FLAGS` itself holds `-G`. `vf env` prints the two variables only when it sets them.
- With `VF_TRITON_TC` set (`1`, or a list of some of `f16`, `bf16`, `i8`), `PYTHONPATH` gains the
  installation's `python/` first (not a second time under itself). Unset, it is the release's
  default, one line of the launcher (`triton_tc_default=0` in `bin/vf`): off in this release; a
  default of `1` would be exported as `VF_TRITON_TC=1`. `VF_SAFE=1` turns it off whatever it is
  set to, and takes the copies of `python/` out of `PYTHONPATH`. Every Python that the command
  starts, and every one that these start, then runs the layer's `sitecustomize`, which installs an
  import hook on Triton and then runs the `sitecustomize` that it hides (the first other one on
  `sys.path`: a distribution's, an environment's, your own). Triton then compiles a kernel whose
  matrix products (`tl.dot`) all multiply two operands of these types for sm_80, where it would
  compile it for 7.5 with float32 `fma`; PyTorch and every other reader of the capability are
  still told 7.5. What it changes, kernel by kernel and type by type, and what it does not cover:
  `LIMITS.md`, "Triton's matrix products". A Python started with `-I`, `-E` or `-S` reads none
  of it, and its Triton compiles as without it. The Pythons of a `vf venv` environment get it as
  the rest, through the launcher. `vf env` prints the variables only when it sets them, and
  `vf doctor` says when it is on, or turned off by `VF_SAFE=1`. The compilation for sm_80 runs in
  a helper: a child of the process that compiles, started at its first kernel that needs one and
  reached through a socket pair that only it holds, about 105 MB (150 MB while it compiles), which
  ends with that process and after `VF_TRITON_TC_IDLE` seconds (10) with nothing to do; an error of
  Triton's compiler for sm_80 that ends it leaves that kernel to Triton's own compilation for 7.5,
  and the program goes on. Its temporary files are in a directory of its own in the temporary
  directory, as Triton's own. The user's helpers on one host are `VF_TRITON_TC_JOBS` at most among
  the processes that share Triton's cache directory (a lock file each in that directory as it is
  when a process compiles its first such kernel; processes whose Triton caches differ, as vLLM's
  engines, which give each model and each rank a cache of its own unless `TRITON_CACHE_DIR` is set,
  count their places apart; a best effort, which a process that cannot use the lock files does
  without), PyTorch Inductor's
  compile workers included: a helper that has had nothing to do for half a second gives its place up
  to a process that waits for one, and a process waits for a place `VF_TRITON_TC_WAIT` seconds once
  in its life. A helper is a child of its program: a program that waits for any child
  (`os.wait()`) may be given a helper (`LIMITS.md`, "Triton's matrix products").
  `pgrep -af 'volta_forever_triton_helper.py --helper'` lists the helpers that run.

A Bazel program that ships a CUDA toolkit in its runfiles can invoke an external assembler through an absolute path. When you start it with `vf`, the launcher prepares a private runfiles directory with the VF assembler and links to the program's other resources. It uses the local temporary directory when the user's cache cannot store private links. Install NVIDIA's `cuda-nvcc-12-9` package for this path. VF looks for `bin/ptxas` beside the selected CUDA 12.9 libraries and under `/usr/local/cuda-12.9`. Set `VF_PTXAS_REAL` to the absolute path of the CUDA 12.9 assembler to select another installation. For a file on disk, run `vf ptxas -arch=sm_80 kernel.ptx -o kernel.cubin` to produce code for sm_70. `vf ptxas` reads the options of `ptxas` 13.5 that take a value as `ptxas` does, ends with status 255 on the errors it finds itself, as `ptxas` does, and passes on the assembler's messages with the names of your files. The private runfiles directories are kept under `$XDG_CACHE_HOME/volta-forever/runfiles-v2` (`~/.cache/...`). The program and every process it starts hold a lock on its directory, which other machines that share the cache over NFS see too; when another directory is made, those that nothing holds are removed. None is removed on NFS mounted with local locks (`local_lock`, `nolock`) or on FUSE, where a lock is not seen by every machine. Releases before 0.1.3 kept theirs under `runfiles-v1`, without a lock: remove that directory once nothing that such a release started runs.

**When only some of the CUDA 12.9.2 libraries are there** (cuBLAS alone, after `vf fetch`
without `--all`), a forwarding library that has nothing to forward to is not used: it is left
out of the table, and the directory of the forwarding libraries is left off the library path.
The program then keeps its own library of that name (a wheel's cuFFT, cuSOLVER or cuSPARSE
for CUDA 13), and the layer translates its kernels like the rest of the program's code. That
is how the PyTorch tests of `LIMITS.md` are run when their environment is set by hand. With
`--no-audit` the forwarding libraries can only be found by their names: their directory stays
on the path, and the launcher says which of them have nothing behind them (a program that
calls one of those stops there).

When no CUDA 12.9.2 library is found at all the command still runs, with one line on standard
error (`VF_QUIET=1` leaves it out).

**The bf16 cuBLAS variants.** `VF_CUBLAS_BF16_TC=1` or `VF_CUBLAS_BF16_SKINNY=1` selects the
bf16 cuBLAS variants of the forwarding libraries: `libcublas.so.13` and `libcublasLt.so.13` in
`forward/bf16tc`, with their runtime library `libvf_cublas_bf16.so`. The launcher then gives
both names to the variants, never one alone, in its table and on the library path
(`forward/bf16tc` before `forward`), and passes both variables on as they are (`vf env` prints
them). It refuses to run, with a line that names what is missing, when the variants are not all
installed or the CUDA 12.9.2 cuBLAS that they forward to is not found; when an entry before its
own in `VF_AUDIT_MAP` gives cuBLAS another file (as `eval "$(vf env)"` without the switch leaves
there); and for a value other than `0` or `1`. `vf doctor` says whether they are installed and,
with a switch set, which libraries a command would get.

Both are off by default. What they change, for bf16 products with fp32 accumulation
(`cublasGemmEx`, `cublasGemmEx_64` and `cublasGemmStridedBatchedEx[_64]` of one matrix, and
`cublasLtMatmul` of one column-major matrix with the default or bias epilogue):

- `VF_CUBLAS_BF16_SKINNY=1`: products of 2 to 16 columns (fewer than 32 when small) where cuBLAS
  12.9 runs its sequential kernel are computed by one kernel of the layer, with 12.9's bits
  (`docs/emulations.md`, section 11, says where exactly, and how that was established).
- `VF_CUBLAS_BF16_VERIFY=1` (verify mode, a diagnostic, with a path on): every call the skinny path
  serves is also made by cuBLAS 12.9 itself, through the same entry point, handle, descriptors,
  algorithm and workspace, on the same stream, with its output in a scratch buffer that has the
  output's address modulo 256 and its leading dimension (and a copy of C made first when the call reads
  C), and the two outputs are compared bit for bit on the GPU, eager or captured in a CUDA graph (the
  copy, 12.9's call and the comparison are then nodes of the graph, which allocates nothing); with the
  tensor-core path on, its fallback's elements too. No result changes. At exit the report says how many
  calls were verified (each replay of a graph counts) and how many differed, in how many elements, and
  says each call signature that differed once, with what it takes to make the call again (entry point,
  operations, sizes, leading dimensions, types, addresses modulo 256, scalars and their pointer mode,
  algorithm, epilogue, workspace) and its first differing element. It costs a 12.9 call and a
  comparison per verified call, whose signature's text is made once, at its first call; per CUDA context,
  40 KiB of counters and a non-blocking stream of the runtime's own; per stream (per thread on the
  per-thread default stream), a scratch buffer of 1 MiB or more, kept to the end of the process. Calls of
  several threads on one stream are verified one after the other (`LIMITS.md`, "Verify mode").
- `VF_SAFE=1` (safe mode: the proxy's table of "Environment variables", below, says what it is and
  how it is read): the variants' tensor-core path serves nothing, whatever `VF_CUBLAS_BF16_TC` says;
  the skinny path, whose results are 12.9's bits, is not affected. Every result is then cuBLAS 12.9's
  bits, eager or captured, at any batch size (12.9's own results depend on n at n = 1 and, on some
  shapes, up to n = 16: `LIMITS.md`). The variants read it at their first intercepted call (a
  handle's creation or a product): set before a program's first cuBLAS call, it is honoured; set
  after, it is not.
- `VF_CUBLAS_BF16_TC=1`: larger products in the TN layout (a linear layer's, in inference and in a training
  step's forward pass; the backward pass's products go to 12.9) with k from 768 to 16384 run on the tensor
  cores, but for the classes of shapes that the rules of speed leave to 12.9 (`LIMITS.md`: TN with m >= 8192 and
  n <= 64, with m >= 6144 and k > 9216, and with k > 8192 and n > 256: classes whose median speed was below 1.10
  times 12.9's or whose slowest product ran below 0.90 times on real operands, or with no layer of a real model
  measured, the path faster at some of their shapes), with
  exact scaling to fp16 and fixed slices of k, and differ from 12.9's in the last bits (against the exact product,
  in units of 2^-24 sum |a b|: on real operands, the linear layers of k = 768 and 896 of five models, single
  elements within 36 units, as are 12.9's own, and a mean error towards zero of 0.82 units at most; sums of
  products of one sign come out low, on synthetic data by about 13 units on average with operands uniform in
  (0, 1], and with operands spread over many binades by 36 to 40 units (the acceptance test's log-normal
  magnitudes) and 49 to 70 units (its adversarial positive construction) at every k from 768 to 16384, single
  elements reaching 90 and 216 units: `LIMITS.md`, "The tensor-core path takes k >= 768 only"), except when they
  are captured in a CUDA graph; the skinny path also serves its products, unless `VF_CUBLAS_BF16_SKINNY=0`. One
  element's bits then depend on the batch size and on capture (`LIMITS.md`), and a model whose layers have fewer
  than 768 in-features gets the tensor cores only on its wider layers. A training step's forward products are
  served too, its backward pass's go to 12.9; a product whose input has many columns failing the exact conversion
  can still run below 12.9's speed (`LIMITS.md`, "The bf16 cuBLAS variants"; `docs/performance.md`, "bf16 products
  through cuBLAS: the opt-in variants on real programs").

Every other call goes to cuBLAS 12.9 as it is: one column, batches, the other entry points,
explicit algorithms (`CUBLAS_GEMM_ALGO*`), pedantic modes, k above 16384 or below 768 for the tensor cores,
products in another layout than TN and the TN shapes of the rules above,
tensor-core calls captured in a CUDA graph (`LIMITS.md`, "The bf16 cuBLAS variants", says why), and
calls whose workspace cannot be allocated. An eager tensor-core call (TN 1536 x 480 x 1536) cost
16.5 microseconds of host time against
6.5 for 12.9's, a skinny call (TN 1536 x 8 x 1536) 2.4 against 5.8 (`tests/gpu/forward/bf16-cost.c`,
mean over 100 calls).
From `VF_LOG_LEVEL=2` the variants' report at exit gives, for each entry point, its calls, the path that
served them and why the others were forwarded, then what the tensor-core path's fallback did: how many of
the path's calls had rows or columns that failed the exact conversion to fp16, how many of them more than
a quarter of their rows or of their columns, and the share of the calls' elements that the fallback
computed (`LIMITS.md`, "Speed that depends on the data"). With `VF_SUMMARY=FILE`, the lines of the entry points
are also in `FILE.PID` while the process runs, at every log level, for a process that is killed without exiting;
what the fallback did is counted on the GPU and read at exit only.
The layer's workspace is not the program's: a framework's allocator does not see it. The variables do nothing without the
launcher: by hand, put `forward/bf16tc` before `forward` and set the variable.

**A program that wants to know.** A program built for later GPUs is told the compute
capability that its code starts at (7.5), so that it loads that code: the driver's answers do
not tell it that the card is a Volta with the layer in front of it. A program that has kernels
written for Volta beside its code for later GPUs can ask, and prefer them: `libcuda.so.1` has
the functions `volta_forever_version()` (the version, a C string) and `volta_forever_abi()`
(the number of this interface, 1) only when it is the layer's.
`dlsym(RTLD_DEFAULT, "volta_forever_version")` gives the first or nothing; in Python,
`ctypes.CDLL("libcuda.so.1").volta_forever_version` or an `AttributeError`.

## 4. By hand

What the launcher does can be done without it. The CUDA 12.9.2 libraries:

```sh
# Needs: python3 and tar. Uses aria2c when available (faster); falls back to Python's urllib.
python3 scripts/fetch-redist.py \
    https://developer.download.nvidia.com/compute/cuda/redist \
    https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.2.json \
    dl-cuda-12.9.2 \
    cuda-12.9.2 \
    --only '^(libcublas|libcufft|libcusolver|libcusparse|libcurand|libnpp|libnvjpeg|cuda_nvrtc|libnvjitlink)$'
```

The manifest argument can be a local JSON file or an https:// URL (an http:// one is refused
unless `--allow-http` is given: the manifest holds the SHA-256 that the archives are checked
against). The script downloads and verifies (SHA-256) the archives listed in it, then unpacks
them into `cuda-12.9.2/`, headers and static libraries included, and keeps the archives (6.8 GB
unpacked for these nine). `vf fetch` adds `--runtime` (the shared libraries and the licence
alone), `--discard` (each archive removed once unpacked) and `--with-nccl` (NVIDIA's CUDA 12
NCCL wheel among the packages that `--only` can choose, as `libnccl`).

Then one of three ways to run a program.

### A. Library path (works when the program finds its libraries through the loader)

```sh
S=/path/to/volta-forever
T=/path/to/cuda-12.9.2/lib

LD_LIBRARY_PATH=$S/lib:$S/forward:$T:$LD_LIBRARY_PATH \
    some-cuda-13-program
```

The program finds the proxy (`libcuda.so.1`) and the forwarding libraries (the CUDA 13 names)
in `$S/lib` and `$S/forward`, and the forwarding libraries find the CUDA 12.9.2 originals in
`$T`.

**PyTorch needs the audit library (way B below), not way A alone.** A stock PyTorch wheel
finds NVIDIA's libraries by a path recorded in its own (RPATH), not by `LD_LIBRARY_PATH`. With
way A alone, the wheel's own cuBLAS 13 is used instead of the forwarding one; its sm_75 kernels
are translated, but some of them are refused by the translator, and the operations that need
them fail.

### B. Audit library with a map (for PyTorch and programs that open libraries by absolute path)

The audit library (`libvf_audit.so`) intercepts the loader's search and substitutes
libraries by file name. The substitution table is read from `VF_AUDIT_MAP` (environment,
colon-separated `name=path` pairs; highest priority) and from a map file named by
`VF_AUDIT_MAP_FILE` or `vf-audit.map` next to the audit library. The first matching entry
wins, and environment entries come before those of the file. An empty replacement makes the
library not found. Relative paths in the file are resolved relative to the file's directory.
(`crates/vf-audit/src/lib.rs`)

**Example: a stock PyTorch wheel (the wheel carries its own cuBLAS)**

Write a map file `torch-audit.map` (the actual map used by `tests/gpu/torch.sh`):

```
libcublas.so.13=/path/to/volta-forever/forward/libcublas.so.13
libcublasLt.so.13=/path/to/volta-forever/forward/libcublasLt.so.13
libcublas.so.12=/path/to/cuda-12.9.2/lib/libcublas.so.12
libcublasLt.so.12=/path/to/cuda-12.9.2/lib/libcublasLt.so.12
```

The proxy (`libcuda.so.1`) is found by the library path, not through the map.

Run:

```sh
S=/path/to/volta-forever
T=/path/to/cuda-12.9.2/lib

LD_AUDIT=$S/lib/libvf_audit.so \
VF_AUDIT_MAP_FILE=torch-audit.map \
LD_LIBRARY_PATH=$S/lib:$T:$LD_LIBRARY_PATH \
    python3 -c "import torch; print(torch.cuda.get_device_name())"
```

**Example from the historical ZML source test**

The default `tests/gpu/zml.sh` run keeps the build’s bundled cuDNN 9.24.0 and maps the following CUDA libraries.

```
libcuda.so.1=/path/to/volta-forever/lib/libcuda.so.1
libnvJitLink.so.13=/path/to/volta-forever/lib/libnvJitLink.so.13
libnvJitLink.so.12=/path/to/cuda-12.9.2/lib/libnvJitLink.so.12
libcublas.so.13=/path/to/volta-forever/forward/libcublas.so.13
libcublasLt.so.13=/path/to/volta-forever/forward/libcublasLt.so.13
libcufft.so.12=/path/to/volta-forever/forward/libcufft.so.12
libcusolver.so.12=/path/to/volta-forever/forward/libcusolver.so.12
libnvrtc.so.13=/path/to/volta-forever/forward/libnvrtc.so.13
```

A separate historical comparison set `VF_ZML_CUDNN=9.10.2` (a setting of `tests/gpu/zml.sh`) and made that older library report 9.24.0 with `@cudnn-version`. Use the cuDNN build the application expects. The verified 35B LLMD server runs through `vf ./llmd` with its matching runfiles and no manual audit map. [LLMD launch command](docs/guides/zml.md#5-benchmarks)

### C. Forwarding libraries on the library path

Place the forwarding libraries and the CUDA 12.9.2 originals on the library path so the
program finds them before the CUDA 13 ones it carries:

```sh
LD_LIBRARY_PATH=/path/to/volta-forever/lib:/path/to/volta-forever/forward:/path/to/cuda-12.9.2/lib:$LD_LIBRARY_PATH \
    some-cuda-13-program
```

### Triton's products, by hand

What `VF_TRITON_TC` does through the launcher ("What the launcher sets") is done without it by
putting the installation's `python/` first in `PYTHONPATH` and `VF_TRITON_TC=1` in the
environment of the program, beside the layer's libraries of A, B or C:

```sh
PYTHONPATH=$S/python${PYTHONPATH:+:$PYTHONPATH} VF_TRITON_TC=1 \
LD_AUDIT=$S/lib/libvf_audit.so VF_AUDIT_MAP_FILE=torch-audit.map \
LD_LIBRARY_PATH=$S/lib:$T:$LD_LIBRARY_PATH \
    python3 program.py
```

The module finds the tool `bin/vf-translate` of the installation from its own place.

### Programs that were run this way

On one Tesla V100, with no switch set (`LIMITS.md` has the counts and what does not work):

| Program | How | What to know |
|---|---|---|
| PyTorch 2.14 (wheel for CUDA 13) | B, with the map for its cuBLAS | 924 of 925 operations tested give the CPU's results (the other one the CPU does not do), 32 of 32 small whole models |
| llama.cpp (official CUDA 13 build) | A or C | flash attention on or off; quantized and NVFP4 weights |
| Historical ZML source test | B, with the map above | the test opens bundled CUDA libraries by absolute path |
| vLLM 0.34 (installed with pip) | B, as PyTorch | the engine and its HTTP server |
| SGLang 0.5 (installed with pip) | B, as PyTorch | the engine and its HTTP server, with its default attention (FlashInfer) or with `--attention-backend triton`. Five of FlashInfer's eighteen prefill kernels were refused when they were last counted (`LIMITS.md`): if a model or a batch asks for one of them, the error names it, and `--attention-backend triton` goes without them |

A kernel that the layer cannot run is refused when the program asks for it (the program gets
error 218 and a NULL handle, and on standard error a line names the kernel and says why:
`LIMITS.md`, "Kernels refused"); it is never run wrong knowingly. A kernel whose code in its image
is a trap and nothing else (how a kernel built for other GPUs only is emitted for the rest) is
given when the program asks for it, and its launch returns 218 with a line that names it. The first run of a program translates its machine code and lets the driver compile
what it compiles (`What the first run costs`, below). Under `CUDA_MODULE_LOADING=EAGER` the CUDA
runtimes ask for every kernel at their start: the CUDA 12.9 libraries start as without the
layer, a program built with the CUDA 12 runtime starts without the kernels of a fat binary of its
own in which one is refused, and all the code is translated and compiled at the start, which
takes many minutes the first time for PyTorch (`LIMITS.md`, "Kernels refused").

## 5. Verify that it works

`vf doctor`, then a PyTorch test that exercises the proxy and cuBLAS forwarding
through the audit library (needs a stock PyTorch wheel for CUDA 13, e.g. `torch==2.14.0` from
`download.pytorch.org/whl/cu130`):

```sh
vf python3 -c "
import torch
print('device:', torch.cuda.get_device_name())
a = torch.randn(256, 256, device='cuda')
b = torch.randn(256, 256, device='cuda')
c = torch.mm(a, b)
print('matmul shape:', c.shape)
print('sum:', c.sum().item())
print('ok')
"
```

Expected output:

```
device: Tesla V100-SXM2-32GB
matmul shape: torch.Size([256, 256])
sum: <a number>
ok
```

## Environment variables

Every variable that the layer, the launcher and the installer read is here (the installer's are
in section 1). The ones of the test scripts are in `tests/README.md`; `docs/architecture.md`,
"Switches and diagnostics", groups these by purpose.

**Proxy (`libcuda.so.1`):**

| Variable | Default | Description |
|---|---|---|
| `VF_LOG_LEVEL` | `1` | 0: fatal only; 1: kernels refused, emulated, images adapted; 2: what was adapted, each image named by the file it was loaded from, or by its size and XXH64 digest (as `xxhsum -H1` gives it for a file of the same bytes) for an image loaded from memory, taken only for a line that prints it, and how many of its kernels have a body that is a trap and nothing else, and how many an empty one; 3: everything |
| `VF_LOG` | stderr | Log to a file instead of standard error (opened for appending, made if it is not there; a link at that path is followed, since a path the user names may be one on purpose, `/dev/stderr` for one) |
| `VF_EMULATION_WARNINGS` | `20` | How many kernels with emulated instructions are listed, and how many kernels refused under a name said before are said again; 0 for none (the first kernel refused under each name is always said). A kernel with emulated instructions is its name and its code, a refused kernel its name and why it is refused: a second kernel of one name (another specialization), of other code or refused for another reason, is said too, with its image's file, or from `VF_LOG_LEVEL=2` its XXH64 digest as the line of its load gave it. What is kept of the kernels said is bounded: past 4,096 kernels (thousands of specializations, for one), the next are neither said nor counted, and one line says so |
| `VF_SAFE` | unset | `1`: safe mode, for the whole layer (`libcuda.so.1`, the nvJitLink adapter, `vf ptxas`, `vf adapt`, the bf16 cuBLAS variants; the launcher passes it on). What the layer does by choice and whose results are not bit-identical to the conservative reference is not used, whatever its own variable says. The reference: for a call relayed to an NVIDIA library, that library's own bits (cuBLAS 12.9); for code that the layer lowers or translates, the program's own instructions as the layer's default lowering computes them, with each tensor-core product of translated machine code computed by itself. Turned off: `VF_BF16_TC_GROUPS=1`, `VF_BF16_TC_TILES=1`, `VF_BF16_COLUMN=tree` and `VF_BF16_ROW=tree` (bf16 products on Volta's fp16 tensor cores, or summed in another order), `VF_TRANSLATE=first` and `force` (another form of the program than the one the default takes), `VF_COMPANIONS=1` (opt-in companion hoisting and paired HMMA scheduling), the tensor-core path of the bf16 cuBLAS variants (`VF_CUBLAS_BF16_TC=1`), for which the default's setting is used instead, and `VF_TRITON_TC` of the launcher (set, or by its default), with which Triton's float16 products would be summed by Volta's tensor cores: Triton compiles every kernel for 7.5, as without the switch; and the pairs of fp16 tensor-core products in translated sm_75 machine code, which are on by default (`VF_PAIRS`; said as `VF_PAIRS in machine code`): a pair adds the halves of k where a product by itself adds the even k, then the odd, and whether a kernel gets pairs depends on its registers and on the block it requires, so each product is then translated by itself, as with `VF_PAIRS=0` in machine code (what that costs: `LIMITS.md`, "Safe mode"). What it does not do: it does not change the capability that programs are told (`VF_CAPABILITY`, `VF_CAPABILITY_ALL`: the code a program chooses for what it is told is its own); it does not turn off what gives the reference's bits (the `ordered` modes of `VF_BF16_COLUMN` and `VF_BF16_ROW`, the skinny path of the bf16 cuBLAS variants, the pairs of the PTX lowering); it leaves `VF_PAIRS=0`, `VF_BORROW` and `VF_FIT` as they are (with each product by itself, the translations they choose give the same bits), and what refuses rather than computes otherwise or keeps the arithmetic (`VF_TRANSLATE=0`, `VF_ALIGNED_ROWS=0`, `VF_LAZY=0`, `VF_JIT_LINEINFO=1`, and the reported traps of `VF_REPORTED_TRAPS`, which stop a kernel where a refused one would not have been given); and it cannot make a model's outputs independent of its batch size or of the GPU, which the reference is not either (`LIMITS.md`, "Safe mode"). It does not turn off the translation of sm_80 machine code, which is not independently validated (an image with nothing but sm_80 code, or a kernel that only the sm_80 code of its image has: `VF_TOLD_CODE=0` is the switch for the second): such code has no other form on this GPU. Only the exact value `1` turns it on: unset, `0`, an empty value, `true`, ` 1` and every other value leave it off, and a value other than `0` is said once. Read once in a process (by `libcuda.so.1` when it is loaded, by the nvJitLink adapter at the first input it adapts, by the bf16 cuBLAS variants at their first intercepted call, by the layer's Python module at the start of a Python and at the import of Triton), and said once at level 1 with what it turned off. The proxy's cache keeps what is made with it and without it apart. Other software on a V100 may honour the same variable |
| `VF_PROXY_CACHE` | `$XDG_CACHE_HOME/volta-forever/proxy`, else `~/.cache/volta-forever/proxy` | Directory for the proxy's cache (translated machine code, adapted PTX); `0` or empty to disable. Must be a directory of the user that no one else can write (group- or world-writable directories are refused, and the cache is then off); the layer makes it with mode 0700. One made otherwise under a umask of 002, the default of many distributions' accounts, can be written by its group: the layer then says `proxy cache disabled` once, every start adapts its images again, and `vf doctor` says so with the fix (`chmod 700 DIR`). The cache can grow to 2 GB (`VF_PROXY_CACHE_MAX`). On a system whose home directory is on a RAM filesystem, this is memory. A write that fails leaves no temporary file, and temporary files left more than ten minutes before are removed at the first use of the cache in a process. |
| `VF_PROXY_CACHE_MAX` | `2147483648` (2 GB) | Maximum size of the cache in bytes |
| `VF_CAPABILITY` | auto | Compute capability reported to callers without Volta code, as `<major>.<minor>` (default: the lowest the program has code for, which is 7.5 for programs with only sm_75 machine code). `VF_CAPABILITY=8.0` reports Ampere and prefers compatible PTX for the highest target up to 8.0, except to a caller whose code, or the code of the libraries that use it (for a CUDA runtime), runs from sm_75 machine code, such as PyTorch's, or from PTX for sm_75 only: that caller is still told 7.5, and the images it loads are adapted for 7.5, with a line at the default log level. Triton compiles for what PyTorch is told whenever PyTorch is installed, so this alone leaves its `tl.dot` without tensor cores (and `VF_TRITON_TC` is for the default capability alone). A value that is not `<major>.<minor>` (`8`, `80`, `sm_80`) or that is below 7.0 is not used, and the layer says so. What it does not cover: `LIMITS.md`, "The capability told with `VF_CAPABILITY`"; how images are chosen: `docs/architecture.md`, section 3. |
| `VF_CAPABILITY_ALL` | `0` | `1`: report `VF_CAPABILITY` also to a CUDA runtime, and to the code that uses it, otherwise still told 7.5 for the code of that program. The kernels that the program then chooses for the reported capability, and whose sm_75 code is a stub that writes nothing (PyTorch's attention), are refused: only for a program that does not launch them, as SGLang with the settings of its guide, which selects its quantized experts from PyTorch's capability. It also gives Triton's `tl.dot` the tensor cores in a program with PyTorch, with PyTorch's hazards: told 7.5, Triton computes it with `fma`, 3 to 150 times slower in the kernels measured; `VF_TRITON_TC` (launcher, below) gives it them at the default capability, with everyone still told 7.5 (`LIMITS.md`, "Speed"). A library that asks for itself without that runtime (cuDNN, cuFFT, cuSOLVER) is still told what its own code allows. |
| `VF_TRANSLATE` | `1` | `0`: disable machine-code translation; `first`: translate even where PTX exists; `force`: translate even where sm_70 code exists |
| `VF_PAIRS` | `1` | `0`: each tensor-core product is computed by itself, in machine code and in PTX; by default two products that share their A are done together on whole rows where the program loads them that way (`docs/pairs.md`; in PTX, section 13: the same results as each product by itself). `VF_SAFE=1` turns off those of machine code, whose last bits differ from each product by itself |
| `VF_COMPANIONS` | `0` | `1`: opt-in companion register hoisting for loop-invariant whole-row matrix load addresses and per-group HMMA scheduling (disjoint scoreboards in `0..3` per interfering group, `usched=18` and `.reuse` on `STEP0..2` of paired `HMMA.884` steps: `docs/pairs.md`, sections 8 and 12). Off by default (`0` or unset leaves the output identical to `main`). Unused when `VF_PAIRS=0` or `VF_SAFE=1` |
| `VF_BORROW` | `7` | Where the translation of machine code takes the registers it needs. `7`: above the program's own wherever that translates, and for a kernel that would otherwise be refused for want of registers, among the program's registers that hold nothing, then in the thread's local memory, kernels with tensor-core products included. `6`: the same, such kernels with products refused (the default until 2026-11-20). `0`: above the program's only (kernels that name nearly every register are refused). `9`: try to preserve the source kernel’s maximum block size when translating machine code for CUDA graphs. [Graph limits](LIMITS.md#cuda-graphs). Other values are for experiments (`docs/translator.md`, "Registers") |
| `VF_FIT` | `1` | `0`: a launch that the driver refuses for its registers (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`) is refused to the program. By default the kernel is then launched from a second translation of its module in which no kernel takes more registers than it declared (slower code; made once, at the first refused launch). Not for a kernel that can reach data of its module (`docs/architecture.md`, "Launch") |
| `VF_TOLD_CODE` | `1` | `0`: a kernel that a program told 8.0 asks for, and that only the sm_80 machine code of its image has (the image is given from its sm_75 code), is not given: the lookup fails as the driver answers (`CUDA_ERROR_NOT_FOUND`, "named symbol not found"), and a line says which code has the kernel. By default the kernel is given from the sm_80 code, translated, in a module of its own, when it can reach no data of its image, with a line for each (`LIMITS.md`, "The capability told with `VF_CAPABILITY`"). `VF_SAFE=1` leaves it on: such a kernel has no other code on this GPU |
| `VF_FIT_SHARED` | `0` | Set to `1` to retry a rejected dynamic shared-memory opt-in on a V100 at the physical maximum available to that kernel. Each launch must still fit its static and dynamic allocation within the device limit. |
| `VF_BF16_TC_GROUPS` | `0` | Set to `1` to run groups of BF16 matrix products on Volta's FP16 tensor cores. A warp takes that path when every multiplicand of the group, in all 32 lanes, converts exactly to a normal FP16 or a signed zero; otherwise the group keeps the scalar path. Volta's tensor cores truncate when they add: results can differ from the scalar path in the last bits, and long sums of terms of one sign come out slightly low. Not bit-reproducible from run to run when a program leaves unused operands of a product uninitialised (stale shared memory that a kernel masks out, as FlashAttention-2 does): the path is chosen by one vote over every multiplicand of the group, including operands whose products are discarded, so a result can change, by rounding, with them (`docs/emulations.md`, section 4). The default path gives the same bits. This can increase first-use compilation time. |
| `VF_BF16_TC_TILES` | `0` | Set to `1` to run two-by-two tiles of BF16 matrix products on Volta's FP16 tensor cores. A warp takes that path when every multiplicand of the tile converts exactly to a normal FP16 or a signed zero; other inputs keep the scalar path. The same truncation and the same caveat as `VF_BF16_TC_GROUPS`: not bit-reproducible from run to run when a program leaves unused operands of a product uninitialised, since one vote over every multiplicand of the tile chooses the path. |
| `VF_BF16_COLUMN` | off | For four successive BF16 matrix products sharing an accumulator, use the single-column path when only column 0 of B is active and the warp passes the operand checks. `ordered` preserves the scalar FP32 FMA order. `tree` sums four partial results and can round differently, especially when large terms cancel; and since a vote over the other columns and over every operand chooses the path, a result of column 0 can change, by rounding, with them: not bit-reproducible from run to run when a program leaves them uninitialised. `ordered` gives the same bits either way, as the default does. |
| `VF_BF16_ROW` | off | For pairs of BF16 matrix products sharing a left fragment, test whether only its first row contributes. `ordered` preserves the scalar FP32 FMA order. `tree` sums four partial dots before adding the accumulator and can round differently; and since a vote over the other rows and over every operand chooses the path, a result of the first row can change, by rounding, with them: not bit-reproducible from run to run when a program leaves them uninitialised. `ordered` gives the same bits either way, as the default does. Other inputs take the ordinary BF16 path. |
| `VF_ALIGNED_ROWS` | `1` | `0`: do not take it that the program keeps the rows that `ldmatrix` reads at multiples of 16 bytes, as the PTX manual requires: test the addresses when the code runs. For a program that stops with a misaligned-address fault (error 716); emulated tensor-core kernels are slower with it |
| `VF_LAZY` | `1` | `0`: let the driver compile a whole module when it is loaded, not each kernel when the program asks for it (a much longer first run) |
| `VF_REPORTED_TRAPS` | `1` | `0` (exactly; any other value leaves it on, as with `VF_LAZY`): a kernel that holds, or may reach, an instruction of the `multimem` family is refused, as before 0.1.6. By default, in a process where no GPU can take part in multicast (the driver says `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` is 0 for each: every V100; asked once in the process), each such instruction is replaced by a reported trap and the kernel is given: the PTX manual requires the multicast address that these instructions take and leaves them undefined without it, and no GPU of such a process can have one. Reached all the same, the trap stops the kernel with an assertion that names the instruction, the function, the block and the thread (`CUDA_ERROR_ASSERT`, 710), and the context ends. The kernels given so are said in one line for each image at level 1 (each kernel from level 2), counted at exit, and named in the report of a fault. The driver's answer is said from level 2; at level 1 only where it refuses a kernel (a GPU of the process that can take part in multicast, or no answer): once, after the first such refusal. Before the driver compiles such an image of 16 MiB of PTX or more, a line says so, once in the process (the first compilation can take many minutes). `VF_SAFE=1` leaves it on: safe mode turns off what computes results not bit-identical to the reference's, and a reported trap computes nothing otherwise (every path that does not reach it runs the program's own instructions, lowered as without it; one that reaches it stops, where a refusal would not have given the kernel), while turning it off would refuse NCCL's collectives under safe mode, which then could not tell whether a difference of a run on several GPUs comes from the layer. Not in `vf ptxas` and the nvJitLink adapter, which refuse as before (`LIMITS.md`, "Several GPUs and NCCL") |
| `VF_THREADS` | auto (`<= 8`) | Worker threads for PTX lowering and dependency analysis, and for the translation of machine code (sm_75 and sm_80 cubins) (default: the CPUs the process may run on, at most 8). Started only for large modules (PTX: 128 KiB or more; machine code: 64 KiB of code or more with two functions or more), per loading thread. Explicit values are capped at the CPUs the process may run on; `0` or `1` runs single-threaded on the calling thread; invalid values use the default; if a thread cannot be created, remaining work runs on the calling thread. The output is the same bytes at every number of threads |
| `VF_ADAPT` | `1` | `0`: give the driver every image as it is (to see what a program does without the layer's rewriting) |
| `VF_REAL_LIBCUDA` | auto | Path to NVIDIA's real `libcuda.so.1` when it cannot be found by itself (by default: the library of the loaded kernel module's version in the system's library directories) |
| `VF_STATS` | `0` | `1`: route every driver function through the layer and print call counts, the most launched kernels, and how often the refusal of a kernel at its use asked the driver what a handle is and refused a call, at exit |
| `VF_PROFILE` | off | `FILE`: write a per-kernel profile to `FILE.PID` at exit. Every launch is then synchronised on its stream: a profile is not a benchmark. The first synchronisation that returns an error (a kernel that faulted ends the context) is said at once, with the kernel it followed and its grid, block and shared-memory sizes; the report of a fault (`LIMITS.md`, "After a kernel faults") is made at that wait too, and each wait that succeeded rules out what it waited for. Each kernel keeps the counts of its first 64 distinct grid, block and shared-memory sizes. Only the process that loaded the layer writes it, and the `VF_STATS` report: a child made by `fork()` (a daemon, a pre-fork worker) writes neither, even when it does all the CUDA work. The file is replaced whole, as `VF_SUMMARY`'s is: written to a new file beside it, never through a link at that name, then renamed over it, its owner's only (mode 0600) |
| `VF_SUMMARY` | off | `FILE`: the report at exit (a fault, with the launches that may be what failed, `LIMITS.md`, "After a kernel faults"; the kernels with emulated instructions given to the program, the second translations, the counts of `VF_STATS`) is also written to `FILE.PID`, and rewritten while the process runs, every `VF_SUMMARY_INTERVAL` seconds when it changed: a process killed without exiting (some servers end their workers with SIGKILL) leaves the report as it was at most that long before. The file is replaced whole: written to a new file beside it, under a name with a random part, never through a link that is there, then renamed over it (a link at `FILE.PID` is replaced, not followed); it is readable by its owner only (mode 0600), and its first line says whether the process had exited. One thread writes it, which blocks every signal but those of a fault; none is started without the variable. Only the process that loaded the layer writes it, as the other reports at exit. Another library can put its own lines in it, after the layer's, each with the library's name in brackets at the start of the line, where a reader must match it (`[NAME] `; its last report, kept when it was removed, `[NAME, last report] `), through two functions of the layer's (`docs/architecture.md`, "The summary's lines of other libraries"); at exit the file waits 2 s at most for those libraries, and says so when their lines are not in it. The bf16 cuBLAS variants do: their exit report's line of each entry point, at every log level, as it is printed at exit from `VF_LOG_LEVEL=2` (`[libvf_cublas_bf16.so] `); the counts of the tensor-core path's fallback, kept on the GPU, are read at exit only, and the line about them says so while the process runs |
| `VF_SUMMARY_INTERVAL` | `30` | Seconds between two writes of `VF_SUMMARY`'s file while the process runs (at least 1) |
| `VF_TIMING` | `0` | `1`: print the time of each stage for every module and kernel |
| `VF_TRACE_LAUNCH` | `0` | `1`: route kernel launches through the layer and log those that the driver refuses (kernel, grid, block, shared memory, launch attributes) |
| `VF_FAULT_TRACE` | unset | `1` (exactly): after every launch that passes through the layer, wait for its stream, so that the launch whose synchronisation meets a fault (an illegal address, a misaligned access, a `trap`: errors 700, 716, 719 and the others of their kind) is named, and returns the fault's error: alone when no other launch of the context is uncovered (a launch made while a capture was under way, which is not waited for, is named with it), and with what it does not rule out (a copy, a memset or a host function before it on its stream, which the layer does not see). Without it the layer still says, once, at every log level, which kernels may have faulted: those launched in the context since its last synchronisation that succeeded, with what it made of each (`LIMITS.md`, "After a kernel faults"), at the first call that it routes and that returns the error (a fault that only calls it does not route met, a copy or a memset, is not said at all when no later call that it routes returns the error: it asks the driver nothing at exit; the report at exit lists the faults that were said); keeping them costs a launch about 25 ns (an empty kernel launched in a loop on a V100: 1.86 µs a launch instead of 1.84, `docs/performance.md`). With it every launch waits for its kernel: the same loop took 5.7 to 5.8 µs a launch (2026-11-24), and work that waits for what the program does later never ends: kernels that wait for one another (a collective over several GPUs, a kernel that waits for a flag that a later kernel sets), a host function (`cuLaunchHostFunc`, `cudaLaunchHostFunc`) that waits for something the launching thread does after the launch, a `cuStreamWaitValue32`/`64` (or a batch of stream memory operations) that waits for a value written later; the launch after it waits for ever. Nothing is waited for while a graph is being captured in the process; a CUDA graph's launch is waited for, and the graph is named as a whole, with the kernels it was made of. Any other value leaves it off |
| `VF_FAULT_REPORT` | on | `0` (exactly): no report of faults. Nothing is kept of the launches and of the synchronisations, nothing is said after a fault, `VF_FAULT_TRACE` has no effect, and the functions that the layer routes only for the report (the synchronisations and queries of contexts, streams and events, the beginnings and ends of captures, the instantiation, launch and destruction of CUDA graphs, the destruction of streams) are forwarded as they are, as without the report: a lookup of one (the CUDA runtime's way) gives NVIDIA's function, and a call by its exported name passes through the layer's entry, which calls NVIDIA's and returns its answer (and counts the call for `VF_STATS`); the calls that reach the driver are then those of volta-forever 0.1.5. To rule the report out of a problem. `VF_SAFE=1` leaves it as it is (the report changes no result). Any other value leaves it on |
| `VF_TRANSLATE_DUMP` | off | `DIR` (which must exist): write every cubin given to the translator there as it was received, as `PID-NUMBER.sm_SM.cubin`, to translate and validate it apart (`reference/validate.py`) |
| `VF_BORROW_SAY` | off | Set: the translator says on standard error, for each function, what registers it borrowed (as `reference/translate.py` does) |

The reports at exit (`VF_STATS`, `VF_PROFILE`, the summaries of emulated instructions and of
second translations, and `VF_SUMMARY`'s file) are printed by the process that loaded the layer,
not by a child that it made with `fork()`.

**Audit library (`libvf_audit.so`):**

| Variable | Default | Description |
|---|---|---|
| `VF_AUDIT_MAP_FILE` | `vf-audit.map` beside the library, when there is one | Path to a map file (one `name=path` per line; `#` comments; empty path = not found) |
| `VF_AUDIT_MAP` | none | Map entries in the environment, colon-separated; these come before the file's entries |

It also reads `VF_LOG`, `VF_LOG_LEVEL` (2: each substitution) and `VF_CAPABILITY`, which it
reports to NVML (`nvmlDeviceGetCudaComputeCapability`) on a V100.

**nvJitLink adapter (`libnvJitLink.so.13`):**

| Variable | Default | Description |
|---|---|---|
| `VF_JITLINK_REAL` | `libnvJitLink.so.12`, as the loader finds it | The nvJitLink 12.9 library to use |
| `VF_JITLINK_REAL_DIR` | none | A directory: take `libnvJitLink.so.12` there (when `VF_JITLINK_REAL` is not set) |
| `VF_JITLINK_DUMP` | off | `DIR`: write every PTX input there as the program gave it, as `PID-NUMBER.ptx` |

It adapts PTX as the proxy does, with the same switches of the rewriting (`VF_CAPABILITY`,
`VF_ALIGNED_ROWS`, `VF_BF16_*`, and `VF_SAFE`, which it does not say), and also reads `VF_LOG`, `VF_LOG_LEVEL` and
`VF_EMULATION_WARNINGS`. Its summary of emulated instructions at exit is printed by the process
that loaded it, not by a child made with `fork()`.

**Forwarding libraries (`forward/`):**

| Variable | Default | Description |
|---|---|---|
| `VF_FWD_<KEY>` | the CUDA 12.9.2 library's name | The library that one forwarding library opens instead, by path or name. KEY is its short name in capitals, `-` written `_`: `VF_FWD_CUBLAS`, `VF_FWD_CUBLASLT`, `VF_FWD_CUFFT`, `VF_FWD_NVRTC_BUILTINS`, `VF_FWD_NPPIF`, ... (`crates/vf-fwdgen/src/libs.rs`) |
| `VF_FWD_VERBOSE` | off | Set: each forwarding library says which library it opened and each function that it binds |
| `VF_CUBLAS_BF16_VERIFY` | `0` | `1`: verify mode of the bf16 cuBLAS variants (`forward/bf16tc`), with a path on: every call the skinny path serves, and the tensor-core path's fallback elements, also computed by cuBLAS 12.9 and compared bit for bit; the report at exit ("The bf16 cuBLAS variants", above). Read at the variants' first intercepted call |
| `VF_SAFE` | unset | Safe mode (the proxy's table, above, is its description): in the bf16 cuBLAS variants (`forward/bf16tc`) the tensor-core path serves nothing, whatever `VF_CUBLAS_BF16_TC` says, and the skinny path keeps cuBLAS 12.9's bits ("The bf16 cuBLAS variants", above). Read at the variants' first intercepted call |

**Launcher (`vf`) and `vf ptxas`:**

| Variable | Default | Description |
|---|---|---|
| `VF_CUDA12` | found (section 2) | The directory of the CUDA 12.9.2 libraries (`--cuda12` comes first). `vf ptxas` takes `bin/ptxas` beside it |
| `VF_QUIET` | `0` | `1`: no line when the CUDA 12.9.2 libraries are not found |
| `VF_VENV` | unset | `0`: a Python of an environment made by `vf venv` starts as it is, not through the layer |
| `VF_PTXAS_REAL` | `bin/ptxas` beside `VF_CUDA12`, else `/usr/local/cuda-12.9/bin/ptxas` | The CUDA 12.9 assembler that `vf ptxas` runs |
| `VF_JIT_LINEINFO` | unset | `1`: `NVCC_APPEND_FLAGS` gains `-lineinfo` and `CUTE_DSL_LINEINFO=1` is set, so that compilers run by the program keep the PTX of what they build, for a program told 8.0 or later ("What the launcher sets", above) |
| `VF_CUBLAS_BF16_TC` | `0` | `1`: `libcublas.so.13` and `libcublasLt.so.13` are the bf16 cuBLAS variants of `forward/bf16tc`, whose runtime library reads the variable. The launcher refuses to run when they are not installed ("The bf16 cuBLAS variants", above). `0` or `1` |
| `VF_CUBLAS_BF16_SKINNY` | `0` | `1`: selects the same variants, whose runtime library reads this variable too; the same refusals. `0` or `1` |
| `VF_SAFE` | unset | Safe mode (the proxy's table, above): passed on as it is, and the libraries that the bf16 switches select are the same with it. It turns `VF_TRITON_TC` off, whatever it is set to: the launcher does not put `python/` in `PYTHONPATH` and takes out the copies that are there (`python/sitecustomize.py` and `python/volta_forever_triton.py` read `VF_SAFE` themselves by the same rule, for a `PYTHONPATH` set by hand). `vf env` prints it when it is `1`; `vf doctor` says that it is on and which of the switches that are set it turns off, or that a value other than `0` and `1` leaves it off |
| `VF_TRITON_TC` | unset (the launcher's `triton_tc_default`: `0`) | `1`: Triton compiles a kernel whose products (`tl.dot`) all multiply float16, bfloat16 or int8 operands for sm_80, which the layer runs on Volta's tensor cores (float16) or with its emulated products (bfloat16: float32 `fma`; int8: `dp4a`), where it would compile it for 7.5 with float32 `fma`; a list of some of `f16`, `bf16`, `i8` takes those alone; `0` or empty: nothing; unset: the release's default, off in this one; `VF_SAFE=1`: nothing. The launcher puts `python/` first in `PYTHONPATH` ("What the launcher sets"); `python/volta_forever_triton.py` reads it again in each Python and acts only at the default capability (`VF_CAPABILITY` unset or 7.5), on a machine whose GPUs are all Voltas, with the layer's `libcuda.so.1` in the process and its tool `vf-translate` in the installation; it says once why when it does not. Told 8.0 it does nothing, so a bfloat16 model that vLLM must be told 8.0 for gains nothing from it; at the default capability, vLLM's engine with a model made float16 gave 6 % more at 32 requests (`docs/guides/vllm.md`). `LIMITS.md`, "Triton's matrix products" |
| `VF_TRITON_TC_TIMEOUT` | `900` | Seconds that a compilation for sm_80 may take in a helper (`python/volta_forever_triton_helper.py`), not counting a time during which the program itself was stopped; the helper is then killed with what it runs (its process group: `ptxas`, which Triton runs with the helper's descriptors), and the kernel compiled by Triton's own stages, for 7.5, as is one whose compilation ends the helper (an error of LLVM or MLIR, an assertion: SIGSEGV, SIGILL or SIGFPE; SIGABRT or SIGBUS, which a lack of memory gives too, after the kernel was compiled again once in a new helper), and Triton's cache keeps it so; but when the helper, or a process of its group, was stopped from outside (SIGSTOP, a debugger, a frozen control group), nothing is kept. It also bounds a helper's import of Triton (it then ends by itself) |
| `VF_TRITON_TC_START` | `20` | Seconds that the first kernel of a process waits for its first helper to import Triton (which may take long from a cold network disk); then that kernel, and each next one until the helper is ready, is compiled with Triton's own stages and not kept in Triton's cache (said once), and no kernel waits for it again. A helper is started only with the interpreter that runs the process (`sys.executable` when it is the file of the running process and has a Python's or PyPy's name; never for a frozen application: the switch is then not used in that process) |
| `VF_TRITON_TC_JOBS` | a quarter of the processors, 2 to 8 | The helpers that the user's processes have at most on one host, all programs that share Triton's cache directory together (processes whose Triton caches differ count their places apart: vLLM gives each model and each rank a cache of its own unless `TRITON_CACHE_DIR` is set): each helper holds one of that many lock files (`slot-HOST-N`, `flock`) in `volta-forever-EUID` of Triton's cache directory (as Triton finds it when the process compiles its first kernel that may be compiled for sm_80) from its start to its end, so that their memory is at most that many times about 150 MB (a process with another value takes its places among its own number of files); a process has that many helpers at most, and compiles that many kernels at once. A best effort: where a lock file cannot be opened or locked for another reason than another holder (a file system whose locks do not work, a directory that cannot be written), the process does not count places, said at `VF_LOG_LEVEL=2`, and its helpers start without one |
| `VF_TRITON_TC_WAIT` | `60` | Seconds that a process waits for a place for a helper (`VF_TRITON_TC_JOBS`) when all are taken, once in its life: when none came (places held by helpers that do not give them up, stopped or stuck), it says so once, waits for none again, and compiles with one helper of its own without a place when none is free, that kernel included (the kernels of its other threads that waited with it then wait for that helper, as below). Meanwhile a helper of another process that has had nothing to do for half a second ends, and gives its place up. Also the time that a kernel waits when every helper of its process is busy, after which it is compiled with Triton's own stages and not kept in Triton's cache (said once) |
| `VF_TRITON_TC_IDLE` | `10` | Seconds after which a helper that has had nothing to do ends (its memory, 105 to 150 MB, is given back); the next kernel that needs one starts another (about 0.6 s, more from a cold disk). A helper also ends with its process |
| `VF_TRITON_TC_GPUS` | the driver's (`/proc/driver/nvidia/gpus`) | For the tests: the names of the machine's GPUs, separated by commas, that `python/volta_forever_triton.py` takes on a machine without the NVIDIA driver (no `/proc/driver/nvidia`) alone; where the driver is, its names are taken |

## What the first run costs

The first time a program runs through the layer, the proxy:

1. **Adapts** each module's PTX for Volta (rewrites instructions Volta lacks) and **analyses**
   what each kernel depends on. For llama.cpp (12 modules of 14-20 MB each), this takes about
   5.6 seconds (3.1 s for the rewriting, 2.5 s for the analysis).
2. **Translates** sm_75 machine code and supported sm_80 functions to sm_70. A stock PyTorch
   wheel includes kernels that are distributed as machine code only.
3. The **driver compiles** the rewritten PTX. Large attention kernels take 1.1 to 1.8 seconds
   each when the driver's own cache of compiled kernels is empty.
4. With `VF_TRITON_TC`, Triton compiles its kernels of these products for sm_80 in helpers, which a
   first run starts, and the driver compiles their PTX at their first launch (about 0.45 s each):
   `torch.compile` with max-autotune took 6 to 18 % longer on a run whose caches were all new
   (about 13 % over the pooled runs of six jobs, most of it the driver's compilations, 4 to 5 s the
   helpers' starts), vLLM's engine about as long (`docs/performance.md`, "With `VF_TRITON_TC`").
   Later runs take the caches.

The proxy caches what it worked out in `~/.cache/volta-forever/proxy/` by default. With a warm
cache, llama.cpp's 48-token generation takes 1.3 seconds where a cold cache takes 6.9 seconds.
For PyTorch, a battery of 820 operations takes 5 to 7 seconds with a warm cache against 63
seconds without (`docs/performance.md`, "Where the time goes"). The cache directory must be owned by the
user and not writable by others; on a machine whose home is on a tmpfs, it is in memory and
lost on reboot.

## Licence

The layer is distributed under the Apache License, version 2.0 (`LICENSE`), except the
instruction tables of the machine-code translator, which derive from descriptions distributed
under the GNU Lesser General Public License, version 2.1, and are distributed under it
(`LICENSE.LGPL-2.1`). The binaries contain those tables: `NOTICE` says where they come from,
where the complete source of the binaries is, and which Rust crates are linked in, with their
licence texts. The archive contains no NVIDIA software; `scripts/fetch-redist.py` downloads
the CUDA 12.9 libraries from NVIDIA's public archives, under NVIDIA's licence.

The layer is not NVIDIA's and is not endorsed by NVIDIA.

## The build this archive was made from

The `VERSION` file holds the version, the commit of the source repository and `-dirty` when
tracked files differed from that commit. The source of that commit is the complete source of
the binaries: `scripts/build.sh` and `scripts/build-forwarders.sh` build them, and
`scripts/package.sh` makes this archive.

## Tools

The offline tool is `bin/vf-translate`; the launcher has its four functions as subcommands:

```sh
vf inspect FILE...                  # which architectures a file carries
vf extract FILE DIR [SM]            # extract the code from a fat binary
vf adapt IN OUT                     # adapt PTX for Volta (what the proxy does)
vf translate IN OUT [MAP [REPORT]]  # translate supported sm_75 and sm_80 machine code to sm_70
```

`translate` writes beside MAP the file `MAP.lives` that `reference/validate.py` reads with it,
and REPORT has one line a function: `ok`, or why it is refused.
