"""Safe mode (VF_SAFE=1) in the proxy, against a stand-in for the driver that keeps every image it is
given: the switches whose results are not bit-identical to the default's are not used, whatever
they say, nor the pairs of products of machine code (each product is translated by itself, as with
VF_PAIRS=0), and the proxy says so once; any value but 1 leaves them in use; the switches that give the
default's bits, or refuse, and the pairs of the PTX lowering are left as they are; and what the
proxy's cache keeps from a run without safe mode is not given to a run with it, nor the reverse.

Run with python3 tests/proxy/safe_mode.py path/to/libcuda.so.1 (each case is a process of its own:
the proxy reads VF_SAFE once, when it is loaded).
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


DRIVER = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Each image given to a loading function is written to $MOCK_IMAGE (the last one stays): PTX text
   to its end, a fat binary by the size in its header, a cubin to the end of its section headers. */
static void keep(const void *image) {
    const unsigned char *bytes = image;
    size_t length;
    unsigned magic;
    memcpy(&magic, bytes, 4);
    if (magic == 0xBA55ED50u) {
        unsigned long long size;
        memcpy(&size, bytes + 8, 8);
        length = 16 + size;
    } else if (!memcmp(bytes, "\177ELF", 4)) {
        unsigned long long shoff;
        unsigned short shentsize, shnum;
        memcpy(&shoff, bytes + 0x28, 8);
        memcpy(&shentsize, bytes + 0x3a, 2);
        memcpy(&shnum, bytes + 0x3c, 2);
        length = shoff + (size_t)shentsize * shnum;
    } else {
        length = strlen(image);
    }
    const char *path = getenv("MOCK_IMAGE");
    FILE *f = path ? fopen(path, "wb") : 0;
    if (!f) return;
    fwrite(image, 1, length, f);
    fclose(f);
}
static uintptr_t next = 0x1000;
int cuInit(unsigned flags) { return 0; }
int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
int cuDeviceGetCount(int *count) { *count = 1; return 0; }
int cuDeviceGetAttribute(int *v, int attribute, int device) {
    *v = attribute == 75 ? 7 : attribute == 76 ? 0 : 42;
    return 0;
}
int cuDeviceComputeCapability(int *major, int *minor, int device) { *major = 7; *minor = 0; return 0; }
int cuCtxGetCurrent(void **context) { *context = (void *)0x100; return 0; }
int cuCtxGetDevice(int *device) { *device = 0; return 0; }
int cuModuleLoadData(void **module, const void *image) { keep(image); *module = (void *)(next += 16); return 0; }
int cuModuleLoadDataEx(void **module, const void *image, unsigned n, int *o, void **v) {
    return cuModuleLoadData(module, image);
}
int cuModuleUnload(void *module) { return 0; }
int cuModuleGetFunction(void **function, void *module, const char *name) { *function = (void *)((uintptr_t)module + 1); return 0; }
int cuLibraryLoadData(void **library, const void *image, int *jo, void **jv, unsigned jn, int *lo, void **lv, unsigned ln) {
    keep(image);
    *library = (void *)(next += 16);
    return 0;
}
int cuLibraryUnload(void *library) { return 0; }
int cuLibraryGetKernel(void **kernel, void *library, const char *name) { *kernel = (void *)((uintptr_t)library + 2); return 0; }
int cuLibraryGetModule(void **module, void *library) { *module = (void *)(next += 16); return 0; }
int cuModuleGetLoadingMode(int *mode) { *mode = 2; return 0; }
"""

# load PROXY FILE [library]: give FILE to cuModuleLoadData (or cuLibraryLoadData) of PROXY.
MAIN = r"""
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    void *proxy = dlopen(argv[1], RTLD_NOW);
    if (!proxy) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    FILE *f = fopen(argv[2], "rb");
    if (!f) return 2;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char *image = calloc(size + 1, 1);
    if (fread(image, 1, size, f) != (size_t)size) return 2;
    fclose(f);
    void *handle = 0;
    int status;
    if (argc > 3) {
        int (*load)(void **, const void *, int *, void **, unsigned, int *, void **, unsigned) =
            (int (*)(void **, const void *, int *, void **, unsigned, int *, void **, unsigned))dlsym(proxy, "cuLibraryLoadData");
        status = load(&handle, image, 0, 0, 0, 0, 0, 0);
    } else {
        int (*load)(void **, const void *) = (int (*)(void **, const void *))dlsym(proxy, "cuModuleLoadData");
        status = load(&handle, image);
    }
    printf("status %d\n", status);
    return status != 0;
}
"""

W = Path(__file__).resolve().parents[2]
CASES = W / "tests" / "kernels" / "ptx-cases"
GROUP, TILES, COLUMN, ROW = (CASES / f"bf16-{name}.ptx" for name in ("group", "tiles", "column", "row"))
PAIRS = CASES / "pairs-adversarial.ptx"
ELF75_PTX75 = W / "crates" / "vf-fatbin" / "tests" / "data" / "elf75-ptx75-zstd.fatbin"
ELF70_ELF75_PTX70 = W / "crates" / "vf-fatbin" / "tests" / "data" / "elf70-elf75-ptx70-lz4.fatbin"
CUBIN = W / "tests" / "kernels" / "k_fit_test.135.sm75.cubin"
PAIRED = W / "tests" / "kernels" / "k_pair_bits.135.sm75.cubin"
K_PAIRS = W / "tests" / "kernels" / "k_pairs.135.sm75.cubin"
MACHINE_PAIRS = "VF_PAIRS in machine code"

checks = wrong = 0


def check(name, ok, what=""):
    global checks, wrong
    checks += 1
    if not ok:
        wrong += 1
    print(f"{'ok' if ok else 'WRONG'}  {name}{'' if ok else ': ' + str(what)}")


class Proxy:
    def __init__(self, proxy, out):
        self.proxy, self.out = proxy, out
        subprocess.run(["gcc", "-shared", "-fPIC", "-O1", "-o", out / "mock.so", "-x", "c", "-"],
                       input=DRIVER.encode(), check=True)
        subprocess.run(["gcc", "-O1", "-o", out / "load", "-x", "c", "-", "-ldl"], input=MAIN.encode(), check=True)
        # What the test's own environment says of the layer is not the cases'.
        self.base = {k: v for k, v in os.environ.items() if not k.startswith("VF_") and k not in ("LD_AUDIT", "LD_PRELOAD")}
        self.base.update(VF_REAL_LIBCUDA=str(out / "mock.so"), VF_PROXY_CACHE="0", VF_LOG_LEVEL="1",
                         VF_EMULATION_WARNINGS="0", VF_LAZY="0", MOCK_IMAGE=str(out / "image"))

    def run(self, image, env=(), library=False):
        """The image that the driver was given for `image`, and what the proxy said."""
        (self.out / "image").unlink(missing_ok=True)
        environment = dict(self.base, **dict(env))
        args = [self.out / "load", self.proxy, image] + (["library"] if library else [])
        done = subprocess.run(args, env=environment, capture_output=True, text=True, timeout=120)
        if done.returncode != 0 or "status 0" not in done.stdout:
            raise RuntimeError(f"{image.name} {dict(env)}: {done.stdout}{done.stderr}")
        given = (self.out / "image").read_bytes() if (self.out / "image").exists() else None
        return given, [line for line in done.stderr.splitlines() if "VF_SAFE" in line]


def images(proxy):
    # Each switch that safe mode turns off changes what this bf16 product is lowered to; with
    # VF_SAFE=1 the driver is given the default's text, and one line names the switch.
    for name, value, ptx in (("VF_BF16_TC_GROUPS", "1", GROUP), ("VF_BF16_TC_TILES", "1", TILES),
                             ("VF_BF16_COLUMN", "tree", COLUMN), ("VF_BF16_ROW", "tree", ROW)):
        default, said = proxy.run(ptx)
        check(f"{ptx.name}: nothing said without VF_SAFE", said == [], said)
        switched, _ = proxy.run(ptx, {name: value})
        check(f"{ptx.name}: {name}={value} changes the lowering (control)", switched != default)
        safe, said = proxy.run(ptx, {name: value, "VF_SAFE": "1"})
        check(f"{ptx.name}: {name}={value} with VF_SAFE=1 gives the default's text", safe == default)
        check(f"{ptx.name}: {name}={value} with VF_SAFE=1 is said once", len(said) == 1 and f"{name}={value}" in said[0]
              and said[0].startswith("[volta-forever] VF_SAFE=1: the layer keeps its conservative paths"), said)
        quiet, said = proxy.run(ptx, {name: value, "VF_SAFE": "1", "VF_LOG_LEVEL": "0"})
        check(f"{ptx.name}: said at level 1, not at 0", quiet == default and said == [], said)

    # The ordered modes give the default's bits: left as they are.
    for name, ptx in (("VF_BF16_COLUMN", COLUMN), ("VF_BF16_ROW", ROW)):
        default, _ = proxy.run(ptx)
        ordered, _ = proxy.run(ptx, {name: "ordered"})
        check(f"{ptx.name}: {name}=ordered changes the lowering (control)", ordered != default)
        safe, said = proxy.run(ptx, {name: "ordered", "VF_SAFE": "1"})
        check(f"{ptx.name}: {name}=ordered stays with VF_SAFE=1", safe == ordered)
        check(f"{ptx.name}: {name}=ordered is not said to be turned off",
              len(said) == 1 and name not in said[0] and MACHINE_PAIRS in said[0], said)

    # Every switch at once: one line, which names each.
    switches = {"VF_BF16_TC_GROUPS": "1", "VF_BF16_TC_TILES": "1", "VF_BF16_COLUMN": "tree", "VF_BF16_ROW": "tree",
                "VF_TRANSLATE": "first", "VF_COMPANIONS": "1"}
    group_default, _ = proxy.run(GROUP)
    safe, said = proxy.run(GROUP, dict(switches, VF_SAFE="1"))
    check("every switch with VF_SAFE=1: the default's text", safe == group_default)
    check("every switch with VF_SAFE=1: one line that names each",
          len(said) == 1 and all(f"{k}={v}" in said[0] for k, v in switches.items()), said)

    # Another form of the program: machine code where the default takes the PTX, and the program's
    # sm_75 machine code where its sm_70 code runs. With VF_SAFE=1, what the default gives.
    for value, image in (("first", ELF75_PTX75), ("force", ELF70_ELF75_PTX70)):
        default, _ = proxy.run(image)
        translated, _ = proxy.run(image, {"VF_TRANSLATE": value})
        check(f"{image.name}: VF_TRANSLATE={value} gives another image (control)", translated != default)
        safe, said = proxy.run(image, {"VF_TRANSLATE": value, "VF_SAFE": "1"})
        check(f"{image.name}: VF_TRANSLATE={value} with VF_SAFE=1 gives the default's image", safe == default)
        check(f"{image.name}: VF_TRANSLATE={value} with VF_SAFE=1 is said",
              len(said) == 1 and f"VF_TRANSLATE={value}" in said[0], said)

    # Only the exact value 1 turns safe mode on; 0 says nothing, another value says it is not 1.
    grouped, _ = proxy.run(GROUP, {"VF_BF16_TC_GROUPS": "1"})
    for value in ("0", "", "true", "yes", " 1", "1 ", "01"):
        given, said = proxy.run(GROUP, {"VF_BF16_TC_GROUPS": "1", "VF_SAFE": value})
        check(f"VF_SAFE={value!r} leaves the switch in use", given == grouped)
        wanted = [] if value == "0" else [f"[volta-forever] VF_SAFE='{value}' is not 1: safe mode stays off"]
        check(f"VF_SAFE={value!r}: {'nothing said' if value == '0' else 'said once'}", said == wanted, said)
    # VF_SAFE=1 alone: the default's text, and a line that says so.
    safe, said = proxy.run(GROUP, {"VF_SAFE": "1"})
    check("VF_SAFE=1 alone: the default's text", safe == group_default)
    check("VF_SAFE=1 alone: said once, with the pairs of machine code",
          len(said) == 1 and said[0].startswith(f"[volta-forever] VF_SAFE=1: the layer keeps its conservative paths: "
                                                f"{MACHINE_PAIRS} not used"), said)

    # Left as they are: the pairs of the PTX lowering (the same bits as each product by itself), the
    # test of the rows' alignment, what is not translated.
    pairs_default, _ = proxy.run(PAIRS)
    safe, said = proxy.run(PAIRS, {"VF_SAFE": "1"})
    check(f"{PAIRS.name}: its pairs stay with VF_SAFE=1", safe == pairs_default)
    for name, value in (("VF_PAIRS", "0"), ("VF_ALIGNED_ROWS", "0")):
        alone, _ = proxy.run(PAIRS, {name: value})
        check(f"{PAIRS.name}: {name}={value} changes the lowering (control)", alone != pairs_default)
        safe, said = proxy.run(PAIRS, {name: value, "VF_SAFE": "1"})
        check(f"{PAIRS.name}: {name}={value} stays with VF_SAFE=1", safe == alone)
        check(f"{PAIRS.name}: {name}={value} is not said to be turned off",
              len(said) == 1 and name not in said[0], said)

    # The pairs of products of machine code (two products on one A from whole-row loads, which the
    # translator computes together, adding the halves of k where a product by itself adds the even k,
    # then the odd): with VF_SAFE=1, each product by itself, as with VF_PAIRS=0, and said.
    paired, _ = proxy.run(PAIRED)
    unpaired, _ = proxy.run(PAIRED, {"VF_PAIRS": "0"})
    check(f"{PAIRED.name}: VF_PAIRS=0 gives another image (control)", unpaired != paired)
    for env in ({"VF_SAFE": "1"}, {"VF_SAFE": "1", "VF_PAIRS": "1"}):
        safe, said = proxy.run(PAIRED, env)
        check(f"{PAIRED.name}: {env} gives the image of VF_PAIRS=0", safe == unpaired)
        check(f"{PAIRED.name}: {env} says that the pairs of machine code are not used",
              len(said) == 1 and f"{MACHINE_PAIRS} not used" in said[0], said)
    safe, said = proxy.run(PAIRED, {"VF_SAFE": "1", "VF_PAIRS": "0"})
    check(f"{PAIRED.name}: VF_PAIRS=0 with VF_SAFE=1 stays", safe == unpaired)
    check(f"{PAIRED.name}: VF_PAIRS=0 with VF_SAFE=1: nothing turned off",
          len(said) == 1 and "nothing that safe mode turns off is in use" in said[0], said)
    notsafe, said = proxy.run(PAIRED, {"VF_SAFE": "true"})
    check(f"{PAIRED.name}: VF_SAFE='true' keeps the pairs", notsafe == paired)

    # Opt-in VF_COMPANIONS=1 (companion hoisting + per-group paired HMMA scheduling): off by
    # default; with VF_PAIRS=0 or VF_SAFE=1 it changes nothing, and VF_SAFE=1 names it.
    kp_default, _ = proxy.run(K_PAIRS)
    kp_comp0, _ = proxy.run(K_PAIRS, {"VF_COMPANIONS": "0"})
    kp_comp1, _ = proxy.run(K_PAIRS, {"VF_COMPANIONS": "1"})
    kp_unpaired, _ = proxy.run(K_PAIRS, {"VF_PAIRS": "0"})
    kp_unpaired_comp1, _ = proxy.run(K_PAIRS, {"VF_PAIRS": "0", "VF_COMPANIONS": "1"})
    check(f"{K_PAIRS.name}: VF_COMPANIONS=0 equals the default", kp_comp0 == kp_default)
    check(f"{K_PAIRS.name}: VF_COMPANIONS=1 changes the image (control)", kp_comp1 != kp_default)
    check(f"{K_PAIRS.name}: VF_COMPANIONS=1 with VF_PAIRS=0 equals VF_PAIRS=0 alone", kp_unpaired_comp1 == kp_unpaired)
    kp_safe_comp1, said = proxy.run(K_PAIRS, {"VF_SAFE": "1", "VF_COMPANIONS": "1"})
    check(f"{K_PAIRS.name}: VF_COMPANIONS=1 with VF_SAFE=1 gives the image of VF_PAIRS=0", kp_safe_comp1 == kp_unpaired)
    check(f"{K_PAIRS.name}: VF_COMPANIONS=1 with VF_SAFE=1 says both are not used",
          len(said) == 1 and "VF_COMPANIONS=1" in said[0] and MACHINE_PAIRS in said[0], said)
    kp_notsafe_comp1, _ = proxy.run(K_PAIRS, {"VF_SAFE": "true", "VF_COMPANIONS": "1"})
    check(f"{K_PAIRS.name}: VF_SAFE='true' keeps VF_COMPANIONS=1", kp_notsafe_comp1 == kp_comp1)


def cache(proxy, out):
    def files(directory, suffix):
        return sorted(p.name for p in directory.iterdir() if p.name.endswith(suffix))

    def fresh(name):
        directory = out / name
        directory.mkdir(mode=0o700)
        return directory

    # Machine code (its translation is kept, keyed by the image and the policy): without products,
    # the same bytes in both modes; with pairs of products, other bytes in safe mode.
    for cubin in (CUBIN, PAIRED):
        cubin_runs = {}
        for order in (("default", "safe"), ("safe", "default")):
            directory = fresh(f"cache-{cubin.stem}-" + "-".join(order))
            settings = {"VF_PROXY_CACHE": str(directory), "VF_LAZY": "1"}
            seen = []
            for _ in range(2):
                for mode in order:
                    given, _ = proxy.run(cubin, dict(settings, **({"VF_SAFE": "1"} if mode == "safe" else {})))
                    cubin_runs.setdefault(mode, given)
                    check(f"{cubin.name} ({' then '.join(order)}): {mode} gives the same image from the cache or not",
                          given == cubin_runs[mode])
                    seen.append(files(directory, ".t"))
            check(f"{cubin.name} ({' then '.join(order)}): one translation kept for each, under keys of their own",
                  [len(s) for s in seen] == [1, 2, 2, 2], seen)
            cubin_runs.setdefault("keys", seen[-1])
            check(f"{cubin.name} ({' then '.join(order)}): the same keys whatever the order", seen[-1] == cubin_runs["keys"])
        check(f"{cubin.name}: the same image in both modes" if cubin == CUBIN else f"{cubin.name}: another image in safe mode",
              (cubin_runs["safe"] == cubin_runs["default"]) == (cubin == CUBIN))

    # PTX loaded as a library and compiled kernel by kernel (a module of two kernels: the bf16 one,
    # and the same under another name): what is worked out of it is kept, keyed the same way. A
    # switch that safe mode turns off does not change the key of a run in safe mode.
    text = GROUP.read_text()
    body = text[text.index(".visible .entry k("):]
    two = out / "two-kernels.ptx"
    two.write_text(text + "\n" + body.replace(".visible .entry k(", ".visible .entry k2(", 1))
    directory = fresh("cache-library")
    settings = {"VF_PROXY_CACHE": str(directory), "VF_LAZY": "1"}
    manifests = []
    for env in ({}, {"VF_BF16_TC_GROUPS": "1"}, {"VF_BF16_TC_GROUPS": "1", "VF_SAFE": "1"}, {"VF_SAFE": "1"}, {}):
        proxy.run(two, dict(settings, **env), library=True)
        manifests.append(files(directory, ".manifest"))
    check("library: default, the switch, the switch in safe mode: three keys; safe mode alone: the third again",
          [len(m) for m in manifests] == [1, 2, 3, 3, 3], manifests)

    # VF_COMPANIONS=1 gets a cache key of its own, separate from the default and from safe mode;
    # in safe mode, VF_COMPANIONS=1 shares the safe-mode key.
    for library in (False, True):
        directory = fresh("cache-companions-" + ("lib" if library else "mod"))
        settings = {"VF_PROXY_CACHE": str(directory), "VF_LAZY": "1"}
        seen = []
        for env in ({}, {"VF_COMPANIONS": "1"}, {"VF_COMPANIONS": "1", "VF_SAFE": "1"}, {"VF_SAFE": "1"}, {}):
            proxy.run(K_PAIRS, dict(settings, **env), library=library)
            seen.append(files(directory, ".t"))
        kind = "library" if library else "module"
        check(f"{K_PAIRS.name} ({kind}): default, VF_COMPANIONS=1, safe: three keys; safe alone and default reuse theirs",
              [len(s) for s in seen] == [1, 2, 3, 3, 3], seen)


def main():
    proxy = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="vf-safe-mode-") as out:
        out = Path(out)
        test = Proxy(proxy, out)
        images(test)
        cache(test, out)
    print(f"{checks} checks, {wrong} wrong")
    print("PASS  safe_mode" if wrong == 0 else "FAIL  safe_mode")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
