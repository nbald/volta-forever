# Common settings for the GPU test scripts.
# Sourced by every script of tests/gpu/ (and by tests/sass/ and tests/forward/ when they need a
# setting), right after it computes its tree (W=...).
#
# Reads tests/site.env when it exists: that file sets the paths of this site and is not tracked.
# Then sets VF_TEST_ROOT to a default (under XDG_CACHE_HOME) unless it is already set.
# Every path that a script uses is either an environment variable with a default under
# VF_TEST_ROOT, or a path of the tree itself.

[ -f "$W/tests/site.env" ] && . "$W/tests/site.env"

: "${VF_TEST_ROOT:=${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever/tests}"
export VF_TEST_ROOT

# vf_torch_map DIR: the path of the map that puts the forwarding libraries in front of the cuBLAS
# that a PyTorch wheel carries (INSTALL.md, way B; the wheel finds its libraries by a path recorded
# in its own, so the library path alone does not do it). $VF_AUDIT_MAP_FILE when it is set; else
# DIR/torch-audit.map, written here from this tree's dist/forward and toolkits/ when it does not
# exist yet.
# vf_layer PROXY MAP: the layer's environment for what the script runs next. By hand, as
# INSTALL.md has it ("By hand", way B): the proxy's directory on the library path and, when MAP
# exists, the audit library with it. With VF_LAUNCHER=1, as the launcher gives it
# (`vf env` of this tree: its dist/, every forwarding library, the CUDA 12.9.2
# libraries of toolkits/): what a user who puts `vf` in front of the command gets.
# (PROXY is then not looked at: the launcher takes this tree's own build.)
vf_layer() {
    if [ "${VF_LAUNCHER:-0}" = 1 ]; then
        eval "$("$W/scripts/vf" env)"
    else
        export LD_LIBRARY_PATH="$1"
        [ -f "$2" ] && export LD_AUDIT="$1/libvf_audit.so" VF_AUDIT_MAP_FILE="$2"
    fi
    return 0
}

vf_torch_map() {
    local map=${VF_AUDIT_MAP_FILE:-$1/torch-audit.map}
    if [ -z "${VF_AUDIT_MAP_FILE:-}" ] && [ ! -f "$map" ]; then
        mkdir -p "$1" && printf '%s\n' \
            "libcublas.so.13=$W/dist/forward/libcublas.so.13" \
            "libcublasLt.so.13=$W/dist/forward/libcublasLt.so.13" \
            "libcublas.so.12=$W/toolkits/cuda-12.9.2/lib/libcublas.so.12" \
            "libcublasLt.so.12=$W/toolkits/cuda-12.9.2/lib/libcublasLt.so.12" >"$map"
    fi
    echo "$map"
}

# vf_clear_settings [NAME...]: unset the settings of the layer that change what it does (INSTALL.md,
# "Environment variables"; not those that say where something is: VF_PROXY_CACHE, VF_REAL_LIBCUDA,
# VF_AUDIT_MAP_FILE, VF_JITLINK_REAL, VF_FWD_<KEY>, VF_CUDA12), except the NAMEs, which the calling
# script takes from its caller, and print one line that names those unset. For the scripts whose
# cases expect what the layer does by default and set what each case needs: a setting left in the
# shell (VF_CAPABILITY_ALL=1, VF_SAFE=1, VF_LOG) would make a right answer fail.
vf_clear_settings() {
    local name keep cleared=
    for name in VF_LOG_LEVEL VF_LOG VF_EMULATION_WARNINGS VF_SAFE VF_PROXY_CACHE_MAX VF_CAPABILITY \
        VF_CAPABILITY_ALL VF_TRANSLATE VF_PAIRS VF_COMPANIONS VF_BORROW VF_FIT VF_FIT_SHARED VF_BF16_TC_GROUPS \
        VF_BF16_TC_TILES VF_BF16_COLUMN VF_BF16_ROW VF_ALIGNED_ROWS VF_LAZY VF_REPORTED_TRAPS VF_ADAPT VF_STATS \
        VF_PROFILE VF_SUMMARY VF_SUMMARY_INTERVAL VF_TIMING VF_TRACE_LAUNCH VF_FAULT_TRACE \
        VF_FAULT_REPORT VF_TRANSLATE_DUMP VF_BORROW_SAY VF_AUDIT_MAP VF_JITLINK_DUMP VF_FWD_VERBOSE \
        VF_CUBLAS_BF16_TC VF_CUBLAS_BF16_SKINNY VF_QUIET VF_VENV VF_JIT_LINEINFO VF_THREADS VF_TRITON_TC \
        VF_TRITON_TC_TIMEOUT VF_TRITON_TC_START VF_TRITON_TC_JOBS VF_TRITON_TC_WAIT VF_TRITON_TC_IDLE \
        VF_TRITON_TC_GPUS; do
        [ -n "${!name+set}" ] || continue
        for keep in "$@"; do [ "$name" = "$keep" ] && continue 2; done
        unset "$name"
        cleared="$cleared $name"
    done
    [ -z "$cleared" ] || echo "$(basename "$0"): unset the layer's settings that this suite does not take from the environment:$cleared"
}
