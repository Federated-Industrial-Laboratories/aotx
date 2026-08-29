#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# module-build.sh: compile the device tool of one module directory to a module file and
# write the digest of that file into the manifest.
# Inputs: the module directory; --arch <number> names the architecture, and without it the
#   architecture comes from "aotx_boot --version" beside this script or in the build.
# Outputs: <directory>/<name>.ptx and the sha256 line of <directory>/module.manifest.
# Exit codes: 0 built, 1 refused, 2 usage or environment error.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
arch=""
dir=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --arch) arch="${2:-}"; shift 2 ;;
        -*) echo "module-build: $1 is not an option" >&2; exit 2 ;;
        *) dir="$1"; shift ;;
    esac
done
if [ -z "$dir" ]; then
    echo "usage: module-build.sh <directory> [--arch <number>]" >&2
    exit 2
fi
manifest="$dir/module.manifest"
if [ ! -f "$manifest" ]; then
    echo "module-build: $dir holds no module.manifest" >&2
    exit 2
fi

# The architecture of the build. A card that is newer than the module runs the module
# through the forward compatibility of the module text.
if [ -z "$arch" ]; then
    for boot in "$here/aotx_boot" "$root/build/aotx_boot"; do
        if [ -x "$boot" ]; then
            arch="$("$boot" --version 2>/dev/null | sed -n 's/.*arch sm_\([0-9]*\).*/\1/p')"
            [ -n "$arch" ] && break
        fi
    done
fi
if [ -z "$arch" ]; then
    echo "module-build: no architecture; give --arch <number>" >&2
    exit 2
fi
if ! command -v nvcc > /dev/null 2>&1; then
    echo "module-build: nvcc is not on the path" >&2
    exit 2
fi

name="$(basename "$dir")"
sources=""
for file in "$dir"/*.cu; do
    [ -e "$file" ] || continue
    sources="$sources $file"
done
if [ -z "$sources" ]; then
    echo "module-build: $dir holds no .cu file" >&2
    exit 2
fi

# A device tool is a device file. The seam gate refuses a host call in it, so a module that
# the gate refuses does not become a module file. The gate steps over a path that holds a
# part which starts with "build". The source therefore goes to a directory of its own
# first, and the gate examines it there.
gate_dir="$(mktemp -d)"
trap 'rm -rf "$gate_dir"' EXIT
# shellcheck disable=SC2086
cp $sources "$gate_dir/" || exit 2
if ! python3 "$here/seam-gate.py" "$gate_dir"; then
    echo "module-build: the seam gate refuses the source of $name" >&2
    exit 1
fi

out="$dir/$name.ptx"
# shellcheck disable=SC2086
if ! nvcc -ptx -arch="sm_$arch" -I "$root/sdk" $sources -o "$out"; then
    echo "module-build: the compiler refused the source of $name" >&2
    exit 1
fi

digest="$(sha256sum "$out" | cut -d' ' -f1)"
if grep -q '^sha256:' "$manifest"; then
    sed -i "s|^sha256:.*|sha256: $digest|" "$manifest"
else
    printf 'sha256: %s\n' "$digest" >> "$manifest"
fi
echo "module-build: $name -> $out for sm_$arch"
echo "module-build: sha256 $digest"
exit 0
