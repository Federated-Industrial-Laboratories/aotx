#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# module_setup.sh: copy the example modules and the check fixtures into the build
# directory. It builds the module file of each one that holds a source file.
# Inputs: the source directory, the build directory, the architecture number.
# Outputs: <build>/modules/<name>/ with a manifest and a module file.
# Exit codes: 0 built, 1 when a build failed, 2 usage or environment error.
set -u
if [ "$#" -lt 3 ]; then
    echo "usage: module_setup.sh <source-dir> <build-dir> <arch>" >&2
    exit 2
fi
source_dir="$1"
build_dir="$2"
arch="$3"
out="$build_dir/modules"
rm -rf "$out"
mkdir -p "$out" || exit 2
status=0
for dir in "$source_dir"/sdk/examples/* "$source_dir"/tests/fixtures/modules/*; do
    [ -d "$dir" ] || continue
    name="$(basename "$dir")"
    cp -r "$dir" "$out/$name" || exit 2
    chmod -R u+w "$out/$name"
    if ls "$out/$name"/*.cu > /dev/null 2>&1; then
        if ! bash "$source_dir/tools/module-build.sh" "$out/$name" --arch "$arch"; then
            echo "module_setup: the module $name did not build" >&2
            status=1
        fi
    fi
done
# A module whose sha256 line is not the digest of its module file. The copy takes the built
# word_count and puts a line of zeros in the place of its digest. The reader of the device
# then meets a manifest that names a file it no longer describes.
if [ -d "$out/word_count" ]; then
    cp -r "$out/word_count" "$out/bad_digest" || exit 2
    chmod -R u+w "$out/bad_digest"
    mv "$out/bad_digest/word_count.ptx" "$out/bad_digest/bad_digest.ptx"
    sed -i 's/^name: .*/name: bad_digest/; s/^module: .*/module: bad_digest.ptx/' \
        "$out/bad_digest/module.manifest"
    sed -i "s/^sha256: .*/sha256: $(printf '0%.0s' $(seq 64))/" \
        "$out/bad_digest/module.manifest"
fi

# The modules the build script must refuse are copied and not built. A check of its own
# runs the build script over each one and asks for the refusal.
for dir in "$source_dir"/tests/fixtures/modules-refused/*; do
    [ -d "$dir" ] || continue
    cp -r "$dir" "$build_dir/modules/$(basename "$dir")" || exit 2
    chmod -R u+w "$build_dir/modules/$(basename "$dir")"
done
echo "module_setup: the modules of the check stand in $out"
exit "$status"
