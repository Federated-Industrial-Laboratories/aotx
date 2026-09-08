#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild fixed CPU row files and require their recorded digests.

Inputs: a corpus or reference index, model files and compiled CPU reference program.
Outputs: a new directory with unchanged reference files and verified rows.
Exit codes: zero on complete matching output, one on a changed or absent input.
"""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys

import numpy as np

from arch_accuracy_io import (checked_text, local_file, read_corpus, reference_metadata, reference_rows,
                              retain_input, verify_inputs, write_json)


def checked_rows(root, sequence, vocab):
    rows = reference_rows(root, sequence, vocab)
    for index, row in enumerate(rows):
        if not np.isfinite(row).all() or not np.any(row != 0):
            raise ValueError("non-finite or zero reference row")
        if int(np.argmax(row)) != sequence["reference_argmax_ids"][index]:
            raise ValueError("reference argmax differs from its row")
    return reference_metadata(root, sequence)


def retain_sized(inputs, name, path, record):
    identity = retain_input(inputs, name, path, record["sha256"])
    if identity["bytes"] != record["bytes"]:
        raise ValueError("input size differs: " + name)
    return identity


def library_environment(inputs, libraries, executable):
    if not libraries:
        raise ValueError("the decoded reference requires pinned CPU libraries")
    directories = {Path(value["path"]).absolute().parent for value in libraries.values()}
    if len(directories) != 1:
        raise ValueError("CPU libraries must share one directory")
    directory = directories.pop()
    for name, value in libraries.items():
        path = local_file(directory, name)
        if path != Path(value["path"]).resolve():
            raise ValueError("CPU library path differs")
        retain_sized(inputs, "library:" + name, value["path"], value)
    env = os.environ.copy()
    for name in ("LD_PRELOAD", "LD_AUDIT"):
        env.pop(name, None)
    env.update(LD_LIBRARY_PATH=str(directory), GGML_BACKEND_PATH=str(directory),
               CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="14")
    result = subprocess.run(["ldd", str(executable.resolve())], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise ValueError("cannot check CPU reference library links")
    loaded = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith(("libllama", "libggml")):
            continue
        if len(fields) < 3 or fields[1] != "=>" or fields[0] not in libraries:
            raise ValueError("CPU reference links an unknown library")
        if Path(fields[2]).resolve() != Path(libraries[fields[0]]["path"]).resolve():
            raise ValueError("CPU reference links a different library")
        loaded.add(fields[0])
    if not any(name.startswith("libllama.so") for name in loaded):
        raise ValueError("CPU reference library link is absent")
    return env, result.stdout


def copy_dependencies(output, copies, inputs):
    for relative, source in copies.items():
        destination = local_file(output, relative)
        expected = retain_input(inputs, "dependency:" + str(source), source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(source).open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        retain_input(inputs, "copy:" + relative, destination, expected["sha256"])


def decoded(args):
    from arch_accuracy_reference import read_reference_index
    index = read_reference_index(args.reference_index, args.asset_root)
    inputs = dict(index["verified_inputs"])
    retain_input(inputs, "reference_index", args.reference_index)
    executable = retain_input(inputs, "executable", args.executable)
    checked_text(str(args.output.resolve()))
    copies = {args.reference_index.name: str(args.reference_index.resolve())}
    models, sequences, libraries = {}, {}, None
    for item in index["corpora"].values():
        path, corpus = Path(item["path"]), item["corpus"]
        context = corpus["_reference"]
        if path.parent.resolve() != args.reference_index.parent.resolve():
            raise ValueError("reference files must share the index directory")
        inputs.update(context["verified_inputs"])
        if libraries is not None and libraries != context["libraries"]:
            raise ValueError("reference CPU libraries differ between files")
        libraries = context["libraries"]
        for relative, source in dict(context["copy_inputs"], **{path.name: str(path.resolve())}).items():
            if relative in copies and Path(copies[relative]).resolve() != Path(source).resolve():
                raise ValueError("reference dependencies have a path conflict")
            copies[relative] = source
        for model in corpus["models"]:
            ident = model["id"]
            pair = (model, context["bases"][ident])
            if ident in models and models[ident] != pair:
                raise ValueError("reference model differs between files")
            models[ident] = pair
        for sequence in corpus["sequences"]:
            if sequence["id"] in sequences:
                raise ValueError("repeated reference input")
            sequences[sequence["id"]] = sequence
    env, links = library_environment(inputs, libraries, args.executable)
    args.output.mkdir()
    links_path = args.output / "library-links.txt"
    links_path.write_text(links)
    retain_input(inputs, "library_links", links_path)
    jobs = args.output / "jobs"
    jobs.mkdir()
    counts = rows = 0
    commands = []
    for ident, (model, basis) in models.items():
        original = local_file(args.asset_root, basis["runtime_original"]["path"])
        derived = local_file(args.asset_root, basis["reference_derived"]["path"])
        retain_sized(inputs, "original_model:" + ident, original, basis["runtime_original"])
        retain_sized(inputs, "derived_model:" + ident, derived, basis["reference_derived"])
        selected = [s for s in sequences.values() if s["model_id"] == ident]
        lines = []
        for sequence in selected:
            ids = jobs / (sequence["id"] + ".ids")
            with ids.open("x") as out:
                for name, values in (("prefill", sequence["prefill_ids"]), ("generated", sequence["teacher_ids"])):
                    out.write(name + " " + " ".join(map(str, values)) + "\n")
            retain_input(inputs, "tokens:" + sequence["id"], ids)
            lines.append(f"{sequence['id']}\t{sequence['row_count']}\t0\t-\t{ids.resolve()}\n")
        job = jobs / (ident + ".tsv")
        with job.open("x") as out:
            out.write("".join(lines))
        retain_input(inputs, "jobs:" + ident, job)
        generated = args.output / ("generated-" + ident)
        command = [str(args.executable.resolve()), str(derived), str(job.resolve()),
                   str(generated.resolve()), "--fixed-teacher"]
        verify_inputs(inputs)
        with (args.output / (ident + ".log")).open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        commands.append(dict(model=ident, command=command, exit=result.returncode))
        if result.returncode:
            raise ValueError("CPU reference failed: " + ident)
        verify_inputs(inputs)
        retain_input(inputs, "actual_run:" + ident, generated / "run.txt")
        for sequence in selected:
            emitted = dict(sequence, reference_rows_file=sequence["id"] + "/rows.f32")
            metadata = checked_rows(generated, emitted, model["vocab"])
            source = local_file(generated, emitted["reference_rows_file"])
            destination = local_file(args.output, sequence["reference_rows_file"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            token_destination = destination.with_name("tokens.txt")
            if destination.exists() or token_destination.exists():
                raise ValueError("the reference output already exists")
            source.rename(destination)
            metadata.rename(token_destination)
            retain_input(inputs, "rows:" + sequence["id"], destination, sequence["reference_rows_sha256"])
            retain_input(inputs, "metadata:" + sequence["id"], token_destination, sequence["token_metadata_sha256"])
            counts += 1
            rows += sequence["row_count"]
    copy_dependencies(args.output, copies, inputs)
    verify_inputs(inputs)
    write_json(args.output / "generation.json", dict(reference_index_sha256=inputs["reference_index"]["sha256"],
               reference_identity=index["identity"], executable_sha256=executable["sha256"],
               regeneration_build=dict(executable=executable, library_links=inputs["library_links"],
                                       libraries={key: value for key, value in inputs.items() if key.startswith("library:")}),
               mode="fixed_teacher", commands=commands, sequences=counts, rows=rows, verified_inputs=inputs))
    print(f"arch_reference: {counts} sequences, {rows} matching rows, 0 failures, 0 skips")
    return 0


def legacy(args):
    try:
        inputs = {}
        retain_input(inputs, "corpus", args.corpus)
        retain_input(inputs, "executable", args.executable)
        checked_text(str(args.output.resolve()))
        corpus = read_corpus(args.corpus, asset_root=args.asset_root)
        if corpus.get("_reference", {}).get("kind") == "decoded_f32":
            raise ValueError("decoded references require --reference-index")
        libraries, copies = {}, {}
        for library in corpus.get("reference_libraries", []):
            local_file(args.corpus.parent, library["file"])
            path = args.corpus.parent.absolute() / library["file"]
            if path.name in libraries:
                raise ValueError("repeated CPU library name")
            libraries[path.name] = dict(library, path=str(path))
            copies[library["file"]] = str(path)
        env, links = library_environment(inputs, libraries, args.executable) if libraries else (None, None)
        args.output.mkdir()
        if links is not None:
            links_path = args.output / "library-links.txt"
            links_path.write_text(links)
            retain_input(inputs, "library_links", links_path)
        jobs = args.output / "jobs"
        jobs.mkdir()
        counts = rows = 0
        commands = []
        for model in corpus["models"]:
            path = args.models / model["model_file"]
            stored = retain_input(inputs, "model:" + model["id"], path, model["sha256"])
            if model.get("bytes", stored["bytes"]) != stored["bytes"]:
                raise ValueError("model size differs: " + model["id"])
            sequences = [s for s in corpus["sequences"] if s["model_id"] == model["id"]]
            lines = []
            for sequence in sequences:
                ids = jobs / (sequence["id"] + ".ids")
                with ids.open("x") as out:
                    for name, values in (("prefill", sequence["prefill_ids"]), ("generated", sequence["teacher_ids"])):
                        out.write(name + " " + " ".join(map(str, values)) + "\n")
                retain_input(inputs, "tokens:" + sequence["id"], ids)
                lines.append(f"{sequence['id']}\t{sequence['row_count']}\t0\t-\t{ids.resolve()}\n")
            job = jobs / (model["id"] + ".tsv")
            with job.open("x") as out:
                out.write("".join(lines))
            retain_input(inputs, "jobs:" + model["id"], job)
            generated = args.output / ("generated-" + model["id"])
            command = [str(args.executable.resolve()), str(path.resolve()), str(job.resolve()), str(generated.resolve())]
            verify_inputs(inputs)
            with (args.output / (model["id"] + ".log")).open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            commands.append(dict(model=model["id"], command=command, exit=result.returncode))
            if result.returncode != 0:
                raise ValueError("CPU reference failed: " + model["id"])
            verify_inputs(inputs)
            for sequence in sequences:
                source = local_file(generated, sequence["id"] + "/rows.f32")
                generated_sequence = dict(sequence, reference_rows_file=sequence["id"] + "/rows.f32")
                metadata = checked_rows(generated, generated_sequence, model["vocab"])
                row_identity = retain_input(inputs, "rows:" + sequence["id"], source,
                                            sequence["reference_rows_sha256"])
                if row_identity["bytes"] != sequence["row_count"] * model["vocab"] * 4:
                    raise ValueError("CPU reference row digest differs: " + sequence["id"])
                destination = local_file(args.output, sequence["reference_rows_file"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                token_destination = destination.with_name("tokens.txt")
                if destination.exists() or token_destination.exists():
                    raise ValueError("the reference output already exists")
                source.rename(destination)
                metadata.rename(token_destination)
                retain_input(inputs, "rows:" + sequence["id"], destination, sequence["reference_rows_sha256"])
                retain_input(inputs, "metadata:" + sequence["id"], token_destination,
                             sequence["token_metadata_sha256"])
                counts += 1
                rows += sequence["row_count"]
        copy_dependencies(args.output, copies, inputs)
        corpus_copy = args.output / "corpus.json"
        with args.corpus.open("rb") as source, corpus_copy.open("xb") as target:
            shutil.copyfileobj(source, target)
        retain_input(inputs, "corpus_copy", corpus_copy, inputs["corpus"]["sha256"])
        verify_inputs(inputs)
        write_json(args.output / "generation.json", dict(corpus_sha256=inputs["corpus"]["sha256"],
                   executable_sha256=inputs["executable"]["sha256"], commands=commands,
                   sequences=counts, rows=rows, verified_inputs=inputs))
        print(f"arch_reference: {counts} sequences, {rows} matching rows, 0 failures, 0 skips")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("arch_reference: FAILED:", str(error) or type(error).__name__, file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path)
    source.add_argument("--reference-index", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.reference_index:
            if args.asset_root is None or args.models is not None:
                raise ValueError("the reference index requires --asset-root without --models")
            return decoded(args)
        if args.models is None:
            raise ValueError("the corpus requires --models")
        return legacy(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("arch_reference: FAILED:", str(error) or type(error).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
