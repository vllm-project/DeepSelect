import hashlib
import os
import re


DEFAULT_CUDA_ARCH_LIST = "10.0a;10.3a"
SM120_SOURCE = "csrc/cuda_kernels/sm120/topk_select.cu"


def parse_cuda_arch_list(requested, cuda_version):
    aliases = {
        "10.0": "100a", "10.0a": "100a", "100a": "100a",
        "10.3": "103a", "10.3a": "103a", "103a": "103a",
        "10.0f": "100f", "100f": "100f",
        "12.0": "120", "120": "120",
        "12.1": "121", "121": "121",
    }
    requested = DEFAULT_CUDA_ARCH_LIST if requested is None else requested
    entries = re.split(r"[;\s]+", requested.strip())
    selected = {}
    for entry in entries:
        ptx = entry.endswith("+PTX")
        base = entry[:-4] if ptx else entry
        if base not in aliases:
            raise ValueError(f"Unsupported DEEP_SELECT_CUDA_ARCH_LIST entry: {entry!r}")
        arch = aliases[base]
        minimum = (12, 9) if arch in {"103a", "100f", "121"} else (12, 8)
        if cuda_version < minimum:
            raise RuntimeError(
                f"{entry} requires CUDA {minimum[0]}.{minimum[1]} or newer"
            )
        selected[arch] = selected.get(arch, False) or ptx
    return tuple(sorted(selected.items()))


def architecture_flags(architectures):
    flags = []
    for arch, ptx in architectures:
        flags += ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
        if ptx:
            flags += ["-gencode", f"arch=compute_{arch},code=compute_{arch}"]
    return flags


def source_group(source):
    path = source.replace(os.sep, "/")
    if path.endswith("/api.cpp"):
        return "host"
    if path.endswith("/cuda_kernels/sm120/topk_select.cu"):
        return "sm120"
    if any(f"/cuda_kernels/{version}/instantiations/" in path
           for version in ("v3", "v3_fp32", "v3_cluster")) and path.endswith(".cu"):
        return "sm100"
    raise ValueError(f"No DeepSelect compilation group for {source}")


def build_plan(cuda_sources, architectures):
    sm100 = tuple(item for item in architectures if item[0] in {"100a", "103a", "100f"})
    sm12 = tuple(item for item in architectures if item[0] in {"120", "121"})
    macros = [("DEEP_SELECT_BUILD_SM100", str(int(bool(sm100)))),
              ("DEEP_SELECT_BUILD_SM120", str(int(any(arch == "120" for arch, _ in sm12)))),
              ("DEEP_SELECT_BUILD_SM121", str(int(any(arch == "121" for arch, _ in sm12))))]
    sources = [source for source in cuda_sources
               if source_group(source) == "host" or sm100]
    if sm12:
        sources.append(SM120_SOURCE)
    flags = {
        "host": [],
        "sm100": architecture_flags(sm100),
        "sm120": architecture_flags(sm12),
    }
    return sources, macros, flags


def grouped_build_extension(base_class, group_flags):
    class GroupedBuildExtension(base_class):
        def finalize_options(self):
            super().finalize_options()
            self.force = True

        def build_extension(self, ext):
            compile_objects = self.compiler.compile

            def compile_groups(sources, output_dir=None, macros=None,
                               include_dirs=None, debug=0, extra_preargs=None,
                               extra_postargs=None, depends=None):
                objects = []
                for group in ("host", "sm100", "sm120"):
                    group_sources = [source for source in sources
                                     if source_group(source) == group]
                    if not group_sources:
                        continue
                    flags = {key: list(value) for key, value in extra_postargs.items()}
                    flags["nvcc"] += group_flags[group]
                    signature = hashlib.sha256(
                        repr((macros, flags, include_dirs, extra_preargs, debug)).encode()
                    ).hexdigest()[:16]
                    objects += compile_objects(
                        group_sources,
                        output_dir=os.path.join(output_dir, f"{group}-{signature}"),
                        macros=macros, include_dirs=include_dirs, debug=debug,
                        extra_preargs=extra_preargs, extra_postargs=flags, depends=depends,
                    )
                return objects

            self.compiler.compile = compile_groups
            try:
                super().build_extension(ext)
            finally:
                self.compiler.compile = compile_objects

    return GroupedBuildExtension
