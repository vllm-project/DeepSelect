import ast
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from csrc.build_utils import architecture_flags, build_plan, parse_cuda_arch_list


ROOT = Path(__file__).resolve().parents[1]


def setup_namespace():
    tree = ast.parse((ROOT / "setup.py").read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "build_on_cuda_platform"
             or isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == "CUDA_SOURCES"
                 for target in node.targets)]
    namespace = {"os": os, "subprocess": subprocess, "Path": Path,
                 "__file__": str(ROOT / "setup.py"),
                 "kk": SimpleNamespace(check_kernel_reg_spill_in_artifact=Mock(return_value=[]))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "setup.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("entry", ["12.1", "121", "12.1+PTX", "121+PTX"])
def test_sm121_minimum_and_ptx(entry):
    with pytest.raises(RuntimeError, match="requires CUDA 12.9"):
        parse_cuda_arch_list(entry, (12, 8))
    architectures = parse_cuda_arch_list(entry, (12, 9))
    assert architectures == (("121", entry.endswith("+PTX")),)
    flags = architecture_flags(architectures)
    assert "arch=compute_121,code=sm_121" in flags
    assert ("arch=compute_121,code=compute_121" in flags) == entry.endswith("+PTX")


@pytest.mark.parametrize("entry", ["", "12.1a", "12.1f", "12.1+ptx", "9.0;12.1"])
def test_invalid_architecture(entry):
    with pytest.raises(ValueError):
        parse_cuda_arch_list(entry, (13, 2))


def test_default_and_deduplication():
    assert parse_cuda_arch_list(None, (12, 9)) == (("100a", False), ("103a", False))
    assert parse_cuda_arch_list("121;12.1+PTX 121", (12, 9)) == (("121", True),)


@pytest.mark.parametrize("entry,expected_macros,counts", [
    ("12.0", (0, 1, 0), [1, 1]),
    ("12.1+PTX", (0, 0, 1), [1, 1]),
    ("12.0;12.1+PTX", (0, 1, 1), [1, 1]),
    ("10.0a;10.3a;12.0;12.1", (1, 1, 1), [1, 78, 1]),
    ("10.0f", (1, 0, 0), [1, 78]),
])
def test_single_extension_grouped_compile(monkeypatch, tmp_path, entry, expected_macros, counts):
    import torch.utils.cpp_extension as cpp
    from setuptools import Distribution
    from setuptools._distutils.ccompiler import new_compiler
    from setuptools._distutils.sysconfig import customize_compiler

    namespace = setup_namespace()
    architectures = parse_cuda_arch_list(entry, (12, 9))
    sources, macros, _ = build_plan(namespace["CUDA_SOURCES"], architectures)
    assert tuple(int(value) for _, value in macros) == expected_macros
    assert sources.count("csrc/api.cpp") == 1
    monkeypatch.setenv("DEEP_SELECT_CUDA_ARCH_LIST", entry)
    monkeypatch.setattr(subprocess, "check_output", Mock(return_value=b"CUDA release 12.9, V12.9.0"))
    monkeypatch.setattr(cpp, "CUDA_HOME", str(tmp_path))
    extensions, cls = namespace["build_on_cuda_platform"]()
    assert len(extensions) == 1
    extension = extensions[0]
    assert extension.name == "deep_select.deep_select_cuda"
    assert extension.py_limited_api
    command = cls(Distribution({"ext_modules": extensions}))
    command.ensure_finalized()
    command.build_temp = str(tmp_path / "objects")
    command.build_lib = str(tmp_path / "lib")
    command.compiler = new_compiler()
    customize_compiler(command.compiler)
    monkeypatch.setattr(command, "_check_abi", lambda: ("g++", cpp.TorchVersion("13.0")))
    monkeypatch.setattr(cpp, "_check_cuda_version", Mock())
    compile_mock = Mock()
    link_mock = Mock()
    monkeypatch.setattr(cpp, "_write_ninja_file_and_compile_objects", compile_mock)
    monkeypatch.setattr(command.compiler, "link_shared_object", link_mock)
    command.build_extensions()
    calls = [call.kwargs for call in compile_mock.call_args_list]
    assert [len(call["sources"]) for call in calls] == counts
    assert link_mock.call_count == 1
    assert len(link_mock.call_args.args[0]) == sum(counts)
    assert command.get_ext_filename(extension.name).endswith(".abi3.so")
    assert "-DPy_LIMITED_API=0x030A0000" in calls[0]["post_cflags"]
    for call in calls[1:]:
        flags = call["cuda_post_cflags"]
        assert "-DTORCH_TARGET_VERSION=0x020a000000000000" in flags
        assert flags.count("--use_fast_math") == 1
        assert flags.count("--ftz=false") == 1
        assert flags.index("--use_fast_math") < flags.index("--ftz=false")
        if "/sm120/" in call["sources"][0]:
            expected = tuple(item for item in architectures if item[0] in {"120", "121"})
        else:
            expected = tuple(item for item in architectures if item[0] not in {"120", "121"})
        actual = [flag for flag in flags if flag.startswith("arch=compute_")]
        assert actual == [flag for flag in architecture_flags(expected) if flag.startswith("arch=")]
    monkeypatch.setattr(cpp.BuildExtension, "run", Mock())
    command.run()
    checker = namespace["kk"].check_kernel_reg_spill_in_artifact
    checker.assert_called_once_with(command.get_ext_fullpath(extension.name), stack_baseline=8,
                                    suppress_checking_env_var="DEEP_SELECT_DISABLE_REG_SPILL_CHECK")
    checker.return_value = ["spilling_kernel"]
    with pytest.raises(RuntimeError, match="Register spilling detected"):
        command.run()
