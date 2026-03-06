"""Tests for drp_sim._constants — pure Python, no Isaac Sim required."""

from __future__ import annotations

import os
import tempfile

from drp_sim._constants import (
    _PACKAGE_PREFIX,
    _PACKAGE_ROOT,
    _PROCESSED_URDF,
    _URDF_PATH,
    N_JOINTS,
    preprocess_urdf,
)


class TestConstants:
    def test_n_joints(self) -> None:
        assert N_JOINTS == 5

    def test_package_prefix(self) -> None:
        assert _PACKAGE_PREFIX == "package://dsr_description2/"

    def test_package_root_is_string(self) -> None:
        assert isinstance(_PACKAGE_ROOT, str)
        assert len(_PACKAGE_ROOT) > 0

    def test_urdf_path_ends_correctly(self) -> None:
        assert _URDF_PATH.endswith("p3020.urdf")

    def test_processed_urdf_path(self) -> None:
        assert _PROCESSED_URDF.endswith(".urdf")


class TestPreprocessUrdf:
    def test_replaces_package_prefix(self) -> None:
        src_content = '<mesh filename="package://dsr_description2/meshes/link_1.stl"/>'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as src:
            src.write(src_content)
            src_path = src.name
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as dst:
            dst_path = dst.name

        try:
            preprocess_urdf(src_path, dst_path)
            with open(dst_path) as f:
                result = f.read()
            assert "package://dsr_description2/" not in result
            assert _PACKAGE_ROOT in result
            assert result.endswith('meshes/link_1.stl"/>')
        finally:
            os.unlink(src_path)
            os.unlink(dst_path)

    def test_no_change_when_no_prefix(self) -> None:
        src_content = '<robot name="p3020"><link name="base_link"/></robot>'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as src:
            src.write(src_content)
            src_path = src.name
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as dst:
            dst_path = dst.name

        try:
            preprocess_urdf(src_path, dst_path)
            with open(dst_path) as f:
                result = f.read()
            assert result == src_content
        finally:
            os.unlink(src_path)
            os.unlink(dst_path)
