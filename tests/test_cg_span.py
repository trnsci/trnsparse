"""The fused CG / power kernels' reads of A's blocks at the 4 GiB rule (#121).

`blocks_t` (128, n_blocks * 128 * parts) has memory rows of 32 MiB or more
once fp32 A stores 65,536 blocks. Device reads over 128 partitions with that
step are misdirected on Trn2 (aws-neuron-sdk#1457), so the kernels read such
a blocks_t in row blocks of at most 64 partitions. CPU checks of the rule and
of the kernel source; the device checks are in tests/test_nki_cg_fused.py.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from trnsparse.nki import kernels as K

MIB = 1 << 20


@pytest.mark.parametrize(
    "row_bytes,rows",
    [
        (512, 128),  # one fp32 block
        (65535 * 512, 128),  # 65,535 fp32 blocks: 32 MiB - 512 B
        (65536 * 512, 64),  # 65,536 fp32 blocks: 32 MiB, 128 * row = 2**32
        (32 * MIB + 16 * 1024, 64),  # 64 rows span exactly 2 GiB + 1 MiB
        (32 * MIB + 16 * 1024 + 4, 32),
        (64 * MIB, 32),
        (64 * MIB + 32 * 1024 + 1, 16),
        (262144 * 512, 16),  # T = 512 dense: every block stored (128 MiB rows)
        (131071 * 256, 128),  # bf16 A: 131,071 blocks
        (131072 * 256, 64),  # bf16 A: 131,072 blocks
        (3 << 31, 1),  # a 6 GiB row: one partition per read
    ],
)
def test_dma_row_block_rule(row_bytes, rows):
    got = K.dma_row_block(128, row_bytes)
    assert got == rows
    if got < 128:
        assert 128 % got == 0 and got <= 64
        assert got == 1 or got * row_bytes <= K._DMA_BLOCK_BYTES
        assert 128 * row_bytes >= 1 << 32


def test_dma_row_block_fewer_rows():
    # The rule counts min(rows, 128) partitions.
    assert K.dma_row_block(32, 64 * MIB) == 32
    assert K.dma_row_block(64, 64 * MIB) == 32  # 64 rows of 64 MiB span 4 GiB
    assert K.dma_row_block(128, 64 * MIB) == 32


def _cg_body_tree():
    src = K.__file__
    with open(src) as f:
        mod = ast.parse(f.read())
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == "_cg_bsr_body":
            return node
    raise AssertionError("_cg_bsr_body not found")


def test_cg_kernel_reads_blocks_only_in_row_blocks():
    """Every read of blocks_t in the CG / power body goes through load_blocks,
    which splits it into row blocks above the rule (#121). Through 0.15.0 the
    body read blocks_t over all 128 partitions at every size."""
    body = _cg_body_tree()
    loader = next(
        (n for n in ast.walk(body) if isinstance(n, ast.FunctionDef) and n.name == "load_blocks"),
        None,
    )
    assert loader is not None, "_cg_bsr_body has no load_blocks"
    inside = {id(n) for n in ast.walk(loader)}
    reads = [
        n
        for n in ast.walk(body)
        if isinstance(n, ast.Subscript)
        and isinstance(n.value, ast.Name)
        and n.value.id == "blocks_t"
    ]
    assert reads, "no blocks_t reads found"
    outside = [ast.unparse(n) for n in reads if id(n) not in inside]
    assert not outside, f"blocks_t read outside load_blocks: {outside}"
    # The split reads slice the partition axis by the row block.
    split = [ast.unparse(n) for n in reads if "a_rows" in ast.unparse(n.slice)]
    assert split, "load_blocks never reads a row block of blocks_t"


def test_dma_row_block_is_plain_python():
    """The rule runs on the host at trace time, so it must not need NKI."""
    assert "nl." not in textwrap.dedent(inspect.getsource(K.dma_row_block))
