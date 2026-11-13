"""Planning and applying folder tidy-ups, tested with a scripted model on temporary folders."""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from test_graph import ScriptedModel

from agent.config import Settings
from agent.organize import (
    APPLIED_FILE,
    MOVES_FILE,
    apply_moves,
    build_organize_graph,
    clean_folder_name,
    initial_organize_state,
    parse_assignments,
    parse_layout,
    save_moves,
    scan,
    undo_moves,
    why_not,
)

SETTINGS = Settings()
# Sorted the way scan() lists them: logo.png, notes.txt, photo.jpg, sales.csv
FILES = {"sales.csv": "month,total\n", "notes.txt": "Call the venue", "photo.jpg": "jpg", "logo.png": "png"}
LAYOUT = "docs: written documents\nimages: pictures\ndata: spreadsheets and exports"
ASSIGN = "1: images\n2: docs\n3: images\n4: data"


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "messy"
    root.mkdir()
    for name, text in FILES.items():
        (root / name).write_text(text)
    return root


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


async def plan(folder, tmp_path, replies=(LAYOUT, ASSIGN), settings=SETTINGS, start=None):
    model = ScriptedModel(replies=[AIMessage(r) for r in replies], calls=[])
    graph = build_organize_graph(model, settings)
    state = await graph.ainvoke(initial_organize_state("Sort by type", folder, settings) | (start or {}))
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    save_moves(run_dir, state)
    return state, run_dir, model


async def test_planning_changes_nothing_and_applying_only_moves(folder, tmp_path):
    before = snapshot(folder)
    state, run_dir, _ = await plan(folder, tmp_path)
    assert snapshot(folder) == before, "planning must not touch the folder"
    assert state["status"] == "done"
    assert "## images/ (new folder, 2 files)" in state["report"]

    result = apply_moves(run_dir)
    assert result.moved == 4 and result.skipped == []
    after = snapshot(folder)
    assert set(after) == {"images/logo.png", "docs/notes.txt", "images/photo.jpg", "data/sales.csv"}
    assert sorted(after.values()) == sorted(before.values()), "every file survives unchanged"


async def test_undo_restores_the_folder_exactly(folder, tmp_path):
    before = snapshot(folder)
    _, run_dir, _ = await plan(folder, tmp_path)
    apply_moves(run_dir)
    result = undo_moves(run_dir)
    assert result.moved == 4
    assert snapshot(folder) == before
    assert [p.name for p in folder.iterdir() if p.is_dir()] == [], "the folders it made are removed"
    assert not (run_dir / APPLIED_FILE).exists()


async def test_a_plan_cannot_be_applied_twice(folder, tmp_path):
    _, run_dir, _ = await plan(folder, tmp_path)
    apply_moves(run_dir)
    with pytest.raises(SystemExit, match="already applied"):
        apply_moves(run_dir)


async def test_unclear_answers_leave_files_where_they_are(folder, tmp_path):
    assign = "1: images\n2: KEEP\n3: somewhere else\n9: docs\nno idea about the rest"
    state, _, _ = await plan(folder, tmp_path, replies=(LAYOUT, assign))
    assert [move["from"] for move in state["moves"]] == ["logo.png"]
    assert "## Left where they are (3 files)" in state["report"]


async def test_files_are_handled_in_batches_with_fresh_context(folder, tmp_path):
    settings = replace(SETTINGS, organize_batch=3)
    state, _, model = await plan(folder, tmp_path, replies=(LAYOUT, "1: images\n2: docs\n3: images", "4: data"), settings=settings)
    assert len(state["moves"]) == 4
    assert len(model.calls) == 3 and all(len(messages) == 2 for messages in model.calls)
    assert "4. sales.csv" in model.calls[2][1].content and "logo.png" not in model.calls[2][1].content


async def test_existing_folders_are_reused_and_clashes_are_skipped(folder, tmp_path):
    (folder / "Images").mkdir()
    (folder / "Images" / "logo.png").write_text("an older logo")
    (folder / "Images" / "deep.txt").write_text("subfolder contents are never listed")
    state, run_dir, model = await plan(folder, tmp_path)
    assert "Existing subfolders: Images" in model.calls[0][1].content
    assert "deep.txt" not in model.calls[0][1].content
    assert {move["to"] for move in state["moves"]} == {"Images/photo.jpg", "docs/notes.txt", "data/sales.csv"}
    assert "logo.png: Images already has a file with that name" in state["report"]
    apply_moves(run_dir)
    assert (folder / "Images" / "logo.png").read_text() == "an older logo"
    assert (folder / "logo.png").read_text() == "png"


async def test_apply_skips_files_that_changed_or_would_overwrite(folder, tmp_path):
    _, run_dir, _ = await plan(folder, tmp_path)
    (folder / "notes.txt").write_text("Edited after the plan was made")
    (folder / "sales.csv").unlink()
    (folder / "images").mkdir()
    (folder / "images" / "photo.jpg").write_text("appeared after the plan")
    result = apply_moves(run_dir)
    assert result.moved == 1
    assert len(result.skipped) == 3
    assert (folder / "notes.txt").exists()
    assert (folder / "images" / "photo.jpg").read_text() == "appeared after the plan"
    assert (folder / "photo.jpg").read_text() == "jpg"


async def test_an_edited_plan_cannot_reach_outside_the_folder(folder, tmp_path):
    _, run_dir, _ = await plan(folder, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    plan_data = json.loads((run_dir / MOVES_FILE).read_text())
    stat = (folder / "notes.txt").stat()
    plan_data["moves"] = [
        {"from": "notes.txt", "to": "../stolen.txt", "size": stat.st_size, "mtime": stat.st_mtime},
        {"from": "../outside.txt", "to": "docs/outside.txt", "size": 7, "mtime": outside.stat().st_mtime},
    ]
    (run_dir / MOVES_FILE).write_text(json.dumps(plan_data))
    result = apply_moves(run_dir)
    assert result.moved == 0 and len(result.skipped) == 2
    assert outside.exists() and not (tmp_path / "stolen.txt").exists()


async def test_time_limit_leaves_the_rest_in_place(folder, tmp_path):
    settings = replace(SETTINGS, organize_batch=2, max_minutes=1)
    state, _, _ = await plan(
        folder, tmp_path, replies=(LAYOUT, "1: images\n2: docs"), settings=settings,
        start={"started_at": time.time() - 3600},
    )
    assert state["status"] == "budget_exhausted"
    assert len(state["moves"]) == 2


def test_scan_lists_only_loose_ordinary_files(folder):
    (folder / ".hidden").write_text("x")
    (folder / "desktop.ini").write_text("x")
    (folder / "sub").mkdir()
    (folder / "sub" / "inner.txt").write_text("x")
    try:
        os.symlink(folder / "notes.txt", folder / "link.txt")
    except OSError:
        pass  # Creating symlinks needs extra permissions on this machine.
    files, folders, left_out = scan(folder, limit=3)
    assert [f["name"] for f in files] == ["logo.png", "notes.txt", "photo.jpg"]
    assert folders == ["sub"] and left_out == 1
    assert files[1]["preview"] == "Call the venue" and files[0]["preview"] == ""


def test_folder_names_from_the_model_are_made_safe():
    assert clean_folder_name(" **Docs** ") == "Docs"
    assert clean_folder_name("Documents/Invoices/") == "Documents/Invoices"
    assert clean_folder_name("../../Windows") == "Windows"
    assert clean_folder_name("C:\\Users\\me") is None
    assert clean_folder_name("a/b/c") is None
    assert clean_folder_name("NUL") is None
    assert clean_folder_name("..") is None
    assert clean_folder_name('bad<>:"|?*name') == "badname"


def test_layout_and_assignments_are_parsed_leniently():
    layout = parse_layout("1. **docs**: papers\n- Images/: pictures\n- images: again\nKEEP: nothing\n: empty", ["Images"], 5)
    assert [(f["name"], f["exists"]) for f in layout] == [("docs", False), ("Images", True)]
    chosen = parse_assignments("1. a.txt -> docs\n2: `images`\n3 = Docs.\n4: KEEP\n5: docs\n2: docs", 0, 4, layout)
    assert chosen == {"0": "docs", "1": "Images", "2": "docs"}


def test_risky_folders_are_refused(tmp_path):
    assert "drive" in why_not(Path(tmp_path.anchor))
    assert "home" in why_not(Path.home().resolve())
    (tmp_path / ".git").mkdir()
    assert "repository" in why_not(tmp_path)
    (tmp_path / "ok").mkdir()
    assert why_not(tmp_path / "ok") is None
