"""
Facade re-exporting split grounding modules (backward compatible).
"""
from __future__ import annotations

import importlib

import core.episode_phase as _episode_phase
from core._grounding_merge import merge_module_exports
from core.scienceworld_schema import *  # noqa: F403
from core.action_primitives import *  # noqa: F403
from core.episode_phase import *  # noqa: F403
from core.substitute_engine import *  # noqa: F403
from core.episode_phase import (
    _context_has_visible_fixtures,
    _fixture_associated_rooms,
    _focused_substance_source_fixtures,
    _manipulation_delivery_action_re,
    _pour_source_aligns_with_focus,
    _prep_collect_pour_min_score,
    _task_manipulation_destination_fixture_words,
    _task_manipulation_destination_rooms,
    _transfer_shuffle_destination_fixtures,
)

# Star import skips leading-underscore helpers; merge all grounding submodules.
for _mod_name in (
    "core.scienceworld_schema",
    "core.action_primitives",
    "core.navigation_helpers",
    "core.episode_phase",
    "core.substitute_engine",
):
    merge_module_exports(globals(), importlib.import_module(_mod_name))

# Ensure episode_phase can resolve symbols from later-loaded modules when imported directly.
for _mod_name in (
    "core.action_primitives",
    "core.navigation_helpers",
    "core.substitute_engine",
):
    merge_module_exports(_episode_phase, importlib.import_module(_mod_name))

# Public aliases for core.task_semantics
context_has_visible_fixtures = _context_has_visible_fixtures
task_manipulation_destination_fixture_words = _task_manipulation_destination_fixture_words
fixture_associated_rooms = _fixture_associated_rooms
task_manipulation_destination_rooms = _task_manipulation_destination_rooms
focused_substance_source_fixtures = _focused_substance_source_fixtures
pour_source_aligns_with_focus = _pour_source_aligns_with_focus
prep_collect_pour_min_score = _prep_collect_pour_min_score
manipulation_delivery_action_re = _manipulation_delivery_action_re
transfer_shuffle_destination_fixtures = _transfer_shuffle_destination_fixtures
