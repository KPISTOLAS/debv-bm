"""Prompt strings for closed-set CLIP/BioCLIP argmax.

Class *ids* and YAML names stay unchanged. Aliases apply only when building text.
FIR ``Bird`` is a taxon dump, not a species: keep the class; do not treat Bird
errors as evidence that Path 1 failed.
"""

from __future__ import annotations

# Prompt-only. Integer labels and YAML names are not rewritten on disk.
PROMPT_ALIASES = {
    "camelus": "camel",
    "Camelus": "camel",
    "Flying-squirrel": "flying squirrel",
    "flying-squirrel": "flying squirrel",
}

JUNK_PLANT_PLACEHOLDERS = {"bh", "bl", "dd", "hs", "htq", "lys", "sql", "ss", "ys"}


def yaml_names(raw) -> list[str]:
    if isinstance(raw, dict):
        keys = sorted(raw.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        return [str(raw[k]) for k in keys]
    return [str(x) for x in raw]


def prompt_phrase(class_name: str) -> str:
    if class_name in PROMPT_ALIASES:
        return PROMPT_ALIASES[class_name]
    return class_name.replace("_", " ").replace("-", " ").strip()


def prompt_for_class(class_name: str, modality: str = "rgb") -> str:
    phrase = prompt_phrase(class_name)
    if modality == "thermal":
        return f"a thermal infrared image of a {phrase}"
    return f"a photo of a {phrase}"


def refuse_placeholder_plants(names: list[str]) -> None:
    hit = [n for n in names if n.lower() in JUNK_PLANT_PLACEHOLDERS]
    if hit:
        raise SystemExit(
            "Plant names look like plants.example.yaml placeholders "
            f"{hit}. Re-download Tree Species Identification v2 and use export names. "
            "This is not a BioCLIP/CLIP test."
        )
