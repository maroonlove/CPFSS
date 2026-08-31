from __future__ import annotations

import math
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterable, List, Set

BIOLOGICAL_PROCESS = "GO:0008150"
MOLECULAR_FUNCTION = "GO:0003674"
CELLULAR_COMPONENT = "GO:0005575"

FUNC_DICT = {
    "bp": BIOLOGICAL_PROCESS,
    "mf": MOLECULAR_FUNCTION,
    "cc": CELLULAR_COMPONENT,
}

NAMESPACES = {
    "bp": "biological_process",
    "mf": "molecular_function",
    "cc": "cellular_component",
}

NAMESPACES_REVERT = {v: k for k, v in NAMESPACES.items()}
ROOT_GO_TERMS = set(FUNC_DICT.values())


class Ontology:
    """Minimal Gene Ontology parser used for ancestor closure and IC metrics."""

    def __init__(self, filename: str | Path, with_rels: bool = True) -> None:
        self.ont = self._load(filename, with_rels=with_rels)
        self.ic: Dict[str, float] | None = None
        self.icdepth: Dict[str, float] | None = None

    def has_term(self, term_id: str) -> bool:
        return term_id in self.ont

    def get_namespace(self, term_id: str) -> str:
        return self.ont[term_id]["namespace"]

    def get_namespace_terms(self, namespace: str) -> Set[str]:
        return {go_id for go_id, obj in self.ont.items() if obj.get("namespace") == namespace}

    def get_parents(self, term_id: str) -> Set[str]:
        if term_id not in self.ont:
            return set()
        return {p for p in self.ont[term_id]["is_a"] if p in self.ont}

    def get_anchestors(self, term_id: str) -> Set[str]:
        """Return ancestors including the term itself. Kept name for compatibility."""
        if term_id not in self.ont:
            return set()
        term_set: Set[str] = set()
        q: deque[str] = deque([term_id])
        while q:
            t_id = q.popleft()
            if t_id in term_set:
                continue
            term_set.add(t_id)
            for parent_id in self.ont[t_id]["is_a"]:
                if parent_id in self.ont:
                    q.append(parent_id)
        return term_set

    def get_depth(self, term_id: str, ont: str) -> int:
        q: deque[str] = deque([term_id])
        layer = 1
        while q:
            parents: Set[str] = set()
            while q:
                parents.update(self.get_parents(q.popleft()))
            if parents:
                layer += 1
                for item in parents:
                    if item == FUNC_DICT[ont]:
                        return layer
                    q.append(item)
        return layer

    def calculate_ic(self, annots: Iterable[Iterable[str]]) -> None:
        cnt: Counter[str] = Counter()
        for terms in annots:
            cnt.update(terms)

        self.ic = {}
        self.icdepth = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if not parents:
                min_n = n
            else:
                min_n = min(cnt[p] for p in parents if p in cnt) if any(p in cnt for p in parents) else n
            self.ic[go_id] = math.log(min_n / n, 2) if n > 0 else 0.0
            namespace = NAMESPACES_REVERT.get(self.get_namespace(go_id))
            depth = self.get_depth(go_id, namespace) if namespace else 1
            self.icdepth[go_id] = math.log(max(depth, 1), 2) * self.ic[go_id]

    def get_ic(self, go_id: str) -> float:
        if self.ic is None:
            raise RuntimeError("IC has not been calculated. Call calculate_ic() first.")
        return float(self.ic.get(go_id, 0.0))

    def get_icdepth(self, go_id: str) -> float:
        if self.icdepth is None:
            raise RuntimeError("IC-depth has not been calculated. Call calculate_ic() first.")
        return float(self.icdepth.get(go_id, 0.0))

    def _load(self, filename: str | Path, with_rels: bool) -> Dict[str, dict]:
        ont: Dict[str, dict] = {}
        obj = None
        with open(filename, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    if obj is not None and "id" in obj:
                        ont[obj["id"]] = obj
                    obj = {"is_a": [], "part_of": [], "alt_ids": [], "is_obsolete": False}
                    continue
                if line == "[Typedef]":
                    if obj is not None and "id" in obj:
                        ont[obj["id"]] = obj
                    obj = None
                    continue
                if obj is None or ": " not in line:
                    continue
                key, value = line.split(": ", 1)
                if key == "id":
                    obj["id"] = value
                elif key == "alt_id":
                    obj["alt_ids"].append(value)
                elif key == "namespace":
                    obj["namespace"] = value
                elif key == "is_a":
                    obj["is_a"].append(value.split(" ! ")[0])
                elif with_rels and key == "relationship":
                    parts = value.split()
                    if len(parts) >= 2 and parts[0] == "part_of":
                        obj["is_a"].append(parts[1])
                elif key == "name":
                    obj["name"] = value
                elif key == "is_obsolete" and value == "true":
                    obj["is_obsolete"] = True
        if obj is not None and "id" in obj:
            ont[obj["id"]] = obj

        for term_id in list(ont.keys()):
            if ont[term_id].get("is_obsolete"):
                del ont[term_id]
                continue
            for alt_id in ont[term_id].get("alt_ids", []):
                ont[alt_id] = ont[term_id]

        for term_id, val in ont.items():
            val.setdefault("children", set())
            for parent_id in val.get("is_a", []):
                if parent_id in ont:
                    ont[parent_id].setdefault("children", set()).add(term_id)
        return ont
