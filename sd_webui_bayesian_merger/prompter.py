import os
from typing import Dict, List

import random
import yaml

from pathlib import Path

PathT = os.PathLike | str


class CardDealer:
    def __init__(self, wildcards_dir: str):
        self.find_wildcards(wildcards_dir)

    def find_wildcards(self, wildcards_dir: str) -> None:
        wdir = Path(wildcards_dir)
        if wdir.exists():
            self.wildcards = {w.stem: w for w in wdir.glob("*.txt")}
        else:
            self.wlldcards: Dict[str, PathT] = {}

    def sample_wildcard(self, wildcard_name: str) -> str:
        if wildcard_name in self.wildcards:
            with open(
                self.wildcards[wildcard_name],
                "r",
                encoding="utf-8",
            ) as f:
                lines = f.readlines()
                return random.choice(lines).strip()

        # TODO raise warning?
        return wildcard_name

    def replace_wildcards(self, prompt: str) -> str:
        chunks = prompt.split("__")
        replacements = [self.sample_wildcard(w) for w in chunks[1::2]]
        chunks[1::2] = replacements
        return "".join(chunks)


def load_yaml(yaml_file: PathT) -> Dict:
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_payload(payload: Dict, additional_defaults: Dict) -> Dict:
    if "prompt" not in payload:
        raise ValueError(f"{payload['path']} doesn't have a prompt")

    # set defaults
    # TODO: get default values from args
    for k, v in {
        "neg_prompt": "",
        "seed": -1,
        "steps": 20,
        "cfg": 7,
        "width": 512,
        "height": 512,
        "sampler": "Euler",
        "batch_size": 1,
        "batch_count": 1,
    }.items():
        if k not in payload:
            payload[k] = v

    return payload


class Prompter:
    def __init__(self, payloads_dir: str, wildcards_dir: str, batch_size: str):
        self.find_payloads(payloads_dir)
        self.load_payloads()
        self.dealer = CardDealer(wildcards_dir)
        self.batch_size = batch_size

    def find_payloads(self, payloads_dir: str) -> None:
        # TODO: allow for listing payloads instead of taking all of them
        pdir = Path(payloads_dir)
        if pdir.exists():
            self.raw_payloads = {
                p.stem: {"path": p}
                for p in pdir.glob("*.yaml")
                if ".tmpl.yaml" not in p.name
            }

        else:
            # TODO: pick a better error
            raise ValueError("payloads directory not found!")

    def load_payloads(self) -> None:
        for payload_name, payload in self.raw_payloads.items():
            raw_payload = load_yaml(payload["path"])
            checked_payload = check_payload(raw_payload, {'batch_size': self.batch_size})
            self.raw_payloads[payload_name].update(checked_payload)

    def render_payloads(self) -> List[Dict]:
        payloads = []
        for _, p in self.raw_payloads.items():
            rendered_payload = p.copy()
            rendered_payload["prompt"] = self.dealer.replace_wildcards(p["prompt"])
            rendered_payload.pop("path")
            payloads.append(rendered_payload)
        return payloads
