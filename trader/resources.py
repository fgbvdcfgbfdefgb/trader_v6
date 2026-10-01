"""Machine inspection + automatic training-plan selection.

Runs *before* anything is allocated. Decides:
  * how many processes to spawn (one per usable GPU, else 1 CPU process)
  * which agent each rank owns
  * model preset from the smallest GPU's free VRAM
  * CPU thread budget that leaves headroom for a co-resident workload (e.g. a miner)

Everything degrades gracefully: 4 GPUs -> 4 ranks, 1 GPU -> 1 rank with all three
agents time-sliced on that device, no GPU -> CPU with the 'tiny' preset.
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from typing import Dict, List

import torch

AGENTS = ["predictor", "analyst", "trader"]


@dataclass
class GPUInfo:
    index: int
    name: str
    total_gb: float
    free_gb: float
    used_gb: float
    capability: str
    busy: bool = False          # something else (mining/other tenant) already on it


@dataclass
class Plan:
    world_size: int
    backend: str
    device_type: str
    size_preset: str
    roles: List[List[str]] = field(default_factory=list)   # roles[rank] -> agents owned
    devices: List[str] = field(default_factory=list)
    cpu_threads: int = 1
    gpus: List[GPUInfo] = field(default_factory=list)
    host_ram_gb: float = 0.0
    host_cpus: int = 0
    notes: List[str] = field(default_factory=list)

    def owners_of(self, agent: str) -> List[int]:
        return [r for r, rs in enumerate(self.roles) if agent in rs]

    def to_dict(self) -> dict:
        d = dict(world_size=self.world_size, backend=self.backend,
                 device_type=self.device_type, size_preset=self.size_preset,
                 roles=self.roles, devices=self.devices, cpu_threads=self.cpu_threads,
                 host_ram_gb=round(self.host_ram_gb, 1), host_cpus=self.host_cpus,
                 notes=self.notes,
                 gpus=[vars(g) for g in self.gpus])
        return d

    def describe(self) -> str:
        L = ["=" * 68, "  TRADER_V6  resource plan", "=" * 68,
             f"  host        : {platform.node()}  "
             f"{self.host_cpus} vCPU / {self.host_ram_gb:.1f} GB RAM  "
             f"(using {self.cpu_threads} torch threads)"]
        if self.gpus:
            for g in self.gpus:
                flag = "  [busy - shared]" if g.busy else ""
                L.append(f"  gpu {g.index}       : {g.name}  "
                         f"{g.free_gb:.1f}/{g.total_gb:.1f} GB free  sm_{g.capability}{flag}")
        else:
            L.append("  gpu         : none detected -> CPU training")
        L.append(f"  world size  : {self.world_size} process(es), backend={self.backend}")
        L.append(f"  model preset: {self.size_preset}")
        for r, (dev, roles) in enumerate(zip(self.devices, self.roles)):
            L.append(f"    rank {r} on {dev:<7} owns: {', '.join(roles)}")
        for n in self.notes:
            L.append(f"  note        : {n}")
        L.append("=" * 68)
        return "\n".join(L)


def _host_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 0.0


def probe_gpus() -> List[GPUInfo]:
    out: List[GPUInfo] = []
    if not torch.cuda.is_available():
        return out
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        try:
            free_b, total_b = torch.cuda.mem_get_info(i)
        except Exception:
            total_b = p.total_memory
            free_b = total_b - torch.cuda.memory_reserved(i)
        free_gb, total_gb = free_b / 1024 ** 3, total_b / 1024 ** 3
        used_gb = total_gb - free_gb
        out.append(GPUInfo(index=i, name=p.name, total_gb=total_gb, free_gb=free_gb,
                           used_gb=used_gb, capability=f"{p.major}{p.minor}",
                           busy=used_gb > 0.25))
    return out


def pick_preset(free_gb: float) -> str:
    if free_gb >= 22:
        return "large"
    if free_gb >= 11:
        return "base"
    if free_gb >= 5.5:
        return "small"
    return "tiny"


def assign_roles(world_size: int) -> List[List[str]]:
    """Round-robin the three agents over ranks.

    1 rank  -> [[predictor, analyst, trader]]            (time-sliced, same device)
    2 ranks -> [[predictor, analyst], [trader]]
    3 ranks -> one agent each
    4 ranks -> one agent each + a second trader rank (PPO is the throughput bottleneck,
               the duplicate owners all-reduce their gradients like DDP)
    """
    if world_size == 1:
        return [list(AGENTS)]
    if world_size == 2:
        return [["predictor", "analyst"], ["trader"]]
    roles: List[List[str]] = [[a] for a in AGENTS]
    extra_order = ["trader", "predictor", "analyst"]
    for i in range(3, world_size):
        roles.append([extra_order[(i - 3) % 3]])
    return roles


def build_plan(force_gpus: int = -1, force_size: str = "auto",
               cpu_threads: int = 0, reserve_cores: int = 1) -> Plan:
    gpus = probe_gpus()
    if force_gpus >= 0:
        gpus = gpus[:force_gpus]

    host_cpus = os.cpu_count() or 1
    ram = _host_ram_gb()
    notes: List[str] = []

    if gpus:
        usable = [g for g in gpus if g.free_gb >= 2.0]
        if len(usable) < len(gpus):
            notes.append(f"{len(gpus) - len(usable)} GPU(s) skipped: <2 GB free VRAM")
        if not usable:
            usable = gpus[:1]
        world = len(usable)
        devices = [f"cuda:{g.index}" for g in usable]
        min_free = min(g.free_gb for g in usable)
        preset = force_size if force_size != "auto" else pick_preset(min_free)
        backend = "nccl" if world > 1 else "single"
        if any(g.busy for g in usable):
            notes.append("GPU(s) already in use by another workload - training shares "
                         "the device and will not evict it")
    else:
        world, devices, preset = 1, ["cpu"], (force_size if force_size != "auto" else "tiny")
        backend = "single"
        notes.append("no CUDA device: CPU fallback, expect ~20x slower epochs")

    if cpu_threads > 0:
        threads = cpu_threads
    else:
        # Leave at least `reserve_cores` for anything else running on the box.
        threads = max(1, (host_cpus - reserve_cores) // max(1, world))
    if host_cpus <= 4:
        notes.append(f"only {host_cpus} vCPU: dataloading is in-process and memory-mapped "
                     "to stay off the CPU")

    return Plan(world_size=world, backend=backend,
                device_type="cuda" if gpus else "cpu", size_preset=preset,
                roles=assign_roles(world), devices=devices, cpu_threads=threads,
                gpus=gpus, host_ram_gb=ram, host_cpus=host_cpus, notes=notes)


if __name__ == "__main__":
    p = build_plan()
    print(p.describe())
    print(json.dumps(p.to_dict(), indent=2))
