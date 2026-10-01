"""Process launcher. Spawns one rank per usable GPU and wires up torch.distributed.

Single GPU / CPU -> runs in-process, no distributed init at all, with all three
agents owned by the one rank.
"""
from __future__ import annotations

import os
import socket
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..config import Config
from ..resources import Plan
from .loop import Trainer


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _worker(rank: int, cfg_d: dict, plan_d: dict, root: str, addr: str, port: int):
    try:
        cfg = Config(**{})
        cfg = _cfg_from_dict(cfg_d)
        plan = _plan_from_dict(plan_d)
        torch.set_num_threads(max(1, plan.cpu_threads))
        if plan.world_size > 1:
            os.environ.setdefault("MASTER_ADDR", addr)
            os.environ.setdefault("MASTER_PORT", str(port))
            backend = plan.backend if plan.backend in ("nccl", "gloo") else "gloo"
            if backend == "nccl" and not torch.cuda.is_available():
                backend = "gloo"
            dist.init_process_group(backend=backend, rank=rank,
                                    world_size=plan.world_size,
                                    init_method=f"tcp://{addr}:{port}")
        if plan.devices[rank].startswith("cuda"):
            torch.cuda.set_device(plan.devices[rank])
        tr = Trainer(cfg, plan, rank, root=root)
        if dist.is_initialized():
            dist.barrier()
        tr.run()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        print(f"[rank {rank}] FAILED\n{traceback.format_exc()}", flush=True)
        raise


def _cfg_from_dict(d: dict) -> Config:
    from ..config import DataConfig, EnvConfig, ModelConfig, PPOConfig, TrainConfig
    return Config(data=DataConfig(**d["data"]), model=ModelConfig(**d["model"]),
                  env=EnvConfig(**d["env"]), ppo=PPOConfig(**d["ppo"]),
                  train=TrainConfig(**d["train"]))


def _plan_from_dict(d: dict) -> Plan:
    from ..resources import GPUInfo
    p = Plan(world_size=d["world_size"], backend=d["backend"],
             device_type=d["device_type"], size_preset=d["size_preset"],
             roles=d["roles"], devices=d["devices"], cpu_threads=d["cpu_threads"],
             gpus=[GPUInfo(**g) for g in d["gpus"]], host_ram_gb=d["host_ram_gb"],
             host_cpus=d["host_cpus"], notes=d["notes"])
    return p


def launch(cfg: Config, plan: Plan, root: str = "."):
    torch.set_num_threads(max(1, plan.cpu_threads))
    if plan.world_size == 1:
        tr = Trainer(cfg, plan, 0, root=root)
        tr.run()
        return
    addr, port = "127.0.0.1", _free_port()
    mp.set_start_method("spawn", force=True)
    mp.spawn(_worker, args=(cfg.to_dict(), plan.to_dict(), root, addr, port),
             nprocs=plan.world_size, join=True)
