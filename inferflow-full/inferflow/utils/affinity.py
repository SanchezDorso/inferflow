import os
import subprocess


def _parse_cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return cpus


def physical_gpu_index(local_index: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return local_index
    ids = [int(part) for part in visible.split(",") if part != ""]
    return ids[local_index]


def bind_current_process_to_gpu(physical_index: int) -> None:
    """把当前进程绑到这张卡所在 NUMA 的 CPU 上。跨 NUMA 发 kernel 会多一次跳。"""
    pci = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(physical_index)],
        text=True,
    ).strip().lower()
    if pci.startswith("00000000:"):
        pci = "0000:" + pci.split(":", 1)[1]
    with open(f"/sys/bus/pci/devices/{pci}/local_cpulist") as handle:
        cpus = _parse_cpulist(handle.read())
    if cpus:
        os.sched_setaffinity(0, cpus)
