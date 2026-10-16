import os
import secrets
import stat
import struct
import time
import zlib
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Callable, Optional

import pynvml


class DegradationLevel(IntEnum):
    NORMAL = 0  # 正常運行
    REDUCED = 1  # 減少緩衝區 (Level 1, >85% VRAM)
    FAST_PATH = 2  # 僅保留 Perception (Level 2, >92% VRAM)
    EMERGENCY = 3  # 低解析度模式 (Level 3, >96% VRAM)


@dataclass
class VRAMStats:
    free_mb: float
    used_mb: float
    total_mb: float
    usage_percent: float


class ResourceManager:
    """
    Saccade 資源管理器 (Industrial Grade Decision Maker)

    1. 實時監控 VRAM 狀態。
    2. 實作階梯式降級 (Stepped Degradation) 邏輯。
    3. 防止系統在臨界點發生抖動 (Hysteresis)。
    """

    def __init__(self, gpu_id: int = 0) -> None:
        self.gpu_id = gpu_id
        self._nvml_initialized = False
        self._init_nvml()

        # 狀態紀錄，用於實現 Hysteresis (避免在 89/91% 之間反覆切換)
        self.current_level = DegradationLevel.NORMAL

    def _init_nvml(self) -> None:
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception as e:
            print(f"⚠️ [ResourceManager] NVML Init failed: {e}. Degradation disabled.")

    def get_stats(self) -> VRAMStats:
        if not self._nvml_initialized:
            return VRAMStats(0.0, 0.0, 0.0, 0.0)

        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage = (info.used / info.total) * 100
        return VRAMStats(
            free_mb=info.free / 1024 / 1024,
            used_mb=info.used / 1024 / 1024,
            total_mb=info.total / 1024 / 1024,
            usage_percent=usage,
        )

    def decide_degradation_level(self) -> DegradationLevel:
        """
        根據當前資源決定降級級別 (具備 Hysteresis 遲滯保護)
        """
        stats = self.get_stats()
        usage = stats.usage_percent

        # 門檻值定義 (上升)
        UP_EMERGENCY = 96
        UP_FAST_PATH = 92
        UP_REDUCED = 85

        # 遲滯緩衝 (5%)
        HYSTERESIS = 5

        new_level = self.current_level

        # 1. 昇級邏輯 (負載增加)
        if usage > UP_EMERGENCY:
            new_level = DegradationLevel.EMERGENCY
        elif usage > UP_FAST_PATH:
            new_level = DegradationLevel.FAST_PATH
        elif usage > UP_REDUCED:
            new_level = DegradationLevel.REDUCED

        # 2. 降階邏輯 (負載減少) - 只有當負載下降超過 HYSTERESIS 時才回歸
        if self.current_level == DegradationLevel.EMERGENCY and usage < (
            UP_EMERGENCY - HYSTERESIS
        ):
            new_level = DegradationLevel.FAST_PATH
        elif self.current_level == DegradationLevel.FAST_PATH and usage < (
            UP_FAST_PATH - HYSTERESIS
        ):
            new_level = DegradationLevel.REDUCED
        elif self.current_level == DegradationLevel.REDUCED and usage < (
            UP_REDUCED - HYSTERESIS
        ):
            new_level = DegradationLevel.NORMAL

        if new_level != self.current_level:
            print(
                f"⚙️ [ResourceManager] System Scaling: {self.current_level.name} -> {new_level.name} (Usage: {usage:.1f}%)"
            )
            self.current_level = new_level

        return self.current_level

    def close(self) -> None:
        if self._nvml_initialized:
            pynvml.nvmlShutdown()


# ── VRAM-level snapshot (#537 B0-F07) ────────────────────────────────────────
# The writer publishes each snapshot as a complete small file under /dev/shm:
# write a private temp file, then rename() it over the public name. A reader
# opens the name, preads the whole record and closes it, so every read sees
# either the whole old snapshot or the whole new one, follows a restarted
# writer on its next read, and holds nothing that could keep a dead inode alive.
# No SharedMemory object is used on either side, so no resource_tracker can
# unlink another process's snapshot.
#
# Freshness means "this level was decided at most max_age_ns ago" on
# CLOCK_BOOTTIME (which keeps counting through suspend). Reader and writer must
# share a time namespace. The record has no meaning across mixed old/new
# versions: stop both sides (and their resource_tracker helpers) before
# upgrading or rolling back.
_SHM_DIR = "/dev/shm"
_SHM_NAME = "saccade_vram_level"
# level, version, pad (zero), crc32 (of all 24 bytes with this field zeroed),
# generation, decided_at_ns
_SNAPSHOT = struct.Struct("<BB2xIQQ")
_SNAPSHOT_SIZE = _SNAPSHOT.size
_SNAPSHOT_VERSION = 1
_TMP_INFIX = ".tmp."
DEFAULT_VRAM_LEVEL_MAX_AGE_NS = 1_000_000_000


def vram_level_clock_ns() -> int:
    """The clock a VRAM-level decision is stamped with and aged against."""
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def _snapshot_path() -> str:
    return os.path.join(_SHM_DIR, _SHM_NAME)


def _record_crc(record: bytes) -> int:
    return zlib.crc32(record[:4] + b"\0\0\0\0" + record[8:])


def _pack_snapshot(level: int, generation: int, decided_at_ns: int) -> bytes:
    unsigned = _SNAPSHOT.pack(level, _SNAPSHOT_VERSION, 0, generation, decided_at_ns)
    crc = _record_crc(unsigned)
    return _SNAPSHOT.pack(level, _SNAPSHOT_VERSION, crc, generation, decided_at_ns)


class VRAMReadStatus(Enum):
    FRESH = "fresh"  # valid record, decided within max_age_ns
    STALE = "stale"  # valid record, but no decision yet or decided too long ago
    MISSING = "missing"  # no published snapshot
    INVALID = "invalid"  # unreadable, malformed, foreign version or future stamp


@dataclass(frozen=True)
class VRAMLevelSnapshot:
    status: VRAMReadStatus
    level: Optional[DegradationLevel]  # set only when status is FRESH
    generation: Optional[int] = None
    age_ns: Optional[int] = None
    reason: str = ""


class VRAMLevelWriter:
    """Dispatcher 端：以原子 rename 發佈降級等級快照（單一 writer）。"""

    def __init__(self) -> None:
        self._path = _snapshot_path()
        self.generation = secrets.randbits(63) | 1
        self._tmp = f"{self._path}{_TMP_INFIX}{os.getpid()}.{self.generation:x}"
        self._closed = False
        self._write_failed = False
        _remove_orphan_temps(self._path)
        # A valid record with no decision: readers see STALE, never a leftover
        # level from a previous writer.
        self._publish(_pack_snapshot(0, self.generation, 0))

    def write(self, level: DegradationLevel, decided_at_ns: int) -> bool:
        """Publish `level`, decided at `decided_at_ns` (`vram_level_clock_ns()`
        read before the decision). Returns False if publication failed; readers
        then age the previous snapshot into STALE."""
        level = DegradationLevel(level)
        if decided_at_ns <= 0:
            raise ValueError(f"decided_at_ns must be positive, got {decided_at_ns}")
        try:
            self._publish(_pack_snapshot(int(level), self.generation, decided_at_ns))
        except OSError as e:
            if not self._write_failed:
                print(f"⚠️ [VRAMLevelWriter] publish failed: {e}")
            self._write_failed = True
            return False
        self._write_failed = False
        return True

    def _publish(self, record: bytes) -> None:
        if self._closed:
            raise OSError("VRAMLevelWriter is closed")
        fd = os.open(
            self._tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            try:
                if os.write(fd, record) != len(record):
                    raise OSError("short write")
            finally:
                os.close(fd)
            os.replace(self._tmp, self._path)
        except BaseException:
            _unlink_quietly(self._tmp)
            raise

    def close(self) -> None:
        """Retire this writer: replace its own snapshot with a no-decision record.

        The public name is never removed. If a successor publishes between the
        ownership check and the replacement (two live writers, unsupported),
        its snapshot reads STALE until its next decision: never MISSING and
        never a wrong FRESH level."""
        if self._closed:
            return
        try:
            if self._owns_public_snapshot():
                self._publish(_pack_snapshot(0, self.generation, 0))
        except OSError:
            pass
        finally:
            self._closed = True
            _unlink_quietly(self._tmp)

    def _owns_public_snapshot(self) -> bool:
        snapshot = VRAMLevelReader(max_age_ns=0).read_snapshot()
        return snapshot.generation == self.generation

    def __enter__(self) -> "VRAMLevelWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class VRAMLevelReader:
    """Orchestrator 端：每次讀取都依名稱開檔讀完整快照；不持有任何資源。"""

    def __init__(
        self,
        max_age_ns: int = DEFAULT_VRAM_LEVEL_MAX_AGE_NS,
        clock: Callable[[], int] = vram_level_clock_ns,
    ) -> None:
        self._max_age_ns = max_age_ns
        self._clock = clock

    def read_snapshot(self) -> VRAMLevelSnapshot:
        """Never raises: every failure is a non-FRESH status."""
        try:
            return self._read()
        except Exception as e:  # noqa: BLE001 - a read must not abort a batch
            return VRAMLevelSnapshot(
                VRAMReadStatus.INVALID, None, reason=f"unexpected {type(e).__name__}"
            )

    def _read(self) -> VRAMLevelSnapshot:
        try:
            # O_NONBLOCK: a FIFO planted at the name must not block the loop.
            fd = os.open(
                _snapshot_path(),
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
        except FileNotFoundError:
            return VRAMLevelSnapshot(VRAMReadStatus.MISSING, None, reason="no snapshot")
        except OSError as e:
            return VRAMLevelSnapshot(
                VRAMReadStatus.INVALID, None, reason=f"open {type(e).__name__}"
            )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return VRAMLevelSnapshot(
                    VRAMReadStatus.INVALID, None, reason="not a regular file"
                )
            data = os.pread(fd, _SNAPSHOT_SIZE + 1, 0)
        finally:
            os.close(fd)

        if len(data) != _SNAPSHOT_SIZE:
            return VRAMLevelSnapshot(
                VRAMReadStatus.INVALID, None, reason=f"size {len(data)}"
            )
        level, version, crc, generation, decided_at_ns = _SNAPSHOT.unpack(data)
        if version != _SNAPSHOT_VERSION:
            return VRAMLevelSnapshot(
                VRAMReadStatus.INVALID, None, reason=f"version {version}"
            )
        if data[2:4] != b"\0\0":
            return VRAMLevelSnapshot(VRAMReadStatus.INVALID, None, reason="padding")
        if crc != _record_crc(data):
            return VRAMLevelSnapshot(VRAMReadStatus.INVALID, None, reason="crc")
        if level > DegradationLevel.EMERGENCY:
            return VRAMLevelSnapshot(
                VRAMReadStatus.INVALID, None, generation, reason=f"level {level}"
            )
        if decided_at_ns == 0:
            return VRAMLevelSnapshot(
                VRAMReadStatus.STALE, None, generation, reason="no decision yet"
            )
        age_ns = self._clock() - decided_at_ns
        if age_ns < 0:
            return VRAMLevelSnapshot(
                VRAMReadStatus.INVALID, None, generation, age_ns, "future timestamp"
            )
        if age_ns > self._max_age_ns:
            return VRAMLevelSnapshot(
                VRAMReadStatus.STALE, None, generation, age_ns, "expired"
            )
        return VRAMLevelSnapshot(
            VRAMReadStatus.FRESH, DegradationLevel(level), generation, age_ns
        )


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _remove_orphan_temps(path: str) -> None:
    """Remove temp files left by writers that died between create and rename."""
    prefix = os.path.basename(path) + _TMP_INFIX
    try:
        names = os.listdir(os.path.dirname(path))
    except OSError:
        return
    for name in names:
        if not name.startswith(prefix):
            continue
        pid_text = name[len(prefix) :].split(".", 1)[0]
        try:
            os.kill(int(pid_text), 0)
            continue  # owner still alive
        except ProcessLookupError:
            pass
        except (ValueError, PermissionError):
            continue
        _unlink_quietly(os.path.join(os.path.dirname(path), name))


if __name__ == "__main__":
    manager = ResourceManager()
    stats = manager.get_stats()
    level = manager.decide_degradation_level()
    print(f"Stats: {stats}")
    print(f"Selected Level: {level.name}")
    manager.close()
