"""Keep Windows awake only while a selected worker process is alive."""

from __future__ import annotations

import argparse
import ctypes
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
SYNCHRONIZE = 0x00100000
INFINITE = 0xFFFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int)
    args = parser.parse_args()
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.SetThreadExecutionState.argtypes = (ctypes.c_uint32,)
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
    handle = kernel32.OpenProcess(SYNCHRONIZE, 0, args.pid)
    if not handle:
        print(f"WORKER_ALREADY_STOPPED pid={args.pid}", flush=True)
        return 2
    try:
        if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
            print("SLEEP_INHIBIT_FAILED", flush=True)
            return 1
        print(f"AWAKE_HOLD_STARTED pid={args.pid}", flush=True)
        kernel32.WaitForSingleObject(ctypes.c_void_p(handle), INFINITE)
        print(f"AWAKE_HOLD_FINISHED pid={args.pid}", flush=True)
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        kernel32.CloseHandle(ctypes.c_void_p(handle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
