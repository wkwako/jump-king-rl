import pymem
import struct
import ctypes
import ctypes.wintypes




#finds addresses
pm = pymem.Pymem("JumpKing.exe")

#gets base
base = pm.base_address
print(f"Base: {hex(base)}")

MEM_COMMIT = 0x1000
PAGE_NOACCESS = 0x01

class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_ulonglong),
        ("AllocationBase", ctypes.c_ulonglong),
        ("AllocationProtect", ctypes.wintypes.DWORD),
        ("RegionSize", ctypes.c_ulonglong),
        ("State", ctypes.wintypes.DWORD),
        ("Protect", ctypes.wintypes.DWORD),
        ("Type", ctypes.wintypes.DWORD),
    ]

def scan_float(target, tolerance=0.05):
    results = []
    address = 0
    mbi = MEMORY_BASIC_INFORMATION()

    while ctypes.windll.kernel32.VirtualQueryEx(pm.process_handle, ctypes.c_ulonglong(address), ctypes.byref(mbi), ctypes.sizeof(mbi)):
        if mbi.State == MEM_COMMIT and mbi.Protect != PAGE_NOACCESS:
            try:
                chunk = pm.read_bytes(mbi.BaseAddress, mbi.RegionSize)
                for i in range(0, len(chunk) - 4, 4):
                    val = struct.unpack('f', chunk[i:i+4])[0]
                    if abs(val - target) < tolerance:
                        results.append((mbi.BaseAddress + i, val))
            except:
                pass
        address = mbi.BaseAddress + mbi.RegionSize
        if address > 0x7FFFFFFFFFFF:
            break

    return results

results = scan_float(313.0000)  # replace with your X coordinate
print(f"Found {len(results)} matches")

#narrows addresses
previous_results = results  # save the first scan results

# replace with your new X coordinate after moving
new_x = 323.5000

narrowed = []
for addr, old_val in previous_results:
    try:
        val = struct.unpack('f', pm.read_bytes(addr, 4))[0]
        if abs(val - new_x) < 0.05:
            narrowed.append((addr, val))
    except:
        pass

print(f"Narrowed to {len(narrowed)} matches")
for addr, val in narrowed:
    print(hex(addr), val)

#narrows addresses down again
previous_results = narrowed

# replace with your new X coordinate
new_x = 317.5000

narrowed2 = []
for addr, old_val in previous_results:
    try:
        val = struct.unpack('f', pm.read_bytes(addr, 4))[0]
        if abs(val - new_x) < 0.05:
            narrowed2.append((addr, val))
    except:
        pass

print(f"Narrowed to {len(narrowed2)} matches")
for addr, val in narrowed2:
    print(hex(addr), val)

#checks if y is near x
for addr, val in narrowed2:
    try:
        x = struct.unpack('f', pm.read_bytes(addr, 4))[0]
        y = struct.unpack('f', pm.read_bytes(addr + 4, 4))[0]
        y_minus = struct.unpack('f', pm.read_bytes(addr - 4, 4))[0]
        print(f"{hex(addr)}: X={x:.4f}, next={y:.4f}, prev={y_minus:.4f}")
    except:
        pass

#checks for address survival after jump king restart
for addr in [0xf45b3fefd0, 0xf45b3ff288, 0x2cfa0c721a0, 0x2cfa14fcc78]:
    try:
        x = struct.unpack('f', pm.read_bytes(addr, 4))[0]
        y = struct.unpack('f', pm.read_bytes(addr + 4, 4))[0]
        print(f"{hex(addr)}: X={x:.4f}, Y={y:.4f}")
    except Exception as e:
        print(f"{hex(addr)}: failed - {e}")


base = 0x26a63160000

addresses = [0x8699dfedf0, 0x26a665e21a0, 0x26a66e6c9a0]

for addr in addresses:
    diff = addr - base
    print(f"{hex(addr)}: offset = {hex(diff)}")