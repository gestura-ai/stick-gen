#!/usr/bin/env python3
"""
Quick script to check training status by monitoring system resources
and checking for new checkpoint files.
"""

import os
import time
import psutil
from datetime import datetime

def check_training_status():
    print("=" * 60)
    print("TRAINING STATUS CHECK")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check for Python processes
    print("Python Processes:")
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                python_procs.append(proc)
                cmdline = ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else 'N/A'
                print(f"  PID {proc.info['pid']}: CPU={proc.info['cpu_percent']:.1f}% "
                      f"MEM={proc.info['memory_percent']:.1f}% CMD={cmdline}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if not python_procs:
        print("  ⚠️  No Python processes found!")
    
    # Check checkpoint files
    print("\nCheckpoint Files:")
    checkpoint_files = []
    for f in os.listdir('.'):
        if f.endswith('.pth'):
            stat = os.stat(f)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            size_mb = stat.st_size / (1024 * 1024)
            checkpoint_files.append((f, mtime, size_mb))
    
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    for fname, mtime, size_mb in checkpoint_files[:5]:
        age = datetime.now() - mtime
        age_str = f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m ago" if age.days == 0 else f"{age.days}d ago"
        print(f"  {fname}: {size_mb:.1f}MB (modified {age_str})")
    
    # Check data files
    print("\nData Files:")
    data_dir = 'data'
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.pt'):
                fpath = os.path.join(data_dir, f)
                stat = os.stat(fpath)
                size_gb = stat.st_size / (1024 * 1024 * 1024)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                print(f"  {f}: {size_gb:.2f}GB (modified {mtime.strftime('%H:%M:%S')})")
    
    # System resources
    print("\nSystem Resources:")
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    print(f"  CPU Usage: {cpu_percent}%")
    print(f"  Memory: {mem.percent}% ({mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB)")
    
    print("\n" + "=" * 60)
    
    # Estimate if training is running
    if python_procs and any(p.info['cpu_percent'] > 50 for p in python_procs):
        print("✅ Training appears to be RUNNING (high CPU usage detected)")
    elif python_procs:
        print("⚠️  Python process found but low CPU - may be loading data or between epochs")
    else:
        print("❌ No training process detected")
    
    print("=" * 60)

if __name__ == "__main__":
    check_training_status()

