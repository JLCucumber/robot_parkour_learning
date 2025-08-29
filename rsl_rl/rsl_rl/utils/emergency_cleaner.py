#!/usr/bin/env python3
"""
Emergency trajectory cleaner for disk quota exceeded situations.
This is a one-time cleaner that removes old pickle files to free up space.
"""
import os
import re
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
TRAJECTORY_PATTERN = re.compile(r"trajectory_(\d+)")
"""Configuration parameters (can be overridden via environment variables):
EMERGENCY_DELETE_COUNT: max trajectory directories (oldest first) to process per sub-dir.
EMERGENCY_THRESHOLD: minimum number of trajectory dirs (with pickle files) that triggers action.
MIN_FREE_SPACE_GB: if free space >= this value, skip entire cleanup.
MIN_FREE_SPACE_RATIO: if free/total >= ratio, skip.
DELETE_WORKERS: thread pool workers for file deletion (I/O bound).
LOG_FILE_PREFIX: prefix for persistent log file (append mode). One log file per day.
"""

EMERGENCY_DELETE_COUNT = int(os.getenv("EMERGENCY_DELETE_COUNT", 3000))
EMERGENCY_THRESHOLD = int(os.getenv("EMERGENCY_THRESHOLD", 1000))
MIN_FREE_SPACE_GB = float(os.getenv("MIN_FREE_SPACE_GB", 5))  # 5 GB
MIN_FREE_SPACE_RATIO = float(os.getenv("MIN_FREE_SPACE_RATIO", 0.05))  # 5%
DELETE_WORKERS = int(os.getenv("EMERGENCY_DELETE_WORKERS", 8))
LOG_FILE_PREFIX = os.getenv("EMERGENCY_CLEAN_LOG_PREFIX", "emergency_cleanup")

def setup_emergency_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """Setup logger for emergency cleaning.

    Adds both console and (persistent) file handler. File name pattern:
      <log_dir>/<LOG_FILE_PREFIX>.log  (rotated daily by date suffix).
    If log_dir is None or unwritable, falls back to current working directory.
    """
    logger = logging.getLogger('emergency_cleaner')
    logger.setLevel(logging.INFO)

    # Always keep a simple formatter
    formatter = logging.Formatter("%(asctime)s - EMERGENCY - %(levelname)s - %(message)s")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if log_dir is None:
        log_dir = os.getcwd()
    else:
        os.makedirs(log_dir, exist_ok=True)

    # Daily file (same prefix)
    date_tag = time.strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"{LOG_FILE_PREFIX}_{date_tag}.log")
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_path) for h in logger.handlers):
        try:
            fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            logger.warning("Failed to attach file handler, continuing with console only.")
    logger.debug(f"Logger initialized. Log file: {log_path}")
    return logger

def disk_space_ok(path: str) -> Tuple[bool, Dict[str, float]]:
    """Check if disk free space is already sufficient.

    Returns (ok_to_skip, metrics_dict)
    metrics_dict keys: free_gb, total_gb, used_gb, free_ratio
    """
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_ratio = usage.free / usage.total if usage.total else 0.0
        ok = (free_gb >= MIN_FREE_SPACE_GB) and (free_ratio >= MIN_FREE_SPACE_RATIO)
        return ok, {
            'free_gb': free_gb,
            'total_gb': total_gb,
            'used_gb': used_gb,
            'free_ratio': free_ratio
        }
    except Exception:
        # If we cannot determine disk status, force cleanup attempt.
        return False, {'free_gb': -1, 'total_gb': -1, 'used_gb': -1, 'free_ratio': -1}

def get_trajectory_folders_with_pickle(directory: str) -> List[Tuple[int, str]]:
    """
    Scans a directory for trajectory folders that contain pickle files.
    Returns them sorted by number, oldest first.
    """
    trajectories = []
    try:
        for entry in os.scandir(directory):
            if entry.is_dir():
                match = TRAJECTORY_PATTERN.match(entry.name)
                if match:
                    traj_num = int(match.group(1))
                    # Check if this trajectory folder contains any pickle files
                    try:
                        files_in_dir = os.listdir(entry.path)
                        has_pickle_files = any(f.endswith('.pickle') or f.endswith('.pkl') for f in files_in_dir)
                        if has_pickle_files:
                            trajectories.append((traj_num, entry.path))
                    except (OSError, FileNotFoundError):
                        continue
    except FileNotFoundError:
        return []
    
    # Sort by trajectory number (oldest first)
    trajectories.sort(key=lambda x: x[0])
    return trajectories

def _collect_deletion_targets(trajectories_to_delete: list) -> Tuple[list, list]:
    """Collect all pickle files and directories to process.
    Returns (file_paths, dir_paths)
    """
    file_paths = []
    dir_paths = []
    for _traj_num, traj_path in trajectories_to_delete:
        if not os.path.exists(traj_path):
            continue
        try:
            files_in_dir = os.listdir(traj_path)
        except OSError:
            continue
        pickle_files = [os.path.join(traj_path, f) for f in files_in_dir if f.endswith('.pickle') or f.endswith('.pkl')]
        if pickle_files:
            file_paths.extend(pickle_files)
        dir_paths.append(traj_path)
    return file_paths, dir_paths

def _delete_file(path: str) -> int:
    """Delete a single file and return bytes freed (0 if fail)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    try:
        os.remove(path)
        return size
    except OSError:
        return 0

def emergency_clean_directory(base_data_dir: str, logger) -> Tuple[bool, int]:
    """Emergency cleaning of trajectory directories with parallel deletion.

    Returns (performed_any_deletion, bytes_freed)
    """
    if not os.path.isdir(base_data_dir):
        logger.error(f"Base data directory not found: {base_data_dir}")
        return False, 0

    total_pickle_files_deleted = 0
    total_dirs_processed = 0
    total_bytes_freed = 0

    logger.info(f"🚨 Starting EMERGENCY cleanup in: {base_data_dir}")

    try:
        for task_entry in os.scandir(base_data_dir):
            if not task_entry.is_dir():
                continue
            task_name = task_entry.name
            task_path = task_entry.path
            logger.info(f"🔍 Task: {task_name}")

            for sub_dir_entry in os.scandir(task_path):
                if not sub_dir_entry.is_dir():
                    continue
                sub_dir_path = sub_dir_entry.path
                sub_dir_name = Path(sub_dir_path).name

                all_trajectories = get_trajectory_folders_with_pickle(sub_dir_path)
                total_count = len(all_trajectories)
                if total_count <= EMERGENCY_THRESHOLD:
                    continue
                trajectories_to_delete = all_trajectories[:EMERGENCY_DELETE_COUNT]
                if not trajectories_to_delete:
                    continue
                first_traj = trajectories_to_delete[0][0]
                last_traj = trajectories_to_delete[-1][0]
                logger.info(
                    f"📁 {task_name}/{sub_dir_name}: {total_count} trajectories; deleting {len(trajectories_to_delete)} (trajectory_{first_traj}..{last_traj})"
                )

                file_paths, dir_paths = _collect_deletion_targets(trajectories_to_delete)
                logger.info(f"➡️  Collected {len(file_paths)} pickle files across {len(dir_paths)} dirs for deletion")

                # Parallel delete pickle files
                if file_paths:
                    with ThreadPoolExecutor(max_workers=DELETE_WORKERS) as ex:
                        futures = {ex.submit(_delete_file, fp): fp for fp in file_paths}
                        for fut in as_completed(futures):
                            freed = fut.result()
                            if freed > 0:
                                total_pickle_files_deleted += 1
                                total_bytes_freed += freed

                # Post-process directories: remove if empty / only npz
                for d in dir_paths:
                    try:
                        remaining = os.listdir(d)
                    except OSError:
                        continue
                    if not remaining:
                        try:
                            os.rmdir(d)
                            total_dirs_processed += 1
                        except OSError:
                            pass
                    elif all(f.endswith('.npz') or f.startswith('.') for f in remaining):
                        total_dirs_processed += 1

                logger.info(
                    f"✅ {task_name}/{sub_dir_name}: deleted_files={total_pickle_files_deleted} dirs_processed={total_dirs_processed} freed={total_bytes_freed/1024**3:.3f} GB"
                )

    except Exception as e:
        logger.error(f"❌ Emergency cleanup failed: {e}")
        return False, total_bytes_freed

    logger.info("🎉 Emergency cleanup completed")
    logger.info(f"📊 Total pickle files deleted: {total_pickle_files_deleted}")
    logger.info(f"📊 Total directories processed: {total_dirs_processed}")
    logger.info(f"📊 Total bytes freed: {total_bytes_freed} ({total_bytes_freed/1024**3:.3f} GB)")

    return total_pickle_files_deleted > 0, total_bytes_freed

def run_emergency_cleanup(base_data_dir: Optional[str] = None):
    """Run emergency cleanup once.

    Returns dict with summary:
      {
        'success': bool,
        'bytes_freed': int,
        'free_space_before_gb': float,
        'free_space_after_gb': float,
        'skipped_due_to_space_ok': bool
      }
    """
    if base_data_dir is None:
        base_data_dir = "/cs/student/projects2/rai/2024/hongboli/network_test/data"

    logger = setup_emergency_logger(base_data_dir)
    logger.info("🚨🚨🚨 EMERGENCY TRAJECTORY CLEANUP STARTED 🚨🚨🚨")
    logger.info(f"Target directory: {base_data_dir}")
    logger.info(f"Emergency threshold: {EMERGENCY_THRESHOLD}; delete cap: {EMERGENCY_DELETE_COUNT}")
    logger.info(f"Free space skip criteria: free >= {MIN_FREE_SPACE_GB} GB AND ratio >= {MIN_FREE_SPACE_RATIO*100:.1f}%")

    ok_to_skip, metrics_before = disk_space_ok(base_data_dir)
    logger.info(
        f"Disk before -> free: {metrics_before['free_gb']:.2f} GB / total {metrics_before['total_gb']:.2f} GB (ratio {metrics_before['free_ratio']*100:.2f}%)"
    )
    if ok_to_skip:
        logger.info("✅ Sufficient free space; skipping cleanup.")
        return {
            'success': True,
            'bytes_freed': 0,
            'free_space_before_gb': metrics_before['free_gb'],
            'free_space_after_gb': metrics_before['free_gb'],
            'skipped_due_to_space_ok': True
        }

    success, bytes_freed = emergency_clean_directory(base_data_dir, logger)
    _, metrics_after = disk_space_ok(base_data_dir)
    logger.info(
        f"Disk after  -> free: {metrics_after['free_gb']:.2f} GB / total {metrics_after['total_gb']:.2f} GB (ratio {metrics_after['free_ratio']*100:.2f}%)"
    )

    if success:
        logger.info("✅ Emergency cleanup completed (work performed)")
    else:
        logger.warning("⚠️  Emergency cleanup completed with no deletions")

    return {
        'success': success,
        'bytes_freed': bytes_freed,
        'free_space_before_gb': metrics_before['free_gb'],
        'free_space_after_gb': metrics_after['free_gb'],
        'skipped_due_to_space_ok': False
    }

if __name__ == "__main__":
    import sys
    
    # Allow command line argument for base directory
    base_dir = sys.argv[1] if len(sys.argv) > 1 else None
    summary = run_emergency_cleanup(base_dir)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
