#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from collections import defaultdict

def print_folder_tree(directory, prefix="", is_last=True, show_size=False, max_depth=None, current_depth=0):
    """只显示文件夹树和统计信息，不列出具体文件"""
    path = Path(directory)
    if not path.exists():
        print(f"路径不存在: {directory}")
        return
    
    # 检查深度限制
    if max_depth is not None and current_depth > max_depth:
        return
    
    try:
        # 获取所有项目
        all_items = list(path.iterdir())
        files = [item for item in all_items if item.is_file()]
        dirs = [item for item in all_items if item.is_dir()]
        
        # 计算总大小（如果需要）
        total_size = 0
        if show_size:
            for file in files:
                try:
                    total_size += file.stat().st_size
                except (OSError, PermissionError):
                    pass
        
        # 打印当前目录信息
        connector = "└── " if is_last else "├── "
        size_info = f" ({format_size(total_size)})" if show_size and total_size > 0 else ""
        
        if current_depth == 0:
            # 根目录显示
            print(f"{path.name}/ [{len(files)} 文件, {len(dirs)} 文件夹{size_info}]")
        else:
            print(f"{prefix}{connector}{path.name}/ [{len(files)} 文件, {len(dirs)} 文件夹{size_info}]")
        
        # 只处理子文件夹，按名称排序
        dirs.sort(key=lambda x: x.name.lower())
        
        for i, subdir in enumerate(dirs):
            is_last_item = i == len(dirs) - 1
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_folder_tree(subdir, next_prefix, is_last_item, show_size, max_depth, current_depth + 1)
                
    except PermissionError:
        connector = "└── " if is_last else "├── "
        if current_depth == 0:
            print(f"{path.name}/ [权限不足]")
        else:
            print(f"{prefix}{connector}{path.name}/ [权限不足]")

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def get_directory_stats(directory, max_depth=None):
    """获取目录详细统计信息"""
    path = Path(directory)
    if not path.exists():
        return None
    
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'total_size': 0,
        'file_types': defaultdict(int),
        'dir_file_counts': [],  # 每个目录的文件数量
        'empty_dirs': [],
        'largest_dirs': []  # 按文件数量排序的目录
    }
    
    def scan_directory(dir_path, depth=0):
        if max_depth is not None and depth > max_depth:
            return
            
        try:
            items = list(dir_path.iterdir())
            files = [item for item in items if item.is_file()]
            dirs = [item for item in items if item.is_dir()]
            
            # 统计当前目录
            file_count = len(files)
            dir_size = 0
            
            for file in files:
                try:
                    file_size = file.stat().st_size
                    stats['total_size'] += file_size
                    dir_size += file_size
                    
                    # 统计文件类型
                    ext = file.suffix.lower() or '[无扩展名]'
                    stats['file_types'][ext] += 1
                    
                except (OSError, PermissionError):
                    continue
            
            stats['total_files'] += file_count
            stats['total_dirs'] += len(dirs)
            
            # 记录目录信息
            if file_count > 0:
                stats['dir_file_counts'].append((dir_path, file_count, dir_size))
            else:
                stats['empty_dirs'].append(dir_path)
            
            # 递归处理子目录
            for subdir in dirs:
                scan_directory(subdir, depth + 1)
                
        except (OSError, PermissionError):
            pass
    
    try:
        scan_directory(path)
        
        # 排序最多文件的目录
        stats['dir_file_counts'].sort(key=lambda x: x[1], reverse=True)
        stats['largest_dirs'] = stats['dir_file_counts'][:10]
        
    except Exception as e:
        print(f"扫描目录时出错: {e}")
    
    return stats

def print_directory_summary(directory, max_depth=None):
    """打印目录摘要信息"""
    stats = get_directory_stats(directory, max_depth)
    if not stats:
        return
    
    depth_info = f" (深度限制: {max_depth})" if max_depth else ""
    print(f"\n目录统计摘要: {directory}{depth_info}")
    print("=" * 60)
    print(f"总文件数: {stats['total_files']:,}")
    print(f"总文件夹数: {stats['total_dirs']:,}")
    print(f"总大小: {format_size(stats['total_size'])}")
    
    if stats['file_types']:
        print(f"\n文件类型分布:")
        sorted_types = sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True)
        for ext, count in sorted_types[:10]:  # 显示前10种类型
            print(f"  {ext}: {count:,} 个文件")
    
    if stats['largest_dirs']:
        print(f"\n文件最多的目录:")
        base_path = Path(directory)
        for i, (dir_path, file_count, dir_size) in enumerate(stats['largest_dirs'][:10], 1):
            try:
                rel_path = dir_path.relative_to(base_path)
                size_info = f" ({format_size(dir_size)})" if dir_size > 0 else ""
                print(f"  {i}. {rel_path}/ - {file_count:,} 个文件{size_info}")
            except ValueError:
                # 如果无法计算相对路径，使用绝对路径
                size_info = f" ({format_size(dir_size)})" if dir_size > 0 else ""
                print(f"  {i}. {dir_path}/ - {file_count:,} 个文件{size_info}")
    
    if stats['empty_dirs']:
        print(f"\n空文件夹 ({len(stats['empty_dirs'])} 个):")
        base_path = Path(directory)
        shown_count = min(10, len(stats['empty_dirs']))
        for empty_dir in stats['empty_dirs'][:shown_count]:
            try:
                rel_path = empty_dir.relative_to(base_path)
                print(f"  {rel_path}/")
            except ValueError:
                print(f"  {empty_dir}/")
        if len(stats['empty_dirs']) > shown_count:
            print(f"  ... 还有 {len(stats['empty_dirs']) - shown_count} 个")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='显示文件夹树结构和统计信息（仅显示目录）')
    parser.add_argument('path', nargs='?', default='.', help='要分析的目录路径（默认为当前目录）')
    parser.add_argument('-s', '--size', action='store_true', help='显示文件夹大小')
    parser.add_argument('--summary', action='store_true', help='显示详细统计摘要')
    parser.add_argument('-d', '--max-depth', type=int, help='最大显示深度')
    
    args = parser.parse_args()
    
    directory = os.path.abspath(args.path)
    
    if not os.path.exists(directory):
        print(f"错误: 路径 '{directory}' 不存在")
        sys.exit(1)
    
    if not os.path.isdir(directory):
        print(f"错误: '{directory}' 不是一个目录")
        sys.exit(1)
    
    depth_info = f" (最大深度: {args.max_depth})" if args.max_depth else ""
    print(f"文件夹树: {directory}{depth_info}")
    print("-" * 60)
    
    print_folder_tree(directory, show_size=args.size, max_depth=args.max_depth)
    
    if args.summary:
        print_directory_summary(directory, args.max_depth)

if __name__ == "__main__":
    main()
