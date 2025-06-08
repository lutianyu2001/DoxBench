#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path
from PIL import Image
import shutil

def calculate_image_hash(image_path):
    """计算图片的像素哈希值"""
    try:
        with Image.open(image_path) as img:
            # 直接获取整个图片的像素数据计算哈希
            pixels = img.tobytes()
            return hashlib.md5(pixels).hexdigest()
    except Exception:
        return None

def get_image_files(directory):
    """递归获取目录下所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return [f for f in Path(directory).rglob('*') 
            if f.suffix.lower() in image_extensions and f.is_file()]

def build_hash_map(image_files):
    """构建哈希值到文件路径的映射"""
    hash_map = {}
    for file_path in image_files:
        hash_value = calculate_image_hash(file_path)
        if hash_value:
            hash_map[hash_value] = file_path
    return hash_map

def main():
    parser = argparse.ArgumentParser(description='根据图片哈希值匹配并覆盖文件')
    parser.add_argument('src', help='源目录路径')
    parser.add_argument('dst', help='目标目录路径')
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    
    if not src_path.exists() or not dst_path.exists():
        print("错误: 源目录或目标目录不存在")
        return

    print("扫描源目录...")
    src_files = get_image_files(src_path)
    src_hash_map = build_hash_map(src_files)
    
    print("扫描目标目录...")
    dst_files = get_image_files(dst_path)
    dst_hash_map = build_hash_map(dst_files)
    
    print(f"源目录: {len(src_hash_map)} 个有效图片")
    print(f"目标目录: {len(dst_hash_map)} 个有效图片")
    
    matched_count = 0
    for hash_value, src_file in src_hash_map.items():
        if hash_value in dst_hash_map:
            dst_file = dst_hash_map[hash_value]
            try:
                shutil.copy2(src_file, dst_file)
                print(f"覆盖: {src_file.name} -> {dst_file}")
                matched_count += 1
            except Exception as e:
                print(f"错误: 无法复制 {src_file} -> {dst_file}: {e}")
    
    print(f"完成: 成功匹配并覆盖 {matched_count} 个文件")

if __name__ == '__main__':
    main()
