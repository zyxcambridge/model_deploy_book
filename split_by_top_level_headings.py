#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按一级标题拆分Markdown文档脚本
将每个一级标题及其内容保存为独立的md文件，使用阿拉伯数字重新编号
"""

import os
import re
from pathlib import Path


def split_markdown_by_top_level_headings(input_file, output_dir="chapters_split"):
    """
    按一级标题拆分Markdown文档

    Args:
        input_file (str): 输入的Markdown文件路径
        output_dir (str): 输出目录
    """
    # 读取原始文档
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 使用正则表达式匹配一级标题
    # 匹配以单个#开头，后面跟空格和标题内容的行
    heading_pattern = r"^# ([^#].*)$"

    # 分割文档内容
    sections = re.split(heading_pattern, content, flags=re.MULTILINE)

    # 第一个元素是文档开头到第一个一级标题之前的内容
    # 如果有内容且不只是空白，则保存为0.md
    if sections[0].strip():
        with open(output_path / "0.md", "w", encoding="utf-8") as f:
            f.write(sections[0].strip())
        print(f"已创建: 0.md (文档前言部分)")

    # 处理一级标题和对应内容
    chapter_num = 1
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            heading = sections[i].strip()
            content_part = sections[i + 1]

            # 创建完整的章节内容
            chapter_content = f"# {heading}\n{content_part}"

            # 保存为阿拉伯数字编号的文件
            filename = f"{chapter_num}.md"
            filepath = output_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(chapter_content.strip())

            print(f"已创建: {filename} - {heading}")
            chapter_num += 1
        elif i < len(sections):
            # 处理最后一个标题（如果没有对应内容）
            heading = sections[i].strip()
            chapter_content = f"# {heading}\n"

            filename = f"{chapter_num}.md"
            filepath = output_path / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(chapter_content.strip())

            print(f"已创建: {filename} - {heading}")
            chapter_num += 1

    print(f"\n拆分完成！共创建了 {chapter_num - 1} 个章节文件")
    print(f"输出目录: {output_path.absolute()}")

    return chapter_num - 1


def main():
    # 输入文件路径
    input_file = "/Users/yixin0909zhang/Documents/GitHub/model_deploy/模型部署-软硬一体优化技术-书稿【概念连接关系】.md"

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在 - {input_file}")
        return

    print(f"开始处理文件: {input_file}")
    print("按一级标题拆分文档...")

    # 执行拆分
    chapter_count = split_markdown_by_top_level_headings(input_file)

    print(f"\n处理完成！总共拆分出 {chapter_count} 个章节文件")


if __name__ == "__main__":
    main()
