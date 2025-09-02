#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按章节拆分Markdown文档的脚本

该脚本读取指定的Markdown文档，并按照章节结构将其拆分成多个独立的Markdown文件。
每个章节将保存为单独的文件，文件名格式为：第X章-章节标题.md
"""

import os
import re
from pathlib import Path


def clean_filename(title):
    """
    清理文件名，移除不合法的字符
    """
    # 移除或替换不合法的文件名字符
    title = re.sub(r'[<>:"/\\|?*]', "", title)
    title = title.replace(":", "：")
    title = title.strip()
    return title


def extract_chapter_title(line):
    """从章节标题行中提取章节标题"""
    line = line.strip()

    # 处理特殊格式
    if "三、Model Compression" in line:
        return "第3章-模型压缩"
    elif "Model Compression" in line:
        return "第3章-模型压缩"
    elif "Quantization" in line or "量化" in line:
        return "第4章-量化技术"
    elif "模型压缩" in line:
        return "第3章-模型压缩"

    # 处理标准的第X章格式
    chapter_match = re.search(r"^#+\s*第(\d+)章[：:]?\s*(.+)$", line)
    if chapter_match:
        chapter_num = chapter_match.group(1)
        chapter_title = chapter_match.group(2)
        return f"第{chapter_num}章-{chapter_title}"

    # 移除开头的#号和空格作为备用
    title = re.sub(r"^#+\s*", "", line)
    return clean_filename(title)


def split_markdown_by_chapters(input_file_path, output_dir=None):
    """
    按章节拆分Markdown文档

    Args:
        input_file_path (str): 输入文档的路径
        output_dir (str): 输出目录，如果为None则使用输入文件所在目录
    """
    input_path = Path(input_file_path)

    if not input_path.exists():
        print(f"错误：文件 {input_file_path} 不存在")
        return

    # 设置输出目录
    if output_dir is None:
        output_dir = input_path.parent / "chapters"
    else:
        output_dir = Path(output_dir)

    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    print(f"正在读取文档: {input_file_path}")

    # 读取文档内容
    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    chapters = []
    current_chapter = None
    current_content = []

    # 定义章节标题的正则表达式模式
    chapter_patterns = [
        r"^#\s+第(\d+)章[：:]?\s*(.+)$",  # 第X章：标题
        r"^##\s+第(\d+)章[：:]?\s*(.+)$",  # ## 第X章：标题
        r"^#\s+三、Model\s+Compression$",  # # 三、Model Compression
        r"^##\s+三、Model\s+Compression$",  # ## 三、Model Compression
        r"^##\s+Model\s+Compression$",  # ## Model Compression
        r"^#\s+模型压缩$",  # # 模型压缩
        r"^##\s+模型压缩$",  # ## 模型压缩
        r"^#\s+Quantization$",  # # Quantization
        r"^##\s+Quantization$",  # ## Quantization
        r"^#\s+量化$",  # # 量化
        r"^##\s+量化$",  # ## 量化
    ]

    print("正在分析文档结构...")

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # 检查是否是章节标题
        is_chapter = False
        for pattern in chapter_patterns:
            if re.match(pattern, line_stripped):
                is_chapter = True
                break

        if is_chapter:
            # 保存前一个章节
            if current_chapter is not None and current_content:
                chapters.append(
                    {"title": current_chapter, "content": current_content.copy()}
                )

            # 开始新章节
            current_chapter = extract_chapter_title(line_stripped)
            current_content = [line]
            print(f"发现章节: {current_chapter}")
        else:
            # 添加到当前章节内容
            if current_chapter is not None:
                current_content.append(line)
            else:
                # 如果还没有遇到第一个章节，创建一个前言章节
                if not chapters and line_stripped:
                    current_chapter = "前言"
                    current_content = [line]

    # 保存最后一个章节
    if current_chapter is not None and current_content:
        chapters.append({"title": current_chapter, "content": current_content.copy()})

    print(f"\n共发现 {len(chapters)} 个章节")

    # 保存各个章节到独立文件
    for i, chapter in enumerate(chapters, 1):
        title = chapter["title"]
        content = chapter["content"]

        # 生成文件名
        if title == "前言":
            filename = "前言.md"
        elif "第" in title and "章" in title:
            # 提取章节号
            chapter_match = re.search(r"第([0-9一二三四五六七八九十]+)章", title)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                clean_title = clean_filename(title)
                filename = f"第{chapter_num}章-{clean_title.replace(f'第{chapter_num}章', '').strip()}.md"
                if filename.startswith("第-"):
                    filename = f"第{chapter_num}章.md"
            else:
                filename = f"{clean_filename(title)}.md"
        elif "Model Compression" in title:
            filename = "第3章-模型压缩技术.md"
        elif "Quantization" in title:
            filename = "第4章-量化技术.md"
        else:
            filename = f"{clean_filename(title)}.md"

        # 确保文件名不为空
        if not filename or filename == ".md":
            filename = f"章节{i}.md"

        output_file = output_dir / filename

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.writelines(content)
            print(f"已保存: {filename} ({len(content)} 行)")
        except Exception as e:
            print(f"保存文件 {filename} 时出错: {e}")

    print(f"\n拆分完成！所有章节已保存到: {output_dir}")


def main():
    """
    主函数
    """
    # 输入文档路径
    input_file = "/Users/yixin0909zhang/Documents/GitHub/model_deploy/模型部署-软硬一体优化技术-书稿【概念连接关系】.md"

    # 输出目录（可选，如果不指定则在输入文件同目录下创建chapters文件夹）
    output_directory = None

    print("=" * 60)
    print("Markdown文档章节拆分工具")
    print("=" * 60)

    split_markdown_by_chapters(input_file, output_directory)


if __name__ == "__main__":
    main()
