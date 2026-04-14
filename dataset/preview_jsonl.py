import json

def preview_jsonl(file_path, num_lines=3, start_line=0):
    """
    预览 .jsonl 文件内容
    
    Args:
        file_path (str): 文件路径
        num_lines (int): 要读取的行数
        start_line (int): 起始行号（从 0 开始计数）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过前面的行
        for _ in range(start_line):
            next(f, None)  # 防止文件行数不足时报错
        
        for i in range(num_lines):
            line = next(f, None)
            if line is None:
                break  # 文件已结束
            try:
                data = json.loads(line)
                print(f"--- Line {start_line + i + 1} ---")
                print(json.dumps(data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"--- Line {start_line + i + 1} (Invalid JSON) ---")
                print(line.strip())


preview_jsonl('dataset\dpo.jsonl', num_lines=3, start_line=0)
