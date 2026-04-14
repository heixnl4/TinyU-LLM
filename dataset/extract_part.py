import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(script_dir, 'pretrain_hq.jsonl')
output_path = os.path.join(script_dir, 'pretrain_part.jsonl')

def extract_jsonl_lines(input_path, output_path, num_lines=2000):
    """
    从 .jsonl 文件中提取指定行数并保存到新文件

    Args:
        input_path (str): 输入文件路径
        output_path (str): 输出文件路径
        num_lines (int): 要提取的行数
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for _ in range(num_lines):
            line = infile.readline()
            if not line:
                break  # 文件结束
            outfile.write(line)

# 使用示例
extract_jsonl_lines(input_path, output_path, num_lines=10000)