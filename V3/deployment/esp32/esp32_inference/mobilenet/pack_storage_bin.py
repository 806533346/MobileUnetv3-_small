"""将 test_input.bin + test_output.bin 合并为 storage 分区镜像

===== 注意 =====
此脚本用于打包单张图像的测试数据 (旧版)。
当前批量推理使用 generate_test_data.py 直接生成 5 张图像的 storage.bin,
包含完整的输入区+输出区+预测区布局, 不再使用此脚本。

如需要单张图像测试:
  1. 确保 main/test_input.bin 和 main/test_output.bin 存在
  2. python pack_storage_bin.py
  3. 烧录 storage.bin 到 Flash 0x610000

===== Flash 分区布局 (单张) =====
  0x000000: test_input  (TEST_INPUT_SIZE  × 4 = 786,432 bytes)
  0x0C0000: test_output (TEST_OUTPUT_SIZE × 4 = 262,144 bytes)
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

header_path = os.path.join(SCRIPT_DIR, 'main', 'test_config.h')
input_path = os.path.join(SCRIPT_DIR, 'main', 'test_input.bin')
output_path = os.path.join(SCRIPT_DIR, 'main', 'test_output.bin')
storage_path = os.path.join(SCRIPT_DIR, 'storage.bin')

def get_define(header, name):
    """从 test_config.h 中解析 C 预处理器宏的值"""
    for line in open(header):
        if line.startswith('#define ' + name):
            return int(line.split()[2])
    raise KeyError(f'{name} not found in test_config.h')

input_size = get_define(header_path, 'TEST_INPUT_SIZE')     # 196608
output_size = get_define(header_path, 'TEST_OUTPUT_SIZE')   # 65536

input_bytes = input_size * 4                                 # 786432
output_bytes = output_size * 4                               # 262144
output_offset = input_bytes                                  # 输出紧接输入之后

with open(input_path, 'rb') as f:
    input_data = f.read()
with open(output_path, 'rb') as f:
    output_data = f.read()

assert len(input_data) == input_bytes, f'input size mismatch: {len(input_data)} vs {input_bytes}'
assert len(output_data) == output_bytes, f'output size mismatch: {len(output_data)} vs {output_bytes}'

# 拼接: [input_data][output_data]
combined = bytearray(output_offset + output_bytes)
combined[:input_bytes] = input_data
combined[output_offset:output_offset + output_bytes] = output_data

with open(storage_path, 'wb') as f:
    f.write(combined)

print(f'storage.bin: {len(combined)} bytes (input={input_bytes}, output={output_bytes})')
print(f'Flash: esptool.py write_flash 0x610000 storage.bin')
