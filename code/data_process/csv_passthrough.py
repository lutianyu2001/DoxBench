import pandas as pd

def csv_passthrough(input_file, output_file):
    """
    将CSV文件原样读取进pandas，再原样输出至CSV
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
    """
    try:
        # 原样读取CSV文件
        # dtype=object 避免pandas自动类型转换
        # keep_default_na=False 避免空值被转换为NaN
        df = pd.read_csv(
            input_file, 
            dtype=object,           # 保持原始数据类型
            keep_default_na=False,  # 不将空字符串转换为NaN
            na_filter=False         # 完全关闭NA过滤
        )
        
        # 原样输出至CSV文件
        df.to_csv(
            output_file, 
            index=False,           # 不输出行索引
            na_rep='',            # NaN值用空字符串表示
            encoding='utf-8'      # 使用UTF-8编码
        )
        
        print(f"成功将 {input_file} 原样复制到 {output_file}")
        return df

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")

if __name__ == "__main__":
    # 示例用法
    input_csv = "exif.csv"
    output_csv = "exif_output.csv"

    # 调用方法
    result_df = csv_passthrough(input_csv, output_csv)
