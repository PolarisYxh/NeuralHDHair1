#!/bin/bash

# 定义起始和结束时间（以秒为单位）
start_time=$(date -d "2023-06-29 09:00:00" +%s)
end_time=$(date -d "2023-06-29 13:50:00" +%s)
dir="/data/HairStrand/checkpoints/StepNet/2023-06-27/checkpoint"
# 获取当前目录下所有文件
files=$(find "$dir" -type f)

# 遍历每个文件
for file in $files; do
    # 获取文件的修改时间（以秒为单位）
    modified_time=$(stat -c %Y "$file")
    # echo "$file"
    # 检查文件是否在时间范围内
    if [[ $modified_time -ge $start_time && $modified_time -le $end_time ]]; then
        echo "$file"
    fi
done
