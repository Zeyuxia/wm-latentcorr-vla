#!/bin/bash

# ================= 配置区域 =================
# 请在此处替换为您实际的数据集 ID
REPO_ID="TianxingChen/RoboTwin2.0" 
# 下载保存的本地目录
LOCAL_DIR="/data/zhenyangfan/RoboTwin/data"
# 最大重试次数
MAX_RETRIES=20
# 每次重试前的等待时间（秒）
RETRY_DELAY=5

# 定义需要下载的文件列表
FILES=(
    "dataset/hanging_mug/aloha-agilex_clean_50.zip"
    "dataset/blocks_ranking_rgb/aloha-agilex_clean_50.zip"
    "dataset/pick_diverse_bottles/aloha-agilex_clean_50.zip"
    "dataset/place_cans_plasticbox/aloha-agilex_clean_50.zip"
)
# ===========================================

# 初始化计数器
attempt=1
success=false

echo "开始下载数据集: $REPO_ID"
echo "目标目录: $LOCAL_DIR"

# 循环尝试下载
while [ $attempt -le $MAX_RETRIES ]; do
    echo "----------------------------------------"
    echo "尝试 #$attempt / $MAX_RETRIES"
    
    # 执行下载命令
    # "${FILES[@]}" 会自动展开为上面定义的文件列表
    huggingface-cli download "$REPO_ID" \
        --include "${FILES[@]}" \
        --repo-type dataset \
        --local-dir "$LOCAL_DIR" \
        --resume-download  # 显式开启断点续传（新版CLI默认开启，加上更保险）

    # 检查上一条命令的退出状态码 ($?)
    if [ $? -eq 0 ]; then
        success=true
        echo "✅ 下载成功！所有文件已就绪。"
        break
    else
        echo "❌ 下载过程中遇到错误。"
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "将在 $RETRY_DELAY 秒后重试..."
            sleep $RETRY_DELAY
        else
            echo "⛔ 已达到最大重试次数，下载任务失败。"
        fi
    fi
    
    ((attempt++))
done

# 最后检查状态以决定脚本退出码
if [ "$success" = true ]; then
    exit 0
else
    exit 1
fi