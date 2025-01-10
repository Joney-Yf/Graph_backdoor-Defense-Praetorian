import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置Seaborn样式
sns.set(style="whitegrid")

# 数据准备
datasets = ['Cora', 'PubMed', 'Ogbn-Arxiv']
methods = ['Praetorian\R', 'Praetorian\RV', 'Praetorian']
data = {
    'Cora': {
        'ASR(%)': [62.36, 2.66, 0.94],
        'CA(%)': [79.82, 81.70, 81.73]
    },
    'PubMed': {
        'ASR(%)': [70.39, 0.825, 0.842],
        'CA(%)': [83.21, 84.13, 84.10]
    },
    'Ogbn-Arxiv': {
        'ASR(%)': [65.46, 0.01, 0.0215],
        'CA(%)': [64.54, 65.83, 65.92]
    }
}

# 转换为DataFrame
rows = []
for dataset in datasets:
    for i, method in enumerate(methods):
        rows.append({
            'Dataset': dataset,
            'Method': method,
            'ASR(%)': data[dataset]['ASR(%)'][i],
            'CA(%)': data[dataset]['CA(%)'][i]
        })

df = pd.DataFrame(rows)

# 创建一个图形和三个子图
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

# 定义调色板
palette = sns.color_palette("Set1", len(methods))

# 纹理样式
hatches = ['', '////', 'xxxx']  # 不同方法的纹理

for ax, dataset in zip(axes, datasets):
    subset = df[df['Dataset'] == dataset]
    subset_melted = subset.melt(id_vars=['Method'], value_vars=['ASR(%)', 'CA(%)'],
                                var_name='Metric', value_name='Percentage')

    # 绘制柱状图
    bars = sns.barplot(x='Metric', y='Percentage', hue='Method', data=subset_melted, palette=palette, ax=ax)

    # 添加纹理到柱状图
    for i, bar in enumerate(bars.patches):
        hatch_idx = i % len(hatches)  # 循环应用纹理
        bar.set_hatch(hatches[hatch_idx])
    
    # 设置标题和标签
    ax.set_title(f'{dataset}', fontsize=22, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=20)
    ax.set_xlabel('')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['ASR', 'CA'], fontsize=16)  # 设置 ASR 和 CA 作为横轴标签

    # 设置y轴范围
    max_percentage = subset_melted['Percentage'].max()
    ax.set_ylim(0, max_percentage * 1.1 if max_percentage > 0 else 1)

    # 添加数值标签
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            offset = 5 if height > 5 else 10
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=12, color='black',
                        xytext=(0, offset),
                        textcoords='offset points')

    # 移除子图的图例
    ax.get_legend().remove()

# 添加统一的图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Method', loc='center right', bbox_to_anchor=(0.96, 0.5), fontsize=14)

# 调整布局以避免图例和子图重叠
plt.tight_layout(rect=[0, 0, 0.85, 0.95])

# 保存图形
plt.savefig('combined_barcharts_with_textures.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
