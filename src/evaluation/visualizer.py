import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    return fig


def plot_per_stain_bars(metrics_df, metric="macro_f1"):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="stain", y=metric, data=metrics_df, ax=ax)
    ax.set_title(f'{metric} per Stain Class')
    ax.set_xlabel('Stain Class')
    ax.set_ylabel(metric)
    ax.axhline(metrics_df[metric].mean(), linestyle='--', color='red', label='mean')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig





def plot_stain_class_heatmap(metrics_df): 
    fig, ax = plt.subplots(figsize=(12, 6))
    heatmap_data = metrics_df.set_index("stain").filter(like="_f1")
    heatmap_data.columns = [c.replace("_f1", "") for c in heatmap_data.columns]
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
    ax.set_title('F1 Scores per Stain Class')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Stain Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
