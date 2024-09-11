import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


CLASS_COLOURS = {
    1: 'red',
    2: 'orange',
    3: 'purple',
    4: 'midnightblue',
    5: 'wheat',
    6: 'green',
    7: 'yellow',
    8: 'salmon',
    9: 'brown',
    10: 'black',
    11: 'lightsteelblue',
    12: 'olive'
}


def barplot_file_distribution(df: pd.DataFrame) -> None:
    """
    Generates a bar plot showing the distribution of file types in the provided DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing a 'type' column.
    """
    file_type_counts = df['type'].value_counts()
    file_type_df = file_type_counts.reset_index()
    file_type_df.columns = ['File Type', 'Count']

    plt.figure(figsize=(10, 6))
    plt.bar(file_type_df['File Type'], file_type_df['Count'], color='green')
    plt.title('File Type Counts')
    plt.xlabel('File Type')
    plt.ylabel('Count')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()


def piechart_file_distribution(df: pd.DataFrame) -> None:
    """
    Generates a pie chart showing the distribution of file types in the provided DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'type' column.
    """
    type_counts = df['type'].value_counts()
    type_percentages = type_counts / type_counts.sum() * 100
    plt.figure(figsize=(8, 8))
    plt.pie(type_percentages, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of File Types')
    plt.axis('equal')
    plt.show()


def scatter_file_sizes(df: pd.DataFrame) -> None:
    """
    Generates a scatter plot for file size distribution by file type.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'type' and 'size KB' column.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='type', y='size KB', data=df)
    plt.title('File Size Distribution by Type')
    plt.xlabel('File Type')
    plt.ylabel('Size (KB)')
    plt.show()


def boxplot_file_sizes(df: pd.DataFrame, size_min: int, size_max: int) -> None:
    """
    Generates a box plot for file size distribution by file type.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'size KB' column.
    """
    df_sample = df[(df['size KB'] >= size_min) & (df['size KB'] <= size_max)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='type', y='size KB', data=df_sample)
    plt.title(f'File Size Distribution by Type (Sizes between {size_min} KB and {size_max} KB)')
    plt.xlabel('File Type')
    plt.ylabel('Size (KB)')
    plt.show()


def plot_byte_grey_scale(byte_sequence: bytes | np.ndarray) -> None:
    """
    Generates a grey scale plot for a byte sequence.

    Args:
        byte_sequence (bytes | np.ndarray): A sequence or array of bytes.
    """
    if type(byte_sequence) == bytes:
        byte_sequence = np.array([b for b in byte_sequence])
    plt.figure(figsize=(10, 2))
    plt.imshow(byte_sequence.reshape(-1, 64), cmap='gray')
    plt.title("Byte Sequence Grey Scale")
    plt.colorbar()
    plt.show()


def plot_byte_value_distribution(byte_sequence: bytes | np.ndarray) -> None:
    """
    Generates a value distribution plot for a byte sequence.

    Args:
        byte_sequence (bytes | np.ndarray): A sequence or array of bytes.
    """
    if type(byte_sequence) == bytes:
        byte_sequence = np.array([b for b in byte_sequence])
    plt.figure(figsize=(10, 6))
    sns.histplot(byte_sequence.flatten(), bins=256, kde=False, color='blue')
    plt.title("Distribution of Byte Values")
    plt.xlabel("Byte Value")
    plt.ylabel("Frequency")
    plt.show()


def correlation_matrix_heatmap(corr_matrix: np.ndarray) -> None:
    """
    Generates a correlation matrix heatmap.

    Args:
        corr_matrix (np.ndarray): A correlation matrix.
    """
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=False)
    plt.title('Correlation Matrix')
    plt.yticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


def plot_pca(byte_sequences: np.ndarray, labels: np.ndarray, class_names: list, title: str, class_colours: dict=CLASS_COLOURS) -> None:
    """
    Generates a PCA plot for byte sequences.
    
    Args:
        byte_sequences (np.ndarray): 2D array where each row is a byte sequence.
        labels (np.ndarray): 1D array of labels corresponding to each byte sequence.
        title (title): The plot title.
        class_colors (dict): Dictionary mapping labels to specific colors.
    """
    reduced_data = PCA(n_components=2).fit_transform(byte_sequences)
    plt.figure(figsize=(10, 6))
    for label, color in class_colours.items():
        indices = labels == label
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], color=color, label=f'{class_names[label-1]}', alpha=0.7)
    plt.title(title)
    plt.legend(title="Classes")
    plt.show()


def plot_tsne(byte_sequences: np.ndarray, labels: np.ndarray, class_names: list, title: str, class_colours: dict=CLASS_COLOURS) -> None:
    """
    Generates a t-SNE plot for byte sequences.

    Args:
        byte_sequences (np.ndarray): 2D array where each row is a byte sequence.
        labels (np.ndarray): 1D array of labels corresponding to each byte sequence.
        class_names (list): Class Names.
        title (title): The plot title.
        class_colors (dict): Dictionary mapping labels to specific colors.
    """
    reduced_data = TSNE(n_components=2).fit_transform(byte_sequences)
    plt.figure(figsize=(10, 6))
    for label, color in class_colours.items():
        indices = labels == label
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], color=color, label=f'{class_names[label-1]}', alpha=0.7)
    plt.title(title)
    plt.legend(title="Classes")
    plt.show()