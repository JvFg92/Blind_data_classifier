import pandas as pd
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script preprocesses CSV datasets for classification tasks.
It imports data, handles missing values, normalizes numerical features,
and generates visualizations for target class distribution and feature box plots.

The criteria for a dataset to be considered suitable for classification
are:
1. More than one unique class in the target column.
2. More than 50 rows after preprocessing.
3. Class balance: Classes in the target column should have a relatively uniform distribution.
   Highly imbalanced classes can lead to biased models.
4. Data Quality:
    - Absence of Duplicates: Duplicated rows can overemphasize certain samples.
    - Consistency: Values in columns should be consistent in format and meaning.
    - Feature Relevance: Features should have predictive power for the target column.
"""

##########################################
def import_data(file_path):
    return pd.read_csv(file_path, header=None)

##########################################
def preprocess_data(df):
    """
    Preprocess the imported data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the imported data.

    Returns:
    pd.DataFrame: DataFrame containing the preprocessed data.
    """
    df_processed = df.copy()

    # Handle missing values
    df_processed = df_processed.dropna()

    # Handle duplicates: Remove duplicate rows
    initial_rows = df_processed.shape[0]
    df_processed.drop_duplicates(inplace=True)
    if df_processed.shape[0] < initial_rows:
        print(f"  Removed {initial_rows - df_processed.shape[0]} duplicate rows.")

    # Normalize numerical features
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        std_dev = df_processed[numerical_cols].std()
        # Avoid division by zero in case std_dev is zero
        std_dev = std_dev.replace(0, 1)
        df_processed[numerical_cols] = (df_processed[numerical_cols] - df_processed[numerical_cols].mean()) / std_dev

    return df_processed

##########################################
def find_best(data_folder='/Blind_data_classifier/bases', plot_results=False):
    """
    Main function to find the best database for classification.
    """
    print("Starting database selection process...\n")
    if not os.path.exists(data_folder):
        print(f"Error: The '{data_folder}' folder was not found.")
        print("Please create a folder named 'bases' and place your CSV files inside it.")
        exit()

    database_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

    if not database_files:
        print(f"Error: No CSV files found in '{data_folder}'.")
        print("Please place your CSV files inside the 'bases' folder.")
        exit()

    database_info = {}

    #Assuming first column as ID and last as Class
    id_column_index = 0
    target_column_index = None

    for file_path in database_files:
        print(f"\nProcessing {file_path}...")
        try:
            df_original_loaded = import_data(file_path)
            df_current_processed = df_original_loaded.copy()
            target_column_index = df_current_processed.columns[-1]

            #Verify if the target column index is the target column
            if id_column_index in df_current_processed.columns and id_column_index != target_column_index:
                df_current_processed = df_current_processed.drop(columns=[id_column_index]).copy()
                print(f"  Column ID ({id_column_index}) removed.")
            elif id_column_index == target_column_index:
                print(f"  Column ID ({id_column_index}) is the same as the target column. Not removing to avoid losing the target.")
            else:
                print(f"  Column ID ({id_column_index}) not found or already removed/not applicable.")

            original_rows = df_current_processed.shape[0]
            original_cols = df_current_processed.shape[1]

            df_preprocessed = preprocess_data(df_current_processed)

            preprocessed_rows = df_preprocessed.shape[0]
            preprocessed_cols = df_preprocessed.shape[1]

            if target_column_index not in df_preprocessed.columns:
                print(f"Warning: Target column with index '{target_column_index}' was removed during preprocessing in {file_path}. Ignoring this dataset.")
                continue

            target_values = df_preprocessed[target_column_index].values
            num_unique_classes = len(np.unique(target_values))
            class_distribution = Counter(target_values)

            #Calculate class balance metric (e.g., standard deviation of class counts)
            class_counts = np.array(list(class_distribution.values()))
            class_balance_std = np.std(class_counts)

            is_suitable_for_classification = num_unique_classes > 1 and preprocessed_rows > 50

            #Check for consistency (basic check: identify non-numeric values in numeric columns)
            inconsistent_data_found = False
            for col in df_preprocessed.columns:
                if df_preprocessed[col].dtype == 'object':
                    if df_preprocessed[col].apply(lambda x: pd.to_numeric(x, errors='coerce')).isnull().any():
                        pass

            database_info[file_path] = {
                'original_rows': original_rows,
                'original_cols': original_cols,
                'preprocessed_rows': preprocessed_rows,
                'preprocessed_cols': preprocessed_cols,
                'num_unique_classes': num_unique_classes,
                'class_distribution': dict(class_distribution),
                'class_balance_std': class_balance_std,
                'has_duplicates': original_rows > preprocessed_rows,
                'inconsistent_data_potentially': inconsistent_data_found,
                'is_suitable_for_classification': is_suitable_for_classification,
                'preprocessed_df': df_preprocessed,
                'target_column_index': target_column_index
            }

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print("\n--- Database Summary ---")
    for file, info in database_info.items():
        print(f"\nDatabase: {file}")
        print(f"  Original Rows: {info['original_rows']}, Original Columns: {info['original_cols']}")
        print(f"  Preprocessed Rows: {info['preprocessed_rows']}, Preprocessed Columns: {info['preprocessed_cols']}")
        print(f"  Number of Unique Classes (Target): {info['num_unique_classes']}")
        print(f"  Class Distribution: {info['class_distribution']}")
        print(f"  Class Balance (Std Dev): {info['class_balance_std']:.2f}")
        print(f"  Has Duplicates Removed: {info['has_duplicates']}")
        print(f"  Suitable for Classification: {info['is_suitable_for_classification']}")

    #Choose the "best" database
    best_database = None
    max_preprocessed_rows = -1

    for file, info in database_info.items():
        if info['is_suitable_for_classification'] and info['preprocessed_rows'] > max_preprocessed_rows:
            max_preprocessed_rows = info['preprocessed_rows']
            best_database = file

    if best_database:
        print(f"\n--- Best Database Selected for Classification: {best_database} ---")
        print(f"Detailed Information: {database_info[best_database]}\n")

        best_db_info = database_info[best_database]
        best_df_preprocessed = best_db_info['preprocessed_df']
        best_target_column_index = best_db_info['target_column_index']

        if plot_results:
            # --- Plot Class Distribution ---
            plt.figure(figsize=(10, 6))
            sns.countplot(x=best_df_preprocessed[best_target_column_index], order=best_df_preprocessed[best_target_column_index].value_counts().index)
            plt.title(f'Class Distribution for Best Database: {os.path.basename(best_database)}')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        #Return the path to the best database for further processing
        return best_database
    else:
        print("\nNo suitable database found for classification based on the criteria.")
        return None

##########################################  
#if __name__ == "__main__": 
#    find_best(plot_results=True)