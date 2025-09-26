import os
import pandas as pd
from itertools import combinations

folder_path = "lake49"

delimiter_map = {
    "table_0.csv": "_",
    "table_1.csv": "\t",
    "table_2.csv": "\t",
    "table_3.csv": ",",
    "table_4.csv": ",",
    "table_5.csv": "_",
    "table_6.csv": ",",
    "table_7.csv": ",",
    "table_8.csv": ",",
    "table_9.csv": ",",
    "table_10.csv": "_",  # combi of , and _
    "table_11.csv": "_",
    "table_12.csv": ",",
    "table_13.csv": "\t",
    "table_14.csv": ",",
    "table_15.csv": ",",
    "table_16.csv": ",",
    "table_17.csv": ",",
    "table_18.csv": "_",
    "table_19.csv": ","}

#list of all 19 tables
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
csv_files_test = ["table_0.csv", "table_1.csv", "table_2.csv"]

similarity_threshold = 0.5

def jaccard_similarity(col1, col2):
    col1_nonan = col1.dropna()
    col2_nonan = col2.dropna()

    # Skip if both columns are entirely zeros
    if (col1_nonan.eq(0).all() and col2_nonan.eq(0).all()):
        return 0.0

    # Convert to strings for set comparison
    set1 = set(col1_nonan.astype(str))
    set2 = set(col2_nonan.astype(str))

    if not set1 or not set2:
        return 0.0

    return len(set1 & set2) / len(set1 | set2)   # Jaccard formula from lecture



# load all the datframes
dataframes = {}
for f in csv_files_test:   # test mode for just first 3 files
    file_path = os.path.join(folder_path, f)

    delimiter = delimiter_map.get(f)
    df = pd.read_csv(file_path, delimiter=delimiter, engine="c", on_bad_lines="skip")
    dataframes[f] = df
    print(f"📂 {f}")
    print("   Columns:", list(df.columns))


#compare columns for jaccard
similar_columns = []
for (file1, df1), (file2, df2) in combinations(dataframes.items(), 2):
    print(f"\nComparing {file1} vs {file2}")
    for col1 in df1.columns:
        for col2 in df2.columns:
            sim = jaccard_similarity(df1[col1], df2[col2])
            if sim >= similarity_threshold:
                similar_columns.append((file1, col1, file2, col2, sim))
                print(f"Similarity found ({sim:.2f}) between:")
                print(f"   {file1} → {col1}")
                print(f"   {file2} → {col2}\n")







def discovery_algorithm():
    """Function should be able to perform data discovery to find related datasets
    Possible Input: List of datasets
    Output: List of pairs of related datasets
    """

    pass
