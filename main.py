from preprocessing import (
    filter_train_data,
    filter_DMS,
    filter_2A3,
    preprocess_csv,
    process_data_test,
    combine_datasets,
)
from models import train
from submit import submit

if __name__ == "__main__":
    # filter out data
    filter_train_data()
    filter_DMS()
    filter_2A3()

    # preprocess csvs
    preprocess_csv("train_data_2a3_preprocessed", "train_data_2a3.csv")
    preprocess_csv("train_data_dms_preprocessed", "train_data_dms.csv")
    preprocess_csv(
        "test_data_preprocessed",
        "test_sequences.csv",
        map_fn=process_data_test,
        extra_cols_to_keep=["id_min", "id_max"],
    )
    combine_datasets()

    # train models
    train("full_32lat", dataset_name="full", batch_size=32)

    # submit predictions
    submit(batch_size=64)
