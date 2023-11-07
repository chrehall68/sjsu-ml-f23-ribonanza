from preprocessing import (
    filter_train_data,
    filter_DMS,
    filter_2A3,
    preprocess_csv,
    process_data_test,
)
from models import train, create_lin_attention
from submit import submit

if __name__ == "__main__":
    # filter out data
    filter_train_data()
    filter_DMS()
    filter_2A3()

    # preprocess csvs
    preprocess_csv("train_data_2a3_preprocessed", "train_data_2a3.csv", samples=1000)
    preprocess_csv("train_data_dms_preprocessed", "train_data_dms.csv", samples=1000)
    preprocess_csv(
        "test_data_preprocessed",
        "test_sequences.csv",
        map_fn=process_data_test,
        extra_cols_to_keep=["id_min", "id_max"],
        samples=1000,
    )

    # train models
    train("2a3_linearfold", dataset_name="2a3", att_factory=create_lin_attention)
    train("dms_linearfold", dataset_name="dms", att_factory=create_lin_attention)

    # submit predictions
    submit(batch_size=64)
