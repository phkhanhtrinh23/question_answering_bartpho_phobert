import os
import json
from sklearn.model_selection import train_test_split

def save_data(data, type):
    # save your preprocessed data
    with open(os.path.join("data/", type + ".json"), "w") as f:
        json.dump(data, f, indent= 4, ensure_ascii=False)
    return


if __name__ == "__main__":

    with open("data/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    f.close()

    train_data, valid_data  = train_test_split(train_data["data"], test_size= 0.1)

    train_data = {"data": train_data}
    valid_data = {"data": valid_data}

    save_data(train_data, "new_train")
    save_data(valid_data, "valid")