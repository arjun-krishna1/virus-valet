import tensorflow as tf

model = tf.keras.models.load_model('C:\\dev\\Hackathon\\RoboHacks\\virus-valet\\app\\model')

"""
data for preprocessing and postprocesing
from Jupyter Notebook
"""
age_min = 18
age_max = 80
age_idx = 0

min_pred = -3.1807234
max_pred = 2.8405762
category_size = 2.0070997873942056


def min_max_scaling(min_val, max_val, data, idx):
    def min_max_scale_val(val):
        return (val - min_val) / (max_val - min_val)

    data[idx] = min_max_scale_val(data[idx])
    return data


def predict_severity(inp):
    """
    - given a list containing a Patient's symptoms
    - predict the severity of their disease
    - from 0 as least severe, to 2 as most severe
    """
    this_pred = model.predict(tf.reshape(list(inp), (1, 17)))[0][0]
    category = -1
    if this_pred < min_pred + category_size:
        category = 2
    elif this_pred >= min_pred + category_size and this_pred <= max_pred - category_size:
        category = 1
    else:
        category = 0
    return category


def predict(data):
    """
    Given a patient's surve responses,
    process the data and get a prediction from the ml model
    """
    scaled_data = min_max_scaling(age_min, age_max, data, age_idx)
    reshaped_data = tf.reshape(scaled_data, (1, 17))
    pred = predict_severity(reshaped_data)
    return pred


def test():
    test_data = [0.3548387096774194, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    return predict(test_data)


if __name__ == "__main__":
    print(test())
