import skl2onnx
import onnx
import sklearn
import numpy
import onnxmltools
import onnxruntime as rt

from numpy import array
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

'''
Example taken from 
https://onnx.ai/sklearn-onnx/auto_examples/plot_convert_model.html#sphx-glr-auto-examples-plot-convert-model-py
'''

"""Train a model"""
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)
print(clr)

"""Convert model to onnx"""
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = skl2onnx.to_onnx(clr, initial_type)
onnxmltools.save_model(onnx_model, "../../models/rf-iris.onnx")

"""Compute the prediction with ONNX Runtime"""
sess = rt.InferenceSession("../../models/rf-iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
X = array([[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 4.2, 1.5], [5.9, 3., 5.1, 1.8]])
pred_onx = sess.run([label_name], {input_name: X.astype(numpy.float32)})[0]
print(pred_onx)
assert all(pred_onx == [0, 1, 2])

"""Full example with a logistic regression"""
clr = LogisticRegression(max_iter=200)
clr.fit(X_train, y_train)
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onx = convert_sklearn(clr, initial_types=initial_type, target_opset=12)
with open("../../models/logreg-iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession("../../models/logreg-iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

"""Versions used for this example"""
print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
