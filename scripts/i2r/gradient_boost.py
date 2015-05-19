


from pynet.datasets.i2r import *

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import *

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.07, max_depth=6)

data = I2R_Posterior_ClnDNN_NoisyFeat(1998, 1998, one_hot=False)

data.__iter__()
data = next(data)

X = data.get_train().X
y = data.get_train().y


clf.fit(X, y)

test_X = data.get_test().X
test_y = data.get_test().y

test_X = test_X
test_y = test_y

pred_y = clf.predict(test_X)

print 'pred accuracy', accuracy_score(test_y, pred_y)
print 'actual accuracy', accuracy_score(test_X.argmax(axis=1), test_y)
