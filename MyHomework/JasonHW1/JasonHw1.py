from pandas import read_csv
import numpy as np
from numpy import set_printoptions
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
print("rescale using sklearn.normalize\n")
filename = 'adult.data.csv'
names = ["age", "workclass", "fnlwgt", "education", "education-num", "martial-status", "occupation", "relationship", "race", "sex", "capital-gain",
         "capital-loss", "hours-per-week", "native-country","label"]
dataread = read_csv(filename, names = names, na_values = '?')
array = dataread.values
# manually convert the last column
for i in array :
    for j, aStr in enumerate(i) :
        if isinstance(aStr, str) :
            i[j] = aStr.strip()
    #print i
    if (i[-1] == ">50K") :
        i[-1] = 1
    elif (i[-1] == "<=50K") :
        i[-1] = 0

x, y = array.shape
le = LabelEncoder()
for i in xrange(y):
    array[: , i] = le.fit_transform(array[: , i])

#print array[0 : 3,  : ]
# hot = OneHotEncoder(handle_unknown = "ignore")
# array = hot.fit_transform(array)
# print array[0 : 3,  : ]
imp = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
array = imp.fit_transform(array)

# array = normalize(array, norm = 'l2')
sd = StandardScaler()
array = sd.fit_transform(array.astype("float64"))
print(array[0:5, :])
