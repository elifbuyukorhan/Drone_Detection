# regression predictive modeling problem
from sklearn.datasets import make_regression
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# make_regression() random regression problemi üretiyor. 
# 20 özelliği olan 1000 adet sample var. 10 tane bağımlı 10 tane bağımsız değişken içeriyor.
print(X)
# histograms of input variables
pyplot.subplot(211)
pyplot.hist(X[:, 0])
pyplot.title("Histogram of First Feature")
pyplot.subplot(212)
pyplot.hist(X[:, 1])
pyplot.title("Histogram of Second Feature")
pyplot.savefig('input_hist')
# pyplot.show()
# histogram of target variable
pyplot.subplot()
pyplot.title("Histogram of Output")
pyplot.hist(y)
pyplot.savefig('output_hist')
#pyplot.show()