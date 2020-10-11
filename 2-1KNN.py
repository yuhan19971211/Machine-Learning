from numpy import*
import operator
import kNN
import matplotlib
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2])
# plt.show()
# normat,ranges,minval=kNN.auto_norm(dating_data_mat)
# kNN.dating_class_test()



if __name__ == '__main__':
    dating_data_mat, dating_labels = kNN.file_to_matrix('datingTestSet2.txt')
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    print(kNN.classify0([0, 0], group, labels, 3))