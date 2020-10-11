from numpy import *
from os import listdir
import kNN
def img_to_vector(filename):
    return_vector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0,(32*i)+j] = int(line_str[j])
    return  return_vector
def hand_writing_class_test():
    hwlabels = []
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m,1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hwlabels.append(class_num_str)
        training_mat[i,:] = img_to_vector('trainingDigits/%s'%file_name_str)
    test_file_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
            file_name_str = test_file_list[i]
            file_str = file_name_str.split('.')[0]
            class_num_str = int(file_str.split('_')[0])
            vector_under_test = img_to_vector('testDigits/%s'%file_name_str)
            classifier_result = kNN.classify0(vector_under_test,training_mat,hwlabels,3)
            print("the classifier came back with: %d , the real answer is:%d"%(classifier_result,class_num_str))
            if(classifier_result!=class_num_str):
                error_count +=1.0
    print("\nthe total number of error is :%d"%error_count)
    print("\nthe total error rate is: %f"%(error_count/float(m_test)))
