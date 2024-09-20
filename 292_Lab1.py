import numpy as numpy

#Q1
list1 = [-1, 2, 3 ,9 ,0]
list2 = [1,2,7,10,14]
print(list1 + list2)

#Q2
new_list = [0,0,0,0,0]
for i in range(0, len(list1)):
    new_list[i] = list1[i]+list2[i]
print(new_list)

#Q3
my_4d_array = numpy.array([[[[1], [2]], [[3], [4]], [[5], [6]]],[[[7], [8]], [[9], [10]], [[11], [12]]]])
print(my_4d_array.shape)

#Q4
my_3d_array = numpy.arange(27).reshape(3,3,3)
new_list2 = my_3d_array[:,:,[0]]
print(new_list2)
new_list3 = my_3d_array[1,1,:]
print(new_list3)
new_list4 = my_3d_array[:,0::2,0::2]
print(new_list4)

#Q5
new_list5 = [my_3d_array[0,1,1],my_3d_array[1,2,2],my_3d_array[2,0,0]]
print(new_list5)
new_list6 = [my_3d_array[1,0,0],my_3d_array[1,2,2]]
print(new_list6)

#Q6
my_2d_array = numpy.arange(-10,20).reshape(5,6)
index_arr = my_2d_array.sum(axis=0) % 10 == 0
print(my_2d_array[:,index_arr])
