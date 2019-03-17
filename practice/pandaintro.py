import pandas as pd 
groceries = pd.Series(data=[30, 6, 'Yes', 'No'], index=['eggs', 'apples', '1', 2])
print(groceries.shape, groceries.ndim, groceries.size)
print('as' in groceries)
#for ind in groceries.index:
#	print(ind, groceries[ind])
dic = {'Bob':pd.Series([1, 2, 4], [1, 2 ,3]), 'Alex':pd.Series([100, 200, 300, 400], [1, 2, 3, 4])}
temp = pd.DataFrame(dic)
print(temp)
