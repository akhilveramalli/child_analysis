import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

data = pd.read_excel('C:/Users/bunny/Desktop/AKHIL/Project_66/Data.xlsx ')





# Calculating ED and BD Subsclaes and adding to the dataset

columns = data.columns

data_output = data.iloc[:, 40:56]
columns2 = data_output.columns #
columns2 = columns2.delete(14) # never, sometimes, always values for this column are different so dropping that column

for i in range(15):
    data_output[columns2[i]].replace({"Never": 0, "Sometimes": 1, "Always": 2}, inplace=True)

data_output['24. Remember, there are no right or wrong answers, just pick which is right for you. [I am calm]'].replace({"Never": 2, "Sometimes": 1, "Always": 0}, inplace=True)

# Emotional Diffiulties
ED_outputs = data_output.iloc[:, :10]
ED_outputs.loc[:,'ED_subscale'] = ED_outputs.sum(axis=1)

for i in range(len(data)):
    if ED_outputs.ED_subscale[i] < 10:
        ED_outputs.ED_subscale[i] = 'Expected'
    elif ED_outputs.ED_subscale[i] > 11:
        ED_outputs.ED_subscale[i] = 'Significant difficulties'
    else:
        ED_outputs.ED_subscale[i] = 'Borderline difficulties'


#data['ED_subscale'] = ED_outputs['ED_subscale']

# Behavioural Dificulties
BD_outputs = data_output.iloc[:, 10:]
BD_outputs.loc[:,'BD_subscale'] = BD_outputs.sum(axis=1)

for i in range(len(data)):
    if BD_outputs.BD_subscale[i] < 6:
        BD_outputs.BD_subscale[i] = 'Expected'
    elif BD_outputs.BD_subscale[i] > 6:
        BD_outputs.BD_subscale[i] = 'Significant difficulties'
    else:
        BD_outputs.BD_subscale[i] = 'Borderline difficulties'

#data['BD_subscale'] = BD_outputs['BD_subscale']








df = data.iloc[:,[14, 16, 17, 21, 25, 34, 35, 37, 38, 39]]

columns3 = df.columns

mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df= pd.DataFrame(mode_imputer.fit_transform(df))

df.info()

df.columns = ['sports_activity_in_a_week','felling_tired_in_a_week', 'concentration_on_your_school_work', 'safe_scale_in_your_area', 'how_many_times_u_going_outside_to_play', 'health','school','friends','appearance','life']

columns4 = df.columns



df.sports_activity_in_a_week.value_counts()
df['sports_activity_in_a_week']=df['sports_activity_in_a_week'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})


df.felling_tired_in_a_week.value_counts()
df['felling_tired_in_a_week']=df['felling_tired_in_a_week'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})


df.concentration_on_your_school_work.value_counts()
df['concentration_on_your_school_work']=df['concentration_on_your_school_work'].replace({'1-2 days':1,'3-4 days':3,'5-6 days':5,'7 days':7,'0 days':0})


df.how_many_times_u_going_outside_to_play.value_counts()
df['how_many_times_u_going_outside_to_play']= le.fit_transform(df['how_many_times_u_going_outside_to_play'])

#  most days - 3
#  A few days a week - 0
#  Hardly ever 1
#  I dont't play - 2


df['ED_subscale'] = ED_outputs['ED_subscale']




# df.safe_scale_in_your_area = df.safe_scale_in_your_area.astype("int64")
# df.health = df.health.astype("int64")
# df.school = df.school.astype("int64")
# df.friends = df.friends.astype("int64")
# df.life = df.life.astype("int64")
# df.appearance = df.appearance.astype("int64")

df.info()

df.life.value_counts() # has no 0 so 1-0, 2-1 so on 
df['life']= le.fit_transform(df['life'])

df.safe_scale_in_your_area.value_counts() # has 0 so same value 0-0, 1-1, 2-2
df['safe_scale_in_your_area']= le.fit_transform(df['safe_scale_in_your_area'])

df.health.value_counts() # has 0 but not having 1 so  value 0-0, 2-1, 3-2,4-3,5-4, 6-5, so on
df['health']= le.fit_transform(df['health']) 

df.school.value_counts() # has 0  so same value 
df['school']= le.fit_transform(df['school'])

df.friends.value_counts() # has 0  so same value 
df['friends']= le.fit_transform(df['friends'])

df.appearance.value_counts() # has 0  so same value 
df['appearance']= le.fit_transform(df['appearance'])


x = df.iloc[:,0:10]
y = df.iloc[:,10]

X_train,X_test,y_train,y_test = train_test_split(x , y , train_size=0.7)

model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(X_train,y_train)
# help(LogisticRegression)

y_pred = model.predict(X_test)


print(accuracy_score(y_test,y_pred)) #0.90
print(classification_report(y_test,y_pred))

y_pred=model.predict(X_train)
print(accuracy_score(y_train,y_pred)) #0.88
print(classification_report(y_train,y_pred))



k = model.predict(pd.DataFrame(X_train.iloc[2:3, :]))

z = pd.DataFrame(X_train.iloc[2:3, :])
print(k[0])



import pickle
pickle.dump(model, open('model.pkl', 'wb'))



# data = {'Name':['Tom', 'nick', 'krish', 'jack'],
#         'Age':[20, 21, 19, 18]}
  
# # Create DataFrame
# df = pd.DataFrame(data)



m = [3,4,6,4,9,6,7,4,4,1]

final_features = [int(x) for x in m]
final_features = [np.array(final_features)]
prediction = model.predict(final_features)



final_features = [float(x) for x in request.form.values()]
final_features = [np.array(final_features)]
prediction = model.predict(final_features)
