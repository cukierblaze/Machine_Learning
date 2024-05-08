import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

np.random.seed(42)
np.set_printoptions(precision=6,suppress=True)

weather=['sunny','rainy','cloudy','rainy','sunny','sunny','cloudy','cloudy','sunny']
temperature=['warm','cold','warm','warm','warm','cool','cool','warm','cold']
walk=['yes','no','yes','no','yes','yes','no','yes','no']
raw_df=pd.DataFrame(data={'weather':weather,'temperature':temperature,'walk':walk})
df=raw_df.copy()

encoder=LabelEncoder()
df['walk']=encoder.fit_transform(walk)
df=pd.get_dummies(df,columns=['weather','temperature'],drop_first=True)
data=df.copy()
target=data.pop('walk')

model=GaussianNB()
model.fit(data,target)
print("Model score",model.score(data,target))
print(encoder.classes_[model.predict(data.iloc[[0]])][0])

