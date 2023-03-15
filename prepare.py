
import os
import pandas as pd

def clean_titanic(df_titanic):
    '''
    Clean titanic will take in a single pandas dataframe
    and will proceed to drop redundant couns 
    and nonuseful information 
    in addition to addressing null values
    and encoding categorical variables
    '''
    
    #impute average age and most common embark_town:
    df_titanic['age'] = df_titanic['age'].fillna(df_titanic.age.mean())
    df_titanic['embark_town'] = df_titanic['embark_town'].fillna('Southhampton')

    #encode categorical values
    df_titanic = pd.concat(
    [df_titanic, pd.get_dummies(df_titanic[['sex','embark_town']], drop_first=True)], axis=1)
    
    df_titanic = df_titanic.drop(columns={'Unnamed: 0','passenger_id','embarked', 'deck','class','sex','embark_town'}).rename(columns={'sibsp' : 'sibling_spouse'})
     
     # convert column names to lowercase, replace '.' in column names with '_'
    df_titanic.columns = [col.lower().replace('.', '_') for col in df_titanic]
    
    return df_titanic







def split_titanic_data(df_titanic):
    '''
     split titanic data will split data based on 
    the values present in a cleaned version of titanic
    that is from clean_titanic
    '''
    
    train_val, test = train_test_split(df_titanic, train_size=0.8, random_state=1349, stratify=df_titanic['survived'])
    
    train, validate = train_test_split(train_val, train_size=0.7, random_state=1349, stratify=train_val['survived'])
    
    return train, validate, test



def prep_iris(df_iris):
    ''' prep iris will take in a single pandas dataframe
    that will presumably match the columns and shape that we 
    expect from our acquire modules get_iris_data 
    functional return 
    '''
    df_iris = df_iris.drop(columns={'species_id','measurement_id','Unnamed: 0'}).rename(columns={'species_name' : 'species'})
    df_iris = pd.concat([df_iris, dummy_df], axis=1)
    df_iris = df_iris

    return df_iris.head()
    
    #pass in original df as the aregument
