import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


def end():
    quit()


def loop():
    ch7 = input("Do you want to continue[Y/N]: ")
    if ch7 == 'Y':
        loop2()
    else:
        end()


print("Big Mart Sales Analysis")
path = input("Enter path of the file: ")
big_mart_data = pd.read_csv(path)


def loop2():
    print("Big Mart Sales Analysis")
    ch = input("Choose an option: \n [1]Data Analyzation \n [2]Visualisation \n [3]Train And Test \n Your option: ")

    if ch == '1':
        print(big_mart_data)
        ch2 = input("Choose an option: \n [1]Information \n [2]Null Values \n [3]Describe \n Your option: ")
        if ch2 == '1':
            print(big_mart_data.info(), "\nThe dimension of file is: " + str(big_mart_data.shape))
            loop()
        elif ch2 == '2':
            print(big_mart_data.isnull().sum())
            ch3 = input("Do you want to fill the null values[Y/N]? ")
            if ch3 == 'Y':
                big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
                mode_of_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type',
                                                                aggfunc=(lambda x: x.mode()[0]))
                miss_values = big_mart_data['Outlet_Size'].isnull()
                big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values, 'Outlet_Type'].apply(
                    lambda x: mode_of_outlet_size[x])
                print(big_mart_data.isnull().sum())
                print("The mean of numerical feature is: " + str(big_mart_data['Item_Weight'].mean())
                      + "\nAnd the mode for categorical feature is: " + big_mart_data['Outlet_Size'].mode())
                loop()
            else:
                end()
        elif ch2 == '3':
            print(big_mart_data.describe())
            loop()
        else:
            end()

    elif ch == '2':
        sns.set()
        ch4 = input("Choose a feature to visualise: \n [1]Categorical Feature \n [2]Numerical Feature \n Your option: ")

        if ch4 == '1':
            ch5 = input("Choose an option: [1]Item Fat Content \n [2]Item TYpe \n [3]Outlet Size \n Your option: ")
            if ch5 == '1':
                print("Item Fat Content: ")
                plt.figure(figsize=(6, 6))
                sns.countplot(x='Item_Fat_Content', data=big_mart_data)
                plt.show()
                loop()
            elif ch5 == '2':
                print("Item Type: ")
                plt.figure(figsize=(30, 6))
                sns.countplot(x='Item_Type', data=big_mart_data)
                plt.show()
                loop()
            elif ch5 == '3':
                print("Outlet Size: ")
                plt.figure(figsize=(6, 6))
                sns.countplot(x='Outlet_Size', data=big_mart_data)
                plt.show()
                loop()
            else:
                end()

        elif ch4 == '2':
            ch6 = input("Choose an option: [1]Item Weight \n [2]Item Visibility \n [3]Item MRP Distribution"
                        " \n [4]Item Outlet Sales Distribution \n [5]Item Outlet Establishment Year \n Your option: ")
            if ch6 == '1':
                print("Item Weight: ")
                plt.figure(figsize=(6, 6))
                sns.distplot(big_mart_data['Item_Weight'])
                plt.show()
                loop()
            elif ch6 == '2':
                print("Item Visibility: ")
                plt.figure(figsize=(6, 6))
                sns.distplot(big_mart_data['Item_Visibility'])
                plt.show()
                loop()
            elif ch6 == '3':
                print("Item MRP Distribution: ")
                plt.figure(figsize=(6, 6))
                sns.distplot(big_mart_data['Item_MRP'])
                plt.show()
                loop()
            elif ch6 == '4':
                print("Item Outlet Sales Distribution: ")
                plt.figure(figsize=(6, 6))
                sns.distplot(big_mart_data['Item_Outlet_Sales'])
                plt.show()
                loop()
            elif ch6 == '5':
                print("Item Outlet Establishment Year: ")
                plt.figure(figsize=(6, 6))
                sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
                plt.show()
                loop()
            else:
                end()
        else:
            end()
    elif ch == '3':
        ch8 = input("Choose Your Option: \n [1]Data Pre-processing \n [2]Train and Test \n Your options: ")
        if ch8 == '1':
            big_mart_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}},
                                  inplace=True)
            big_mart_data['Item_Fat_Content'].value_counts()
            encoder = LabelEncoder()

            big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
            big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
            big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
            big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
            big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
            big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
            big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

            print(big_mart_data.head())
            loop()
        elif ch8 == '2':
            x = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
            y = big_mart_data['Item_Outlet_Sales']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

            regressor = XGBRegressor()
            regressor.fit(x_train, y_train)

            training_data_prediction = regressor.predict(x_train)
            r2_train = metrics.r2_score(y_train, training_data_prediction)
            print('Training R Squared value = ', r2_train)

            test_data_prediction = regressor.predict(x_test)
            r2_test = metrics.r2_score(y_test, test_data_prediction)
            print('Testing R Squared value = ', r2_test)
            loop()
        else:
            end()


loop2()
