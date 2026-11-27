import pandas as pd
from sklearn.preprocessing import LabelEncoder
def preprocess(df):
    df = df.copy()

    #Title
    df["Title"] = df["Name"].str.extract(
        r'([A-Za-z]+)\.',
        expand = False
    )
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona"
        ],
        "Rare"
    )

    df["Title"] = df["Title"].replace("Mlle", "Miss")
    df["Title"] = df["Title"].replace("Ms", "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")
    title_map = {
        "Mr": 0,
        "Miss": 1,
        "Mrs": 2,
        "Master": 3,
        "Rare": 4
    }

    df["Title"] = df["Title"].map(title_map)

    #CabinKnown & CabinLetter
    df["CabinKnown"] = (
        df["Cabin"].notnull()
    ).astype(int)
    df["CabinLetter"] = df["Cabin"].str[0]
    df["CabinLetter"] = df["CabinLetter"].fillna("U")
    cabin_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "T": 7,
        "U": 8
    }

    df["CabinLetter"] = df["CabinLetter"].map(cabin_map)

    #Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    #Age
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # AgeBand
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[0, 16, 32, 48, 64, 120],
        labels=False,
        include_lowest=True
    )

    #Fare
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # FareBand
    df["FareBand"] = pd.qcut(
        df["Fare"],
        4,
        labels=False,
        duplicates="drop"
    )

    #Embarked
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    #FamilySize and Type
    df["FamilySize"] = (df["SibSp"] + df["Parch"] + 1)
    df['FamilyType'] = pd.cut(df['FamilySize'], bins=[0, 1, 4, 20], labels=['Single', 'Small', 'Large'])
    df["FamilyType"] = df["FamilyType"].cat.codes

    #IsAlone
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    #IsWoman Or Child
    df['IsWomanOrChild'] = ((df['Sex'] == 1) | (df['Age'] < 14) | (df['Title'] == 3)).astype(int)

    return df