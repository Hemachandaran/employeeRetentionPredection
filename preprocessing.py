from imblearn.over_sampling import SMOTENC
import pandas as pd

class Preprocessing:
    def __init__(self, df):
        self.df = df
        self.imputed_df = None
        self.balance_df = None
        self.featureEncoded_df = None
        self.target_mean_feature = []
        self.categorical_features_indices = []




    def handle_nulls(self):
      """Fill null values in specific columns."""
      self.df["enrolled_university"] = self.df["enrolled_university"].fillna("none")
      self.df["education_level"] = self.df["education_level"].fillna("Other")

      # Fill nulls in 'experience' with mode and ensure it's treated as numeric
      mode = self.df["experience"].mode()[0]
      self.df["experience"] = self.df["experience"].fillna(mode)  # Ensure it's float

      self.df["last_new_job"] = self.df["last_new_job"].fillna("Not Specified")
      self.df['major_discipline'] = self.df['major_discipline'].fillna("Not_Specified")
      self.df["gender"] = self.df["gender"].fillna("Not_specified")
      self.df["company_size"] = self.df["company_size"].fillna("NS")
      self.df["company_type"] = self.df["company_type"].fillna("not_specified")
      self.imputed_df = self.df
      return self.df

    def encode_features(self):
        # Encode categorical features with target mean.
        features = ['gender', "enrolled_university", "major_discipline",
                    "education_level", "company_type", "city"]

        for i, feature in enumerate(features):
            self.target_mean_feature.append(self.df.groupby(feature)['target'].mean())
            self.df[feature] = self.df[feature].map(self.target_mean_feature[i])

        rel_exp={'Has relevent experience':1,'No relevent experience':0}
        # Map relevant experience to binary values.
        self.df["relevent_experience"] = self.df["relevent_experience"].map(rel_exp)

        # Map company size categories to numerical values using map().
        size_mapping = {'NS':1,'<10': 2,'10/49': 3,'50-99': 4,'100-500': 5,'500-999': 6,'1000-4999': 7,'5000-9999': 8,'10000+': 9}
        self.df["company_size"]=self.df["company_size"].map(size_mapping)


        # Map last new job categories to numerical values.

        u=self.df["last_new_job"].unique().tolist()
        u.sort()
        un=[2,3,4,5,6,0,1]

        for i,j in zip(u,un):
          self.df["last_new_job"]=self.df["last_new_job"].replace(i,j)


        # Map experience categories to numerical values.

        experience_mapping = {'<1': 0,'1': 1,'2': 2,'3': 3,'4': 4,\
                              '5': 5,'6': 6,'7': 7,'8': 8,'9': 9,\
                              '10': 10,'11': 11,'12': 12,'13': 13,\
                              '14': 14,'15': 15,'16': 16,'17': 17,\
                              '18': 18,'19': 19,'20': 20,'>20': 21,}
        for i,j in zip(experience_mapping.keys(),experience_mapping.values()):
          self.df["experience"]=self.df["experience"].replace(i,j)

        self.featureEncoded_df = self.df
        return self.df

    def balance_data(self):
        """Balance the dataset using SMOTENC."""
        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        categorical_features_indices = [self.df.columns.get_loc(col) for col in['city', 'gender','relevent_experience',\
                                                                                'enrolled_university','education_level',\
                                                                                'major_discipline', 'experience',\
                                                                                'company_size', 'company_type',\
                                                                                'last_new_job']]

        smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)

        X_resampled, y_resampled = smote_nc.fit_resample(X, y)

        # Concatenate the resampled data back into a DataFrame
        self.balance_df = pd.concat([X_resampled, y_resampled], axis=1)

        # Store the balanced DataFrame back into self.df
        self.df = self.balance_df
        return self.balance_df

    def preprocess(self):
        """Run all preprocessing steps."""
        self.inputed_df = self.handle_nulls()
        self.balance_df = self.balance_data()
        self.featureEncoded_df = self.encode_features()
        return self.df

