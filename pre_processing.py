from imblearn.over_sampling import SMOTENC
import pandas as pd
import numpy as np

class Preprocessing:
    def __init__(self,df,target_mean_feature ={}):
        """

        df                           : input data,
        imputed_df                   : data after cleaning,
        balance_df                   : data after balancing the target,
        featureEncoded_df            : data after encoded ,
        target_mean_feature          : dictionary containing feature engineering values 
        categorical_features_indices : cointsin indedx of categorical values

        """
        self.df = df
        self.imputed_df = None
        self.balance_df = None
        self.mode= None
        self.featureEncoded_df = None
        self.target_mean_feature =target_mean_feature
        self.categorical_features_indices = []

    def handle_nulls(self,data):
        df=data
        """Fill null values in specific columns."""
        df["enrolled_university"] = df["enrolled_university"].fillna("none")
        df["education_level"] =df["education_level"].fillna("Other")

        # Fill nulls in 'experience' with mode and ensure it's treated as numeric
        if "target" in df.columns:
            self.mode = df["experience"].mode()[0]
            df["experience"] = df["experience"].fillna(self.mode)
        else:
            # Calculate mode for experience
            self.mode = df["experience"].mode()[0] if not df["experience"].mode().empty else None

            # Provide a fallback value if mode is None
            if self.mode is None:
                self.mode = ">20"  # Set this to whatever makes sense in your context
            
            # Fill missing values in experience column
            df["experience"] = df["experience"].fillna(self.mode)

        df["last_new_job"] = df["last_new_job"].fillna("Not Specified")
        df['major_discipline'] = df['major_discipline'].fillna("Not_Specified")
        df["gender"] = df["gender"].fillna("Not_specified")
        df["company_size"] = df["company_size"].fillna("NS")
        df["company_type"] = df["company_type"].fillna("not_specified")
        self.imputed_df = df
        self.df = df
        return df

    def encode_features(self,data):
        df=data
        print("data created")
        if "target" in df.columns:
            print ("inside the target mean feature learning")
            # Encode categorical features with target mean.
            features = ['gender', "enrolled_university", "major_discipline",
                    "education_level", "company_type", "city"]
            print ("feature list created")

            for feature in features:
                print ("inside the feature iterstion loop")
                # Calculate target means for the current feature
                target_means = df.groupby(feature)['target'].mean().to_dict()
                # Store in target_mean_feature dictionary
                self.target_mean_feature[feature] = target_means
               
                # Map the feature values to their corresponding target means
                df[feature] = df[feature].map(target_means)

            # Store in rel_exp dictionary
            rel_exp = {'Has relevent experience': 1, 'No relevent experience': 0}
            # Store in target_mean_feature dictionary
            self.target_mean_feature["relevent_experience"] = rel_exp
            # Map relevant experience to binary values.
            for i, j in rel_exp.items():
                df["relevent_experience"] = df["experience"].replace(i, j)

            # Map company size categories to numerical values using map().
            size_mapping = {'NS': 1, '<10': 2, '10/49': 3, '50-99': 4, '100-500': 5,
                                '500-999': 6, '1000-4999': 7, '5000-9999': 8, '10000+': 9}
            # Store in target_mean_feature dictionary
            self.target_mean_feature["company_size"] = size_mapping
            for i, j in size_mapping.items():
                df["company_size"] = df["company_size"].replace(i, j)

            # Map last new job categories to numerical values.
            u = df["last_new_job"].unique().tolist()
            un = [2, 6, 0, 5, 4, 3, 1]
            for i, j in zip(u, un):
                df["last_new_job"] = df["last_new_job"].replace(i, j)
            last_job_dic={'1': 2, '>4': 6, 'never': 0, '4': 5, '3': 4, '2': 3, np.nan: 1}
            # Store in target_mean_feature dictionary
            self.target_mean_feature["last_new_job"] = last_job_dic

            # Map experience categories to numerical values.
            experience_mapping = {
                    '<1': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                    '10': 10, '11': 11, '12': 12, '13': 13,
                    '14': 14, '15': 15, '16': 16, '17': 17,
                    '18': 18, '19': 19, '20': 20, '>20': 21,
                }
            # Store in target_mean_feature dictionary
            self.target_mean_feature["experience"] = experience_mapping
            
            for i, j in experience_mapping.items():
                df["experience"] = df["experience"].replace(i, j)

        else:
            print("inside the target mean feature else")
            features = ['city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']
            l = list(self.target_mean_feature.keys())
            for x in features:
                print(x+"\n")
                if x in l:
                    print (f"inside the if: {x}\n")
                    k=list(self.target_mean_feature[x].keys())
                    v=list(self.target_mean_feature[x].values())
                    for i,j in zip(k,v):
                        print (f"inside the zip loop {i}->{j}\n")
                        if (str(df[x][0])==i):
                            df[x]=df[x].replace(i,j)
                            break

        
        # Store the encoded DataFrame
        self.featureEncoded_df = df
        self.df = df
        return df

    def handle_outliers(self,df, features):
        """Handle outliers by capping them for specified features."""
        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap the outliers
            df[feature] = np.where(self.df[feature] > upper_bound,
                                         upper_bound,
                                         np.where(self.df[feature] < lower_bound,
                                                  lower_bound,
                                                  self.df[feature]))
        
        print("Outliers have been capped.")
        self.df=df
        
        return df   # Return the modified DataFrame

    def handle_skewness(self,df, features):
        """Handle skewness using log transformation if necessary."""
        for feature in features:
            skewness_value = df[feature].skew()
            if skewness_value > 0.5:  
                # Apply log transformation to reduce skewness
                df[feature] = np.log(df[feature] + 1)  
        
        print("Skewness has been handled using log transformation where applicable.")
        self.df = df        
        return self.df   # Return the modified DataFrame

    def handle_imbalance(self,df):
        """Balance the dataset using SMOTENC."""
        if "target" not in df.columns:
            raise ValueError("Target column not found in DataFrame.")
        X = df.drop("target", axis=1)
        y = df["target"]

        categorical_features_indices = [df.columns.get_loc(col) for col in ['city', 
                                                                                'gender', 
                                                                                'relevent_experience',
                                                                                'enrolled_university', 
                                                                                'education_level',
                                                                                'major_discipline', 
                                                                                'experience',
                                                                                'company_size', 
                                                                                'company_type',
                                                                                'last_new_job']]

        smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)

        X_resampled, y_resampled = smote_nc.fit_resample(X, y)

        # Concatenate the resampled data back into a DataFrame
        self.balance_df = pd.concat([X_resampled, y_resampled], axis=1)

        # Store the balanced DataFrame back into self.df
        self.df = self.balance_df
        return self.balance_df

    def preprocess(self,df):
        """Run all preprocessing steps."""       
       
        # Handle nulls first
        self.handle_nulls(df)
        
        if "target" in df.columns:
            # Handle imbalance only if target exists
            print("Handling imbalance...")
            self.handle_imbalance(df)
        
        print("Encoding features...")
        # Encode features after handling imbalance (if applicable)
        self.encode_features(df)

         # Handle outliers and skewness after encoding
        print("Handling outliers...")
        self.handle_outliers(df,['training_hours', 
                                                    'last_new_job', 
                                                    'city_development_index'])
         
        print("Handling skewness...")
        self.handle_skewness(df,['training_hours', 
                                                    'last_new_job', 
                                                    'city_development_index'])
        
        return self.df   # Return processed DataFrame after encoding and balancing
