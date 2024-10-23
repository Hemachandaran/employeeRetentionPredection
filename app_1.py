import pandas as pd

class Preprocessing:
    def __init__(self, df):
        self.df = df
        self.target_mean_feature=[]

    def handle_nulls(self):
        """Fill null values in specific columns."""
        self.df["enrolled_university"] = self.df["enrolled_university"].fillna("none")
        self.df["education_level"] = self.df["education_level"].fillna("Other")
        mode = self.df["experience"].mode()
        self.df["experience"] = self.df["experience"].fillna(mode[0])
        self.df["last_new_job"] = self.df["last_new_job"].fillna("Not Specified")
        self.df['major_discipline'] = self.df['major_discipline'].fillna("Not Specified")
        self.df["gender"] = self.df["gender"].fillna("Not Specified")
        self.df["company_size"] = self.df["company_size"].fillna("Not Specified")
        self.df["company_type"] = self.df["company_type"].fillna("Not Specified")

    def encode_features(self):
        """Encode categorical features with target mean."""
        features = ['gender', "enrolled_university", "major_discipline",
                    "education_level", "company_type", "city"]

        for i,feature in enumerate(features):
            self.target_mean_feature.append(self.df.groupby(feature)['target'].mean())
            self.df[feature] = self.df[feature].map(target_mean_feature[i])

        """Map relevant experience to binary values."""
        self.df["relevent_experience"] = self.df["relevent_experience"].map({
            'Has relevent experience': 1,
            'No relevent experience': 0
        })

        """Map company size categories to numerical values."""
        size_mapping = {
            'Not Specified': 1,
            '<10': 2,
            '10/49': 3,
            '50-99': 4,
            '100-500': 5,
            '500-999': 6,
            '1000-4999': 7,
            '5000-9999': 8,
            '10000+': 9
        }

        self.df["company_size"] = self.df["company_size"].replace(size_mapping)

        """Map last new job categories to numerical values."""
        last_new_job_mapping = {
            'Not Specified': 0,
            '<1': 1,
            '1': 2,
            '2': 3,
            '3': 4,
            '4': 5,
            '5': 6,
            '6': 7,
            '7': 8,
            '8': 9,
            '9': 10,
            '10': 11,
            '11': 12,
            '12': 13,
            '13': 14,
            '14': 15,
            '15': 16,
            '16': 17,
            '17': 18,
            '18': 19,
            '19': 20,
            '>20': 21
        }

        unique_jobs = sorted(self.df["last_new_job"].unique().tolist())

        # Map last new job based on unique values
        for i in unique_jobs:
            if i in last_new_job_mapping:
                self.df["last_new_job"] = self.df["last_new_job"].replace(i, last_new_job_mapping[i])

        """Map experience categories to numerical values."""
        experience_mapping = {
            '>20': 21,
            '<1': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 6,
            '7': 7,
            '8': 8,
            '9': 9,
            '10': 10,
            '11': 11,
            '12': 12,
            '13': 13,
            '14': 14,
            '15': 15,
            '16': 16,
            '17': 17,
            '18': 18,
            '19': 19,
            '20': 20
        }

        self.df["experience"] = self.df["experience"].map(experience_mapping)

    def preprocess(self):
        """Run all preprocessing steps."""
        self.handle_nulls()
        self.encode_features()


# Example usage:
# df = pd.read_csv('your_data.csv')
# preprocessor = Preprocessing(df)
# preprocessor.preprocess()
# processed_df = preprocessor.df