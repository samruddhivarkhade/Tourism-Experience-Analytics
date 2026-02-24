import pandas as pd
import os


class DataPipeline:

    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path

    # --------------------------------------------------
    # Load Excel File
    # --------------------------------------------------
    def load_excel(self, filename):
        return pd.read_excel(os.path.join(self.raw_path, filename))

    # --------------------------------------------------
    # Load All Files
    # --------------------------------------------------
    def load_all_data(self):
        print("📥 Reading Excel files...")

        self.transactions = self.load_excel("Transaction.xlsx")
        self.users = self.load_excel("User.xlsx")
        self.cities = self.load_excel("City.xlsx")
        self.countries = self.load_excel("Country.xlsx")
        self.regions = self.load_excel("Region.xlsx")
        self.continents = self.load_excel("Continent.xlsx")
        self.attractions = self.load_excel("Updated_Item.xlsx")
        self.types = self.load_excel("Type.xlsx")

        print("✅ All files loaded successfully.")

    # --------------------------------------------------
    # Merge Everything Safely
    # --------------------------------------------------
    def merge_data(self):
        print("🔗 Merging datasets...")

        # 1️⃣ Merge Transaction + User
        df = pd.merge(
            self.transactions,
            self.users,
            on="UserId",
            how="left"
        )

        # 2️⃣ Merge Attraction
        df = pd.merge(
            df,
            self.attractions,
            on="AttractionId",
            how="left"
        )

        # 3️⃣ Merge Attraction Type
        df = pd.merge(
            df,
            self.types,
            on="AttractionTypeId",
            how="left"
        )

        # --------------------------------------------------
        # USER GEOGRAPHY BLOCK
        # --------------------------------------------------

        # Merge Country for User
        df = pd.merge(
            df,
            self.countries,
            on="CountryId",
            how="left",
            suffixes=("", "_UserCountry")
        )

        # Merge Region (User)
        df = pd.merge(
            df,
            self.regions,
            on="RegionId",
            how="left",
            suffixes=("", "_UserRegion")
        )

        # Merge Continent (User)
        df = pd.merge(
            df,
            self.continents,
            on="ContinentId",
            how="left"
        )

        # --------------------------------------------------
        # ATTRACTION GEOGRAPHY BLOCK
        # --------------------------------------------------

        # Merge City (Attraction location)
        df = pd.merge(
            df,
            self.cities,
            left_on="AttractionCityId",
            right_on="CityId",
            how="left",
            suffixes=("", "_AttractionCity")
        )

        print("✅ Merge completed successfully.")
        print("Shape after merge:", df.shape)
        return df

    # --------------------------------------------------
    # Clean Data
    # --------------------------------------------------
    def clean_data(self, df):
        print("🧹 Cleaning data...")

        df.drop_duplicates(inplace=True)

        # Keep valid ratings
        if "Rating" in df.columns:
            df = df[df["Rating"].between(1, 5)]

        # Create proper VisitDate column
        if "VisitYear" in df.columns and "VisitMonth" in df.columns:
            df["VisitDate"] = pd.to_datetime(
                df["VisitYear"].astype(str) + "-" +
                df["VisitMonth"].astype(str) + "-01",
                errors="coerce"
            )

        # Fill categorical columns with "Unknown"
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna("Unknown")

        # Fill numeric columns with 0
        for col in df.select_dtypes(include=["int64", "float64"]).columns:
            df[col] = df[col].fillna(0)

        print("✅ Cleaning completed.")
        return df

    # --------------------------------------------------
    # Save Processed File
    # --------------------------------------------------
    def save_processed(self, df):
        os.makedirs(self.processed_path, exist_ok=True)

        output_path = os.path.join(
            self.processed_path,
            "master_dataset.csv"
        )

        df.to_csv(output_path, index=False)
        print(f"💾 Processed dataset saved at: {output_path}")

    # --------------------------------------------------
    # Run Pipeline
    # --------------------------------------------------
    def run(self):
        self.load_all_data()
        df = self.merge_data()
        df = self.clean_data(df)
        self.save_processed(df)

        print("🎉 Data pipeline completed successfully.")