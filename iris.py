import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        # Load Iris dataset from CSV URL
        csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        col_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']
        df = pd.read_csv(csv_url, names=col_names)

        # Task 1: Data Exploration
        print("First 5 rows of the dataset:")
        print(df.head())

        # Check for missing values
        missing = df.isnull().sum()
        print("\nMissing values per column:")
        print(missing)

        # No missing values found, so no cleaning needed
        if missing.sum() == 0:
            print("\nNo missing values detected, no cleaning required.")

        print("\nData types and info:")
        print(df.info())

        # Task 2: Basic Data Analysis
        print("\nBasic statistics for numerical columns:")
        print(df.describe())

        print("\nMean values per species:")
        print(df.groupby('species').mean())

        # Observations
        print("\nObservations:")
        print("- Setosa species generally have smaller sepal and petal lengths and widths.")
        print("- Virginica species tend to have largest measurements.")
        print("- Versicolor species measurements fall between setosa and virginica.")

        # Task 3: Data Visualizations

        # Line Chart: Since no time data, use species index for line chart of mean sepal length
        plt.figure(figsize=(10, 6))
        mean_sepal_length = df.groupby('species')['sepal length (cm)'].mean()
        mean_sepal_length.plot(kind='line', marker='o')
        plt.title('Mean Sepal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Mean Sepal Length (cm)')
        plt.grid(True)
        plt.show()

        # Bar Chart: Average Petal Length per Species
        plt.figure(figsize=(10, 6))
        sns.barplot(x='species', y='petal length (cm)', data=df)
        plt.title('Average Petal Length per Species')
        plt.xlabel('Species')
        plt.ylabel('Petal Length (cm)')
        plt.show()

        # Histogram: Distribution of Sepal Width
        plt.figure(figsize=(10, 6))
        df['sepal width (cm)'].hist(bins=20)
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        plt.show()

        # Scatter Plot: Sepal Length vs Petal Length colored by species
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
        plt.title('Sepal Length vs Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend(title='Species')
        plt.show()

    except FileNotFoundError:
        print("Error: Dataset file not found at the URL.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
