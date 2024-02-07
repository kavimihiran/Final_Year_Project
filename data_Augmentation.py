import pandas as pd
import numpy as np

df = pd.read_csv('datasets_with_sentiment_Score/bitcoin_sentiments_usable_filled.csv')  

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])


# Apply time warping by randomly stretching or compressing time
def time_warping(df):
    factor = np.random.uniform(0.9, 1.1)  
    days_to_add = pd.to_timedelta(np.random.uniform(-5, 5), unit='D') 

    # Apply time warping
    df['Date'] = df['Date'] + days_to_add

    return df

# Apply noise injection
def add_noise(df, noise_factor=0.05):
    noise = np.random.normal(0, noise_factor, len(df))
    df['Close'] = df['Close'] + noise
    return df

# Apply window slicing
def window_slicing(df, window_size=5):
    if len(df) <= window_size:
        return df

    # Choose a random starting index
    start_idx = np.random.randint(0, len(df) - window_size + 1)

    # Slice the dataframe to get the window
    window_df = df.iloc[start_idx:start_idx + window_size]

    return window_df

# Number of augmented samples to generate
num_augmented_samples = 1000

# Generate and combine augmented data
augmented_data = []
for _ in range(num_augmented_samples):
    augmented_df = df.copy()
    augmented_df = time_warping(augmented_df)
    augmented_df = add_noise(augmented_df)
    augmented_df = window_slicing(augmented_df)
    augmented_data.append(augmented_df)

# Concatenate augmented data with the original DataFrame
augmented_df = pd.concat([df] + augmented_data, ignore_index=True)
# Save the combined dataset to a new CSV file
augmented_df.to_csv('datasets_with_sentiment_Score/BTC_augmented_dataset.csv', index=False)

print(f"Augmented dataset saved to augmented_dataset.csv")
