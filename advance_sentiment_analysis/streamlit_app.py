import pandas as pd 
df = pd.read_csv("data.csv")
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("data.csv")

# 2. Define inputs (X) and outputs (y)
X = df['text']             # input text
y_sentiment = df['sentiment']   # sentiment labels
y_ctype = df['ctype']           # comment type labels

# 3. Split for sentiment classification
X_train, X_test, y_sentiment_train, y_sentiment_test = train_test_split(
    X, y_sentiment,
    test_size=0.2,
    random_state=42,
    stratify=y_sentiment
)

# 4. Split for comment type classification
X_train_c, X_test_c, y_ctype_train, y_ctype_test = train_test_split(
    X, y_ctype,
    test_size=0.2,
    random_state=42,
    stratify=y_ctype
)

# 5. Show dataset sizes
print("=== Sentiment split ===")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_sentiment_train shape:", y_sentiment_train.shape)
print("y_sentiment_test shape:", y_sentiment_test.shape)

print("\n=== Ctype split ===")
print("X_train_c shape:", X_train_c.shape)
print("X_test_c shape:", X_test_c.shape)
print("y_ctype_train shape:", y_ctype_train.shape)
print("y_ctype_test shape:", y_ctype_test.shape)

# 6. Show some rows from sentiment split
print("\nSample Sentiment Training Data:")
for i in range(5):
    print(f"Text: {X_train.iloc[i]}")
    print(f"Sentiment: {y_sentiment_train.iloc[i]}")
    print("---")

# 7. Show some rows from ctype split
print("\nSample Ctype Training Data:")
for i in range(5):
    print(f"Text: {X_train_c.iloc[i]}")
    print(f"Ctype: {y_ctype_train.iloc[i]}")
    print("---")


