import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Load the dataset with a raw string
data = pd.read_csv(r'C:\Users\user\OneDrive\Desktop\oriserve data science\Evaluation-dataset.csv')
print("Data loaded successfully.")

# Print the columns of the DataFrame
print("Columns in the dataset:", data.columns)

# Rename the first column to 'review'
data.rename(columns={data.columns[0]: 'review'}, inplace=True)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess the text: tokenize and remove stop words
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word.lower() not in stop_words]
    return words

# Apply preprocessing to the 'review' column
data['processed_text'] = data['review'].apply(preprocess)
print("Text preprocessing completed.")
print(data['processed_text'].head())

# Sentiment analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    # Determine the sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Apply sentiment analysis
data['sentiment'] = data['review'].apply(get_sentiment)
print("Sentiment analysis completed.")
print(data[['review', 'sentiment']].head())

# Identifying subthemes
subthemes = {
    'garage service': ['garage', 'service'],
    'wait time': ['wait', 'delay', 'time'],
    'incorrect tyres': ['incorrect', 'wrong', 'tyre', 'tyres']
}

def identify_subthemes(text):
    themes = {}
    for theme, keywords in subthemes.items():
        for keyword in keywords:
            if keyword in text:
                themes[theme] = get_sentiment(text)
    return themes

# Apply subtheme identification
data['subthemes'] = data['review'].apply(lambda x: identify_subthemes(x.lower()))
print("Subtheme identification completed.")
print(data[['review', 'subthemes']].head())

# Save the results to a new CSV file (optional)
data.to_csv('processed_reviews.csv', index=False)
print("Processed data saved to 'processed_reviews.csv'.")
