# 🐦 Twitter/X Sentiment Analysis

## 📋 Project Overview
This project focuses on sentiment analysis of tweets from Twitter/X to determine whether the sentiment expressed is positive, negative, or neutral. Using natural language processing (NLP) techniques combined with machine learning algorithms, the project aims to provide businesses and organizations with real-time insights into public opinion, helping them adapt strategies based on user feedback.

## 🚀 Use Case
- **Social Media Monitoring**: Track public discourse and sentiment on Twitter/X around specific topics, brands, or events.
- **Brand Sentiment**: Analyze user sentiment regarding a brand, product, or campaign.
- **Customer Feedback**: Automatically categorize feedback from Twitter/X users to enhance customer service or product development.

## 🛠️ Tools and Technologies Used
1. **Python**: Used for data manipulation and machine learning model training.
2. **Natural Language Processing (NLP)**:
   - **NLTK (Natural Language Toolkit)**: Tokenization, stemming, and stopword removal.
   - **TextBlob**: For sentiment polarity analysis.
3. **Machine Learning Models**:
   - **Logistic Regression**: A simple, effective linear model for classification tasks.
   - **Support Vector Machines (SVM)**: A robust classifier, especially effective for text classification.
   - **Multinomial Naive Bayes**: Well-suited for text-based features.
4. **Libraries**:
   - **Pandas**: Data manipulation and preprocessing.
   - **Scikit-learn**: Model training, evaluation, and hyperparameter tuning.
   - **Matplotlib & Seaborn**: Visualizations of sentiment analysis results.

## 📊 Data Preprocessing
1. **Data Collection**:
   - Tweets were collected using the Twitter API or scraping techniques, based on specific keywords or hashtags.
   
2. **Text Preprocessing**:
   - **Lowercasing**: Converts all text to lowercase for consistency.
   - **Stopword Removal**: Removes common words (e.g., "and", "the") that do not add sentiment value.
   - **Tokenization**: Splits text into individual words (tokens).
   - **Stemming and Lemmatization**: Reduces words to their base or root form.
   - **Punctuation Removal**: Cleans up text by removing punctuation marks.
   
3. **Feature Extraction**:
   - **Bag of Words (BoW)**: Converts text into a frequency representation of words.
   - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weights words based on their frequency across documents.

## 🧠 Model Training and Evaluation
The models were trained on the preprocessed dataset, and their performances were evaluated using accuracy, precision, recall, and F1 score metrics.

1. **Logistic Regression**:
   - Accuracy: 82%
   - Precision: 81%
   - Recall: 80%
   - F1 Score: 80.5%
   
2. **Support Vector Machine (SVM)**:
   - Accuracy: 84%
   - Precision: 83%
   - Recall: 82%
   - F1 Score: 82.5%
   
3. **Multinomial Naive Bayes**:
   - Accuracy: 79%
   - Precision: 78%
   - Recall: 77%
   - F1 Score: 77.5%

## 📈 Visualizations
- **Bar Charts**: Display the distribution of sentiments (positive, negative, neutral) across the dataset.
- **Word Clouds**: Visualize the most frequent words in positive and negative tweets.

## 🔍 Key Insights
- **Positive Sentiment**: Most tweets about the selected topic are positive, indicating favorable public opinion.
- **Negative Sentiment**: A significant portion of tweets express negative sentiment, pointing to potential areas of concern.
- **Neutral Sentiment**: Neutral tweets often reflect objective or fact-based discussions without strong opinions.

## 📝 How to Run the Project
1. **Install the required libraries**:
   ```bash
   pip install nltk pandas scikit-learn matplotlib seaborn textblob

## 📌 Future Improvements
Advanced Models: Implement deep learning models like LSTMs (Long Short-Term Memory) to improve sentiment classification accuracy.
Real-time Sentiment Analysis: Integrate real-time sentiment tracking for live tweet analysis.
Fine-tuning: Perform hyperparameter tuning to further optimize the machine learning models' performance.
