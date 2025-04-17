import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv("Combined_Transcription_Factors.csv")

# Encode Species Labels
label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])

# Normalize Scores
df["Score"] = (df["Score"] - df["Score"].min()) / (df["Score"].max() - df["Score"].min())

X = df[["Score", "Species"]].values
y = df["Score"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def augment_data(X, y, noise_level=0.05):
    noise = np.random.normal(loc=0, scale=noise_level, size=X.shape)
    X_augmented = np.concatenate([X, X + noise], axis=0)
    y_augmented = np.concatenate([y, y], axis=0)
    return X_augmented, y_augmented

X_train, y_train = augment_data(X_train, y_train)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5)

model = Sequential([
    Dense(64, kernel_regularizer=l2(0.001), input_shape=(X.shape[1],)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),

    Dense(32, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),

    Dense(16, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.2),

    Dense(1, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001), loss='mse', metrics=['mae'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1,
          callbacks=[early_stopping, lr_scheduler])

# Evaluate Model
eval_results = model.evaluate(X_test, y_test, verbose=1)
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

print(f"Test Loss: {eval_results[0]}, Test MAE: {eval_results[1]}")
print(f"Model Accuracy (R² Score): {accuracy}")



def classify_species(row):
    protein_id = str(row['Protein_ID']).upper()
    score = float(row['Score']) if pd.notna(row['Score']) else 0
    species = str(row['Species']).upper()
    identified = str(row.get('Identified_Species', '')).lower()  

    prokaryotic_patterns = ['XP_', 'WP_', 'GP_']
    eukaryotic_patterns = ['YP_', 'AP_', 'NP_']

    high_confidence_threshold = 0.8
    low_confidence_threshold = 0.5

    if any(pid in protein_id for pid in prokaryotic_patterns):
        if score >= high_confidence_threshold:
            return 'Prokaryote'
        elif score >= low_confidence_threshold:
            return 'Prokaryote'
        else:
            return 'Unknown'

    if any(pid in protein_id for pid in eukaryotic_patterns):
        if score >= high_confidence_threshold:
            return 'Eukaryote'
        elif score >= low_confidence_threshold:
            return 'Eukaryote'
        else:
            return 'Unknown'

    prokaryotic_keywords = ['prok', 'prokary', 'bacter', 'archae']
    if any(kw in identified for kw in prokaryotic_keywords):
        return 'Prokaryote'
    if species in ['NC', 'EC', 'BS', 'PA']:
        return 'Prokaryote'

    eukaryotic_keywords = ['euk', 'eukary', 'fung', 'animal', 'plant']
    if any(kw in identified for kw in eukaryotic_keywords):
        return 'Eukaryote'
    if species in ['SC', 'HS', 'MM', 'DM']:
        return 'Eukaryote'

    return 'Unknown'

def extract_features(protein_id):
    features = {
        'starts_with_NP': int(protein_id.startswith('NP_')),
        'starts_with_XP': int(protein_id.startswith('XP_')),
        'starts_with_YP': int(protein_id.startswith('YP_')),
        'has_numbers': int(bool(re.search(r'\d', protein_id))),
        'length': len(protein_id)
    }
    return features

def train_model(data):
    X = data.apply(lambda row: extract_features(row['Protein_ID']), axis=1, result_type='expand')
    X['Score'] = data['Score']
    y = data['Classification'].map(lambda x: 'Eukaryote' if 'Eukaryote' in x else 'Prokaryote')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("\nModel Evaluation:")
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, 'species_classifier.joblib')
    return model

def main():
    data = pd.read_csv('Combined_Transcription_Factors.csv')
    data.drop_duplicates(subset='Protein_ID', inplace=True)
    data['Classification'] = data.apply(classify_species, axis=1)
    model = train_model(data)

    while True:
        protein_id = input("\nEnter Protein ID (or 'quit' to exit): ")
        if protein_id.lower() == 'quit':
            break
        score = float(input("Enter Score (0-1): "))

        features = extract_features(protein_id)
        features['Score'] = score
        features_df = pd.DataFrame([features])

        prediction = model.predict(features_df)[0]
        confidence = model.predict_proba(features_df).max()

        print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")

def extract_features_proteinid_only(protein_id):
    features = {
        'starts_with_NP': int(protein_id.startswith('NP_')),
        'starts_with_XP': int(protein_id.startswith('XP_')),
        'starts_with_YP': int(protein_id.startswith('YP_')),
        'has_numbers': int(bool(re.search(r'\\d', protein_id))),
        'length': len(protein_id)
    }
    return features

def classify_species_proteinid_only(row):
    protein_id = str(row['Protein_ID']).upper()
    species = str(row['Species']).upper()
    identified = str(row.get('Identified_Species', '')).lower()

    prokaryotic_patterns = ['YP_', 'WP_', 'GP_','XP_']
    eukaryotic_patterns = ['XP_', 'AP_', 'NP_']

    if any(pid in protein_id for pid in prokaryotic_patterns):
        return 'Prokaryote'
    if any(pid in protein_id for pid in eukaryotic_patterns):
        return 'Eukaryote'

    prokaryotic_keywords = ['prok', 'prokary', 'bacter', 'archae']
    if any(kw in identified for kw in prokaryotic_keywords):
        return 'Prokaryote'
    if species in ['NC', 'EC', 'BS', 'PA']:
        return 'Prokaryote'

    eukaryotic_keywords = ['euk', 'eukary', 'fung', 'animal', 'plant']
    if any(kw in identified for kw in eukaryotic_keywords):
        return 'Eukaryote'
    if species in ['SC', 'HS', 'MM', 'DM']:
        return 'Eukaryote'

    return 'Unknown'

def train_model_proteinid_only(data):
    X = data['Protein_ID'].apply(lambda pid: extract_features_proteinid_only(str(pid).upper()))
    X = pd.DataFrame(list(X))
    y = data['Classification'].map(lambda x: 'Eukaryote' if 'Eukaryote' in x else 'Prokaryote')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("\\nModel Evaluation:")
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, 'species_classifier_proteinid_only.joblib')
    return model

def main_full_model():
    data = pd.read_csv('Combined_Transcription_Factors.csv')
    data.drop_duplicates(subset='Protein_ID', inplace=True)
    data['Classification'] = data.apply(classify_species, axis=1)
    model = train_model(data)

    while True:
        protein_id = input("Enter Protein ID (or 'quit' to exit): ")
        if protein_id.lower() == 'quit':
            break
        score = float(input("Enter Score (0-1): "))

        features = extract_features(protein_id)
        features['Score'] = score
        features_df = pd.DataFrame([features])

        prediction = model.predict(features_df)[0]
        confidence = model.predict_proba(features_df).max()

        print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")

def main_proteinid_only():
    print("=== Protein ID Species Predictor (No Score Required) ===")
    data = pd.read_csv('Combined_Transcription_Factors.csv')
    data.drop_duplicates(subset='Protein_ID', inplace=True)
    data['Classification'] = data.apply(classify_species_proteinid_only, axis=1)
    model = train_model_proteinid_only(data)

    while True:
        protein_id = input("Enter Protein ID (or 'quit' to exit): ").strip()
        if protein_id.lower() == 'quit':
            break

        match = data[data['Protein_ID'].str.upper() == protein_id.upper()]
        if not match.empty:
            classification = match.iloc[0]['Classification']
            print(f"Classification for Protein ID '{protein_id}': {classification}")
        else:
            features = extract_features_proteinid_only(protein_id.upper())
            features_df = pd.DataFrame([features])

            prediction = model.predict(features_df)[0]
            confidence = model.predict_proba(features_df).max()

            print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")

def main():
    print("Select mode:")
    print("1 - Full model (Score + Species)")
    print("2 - Protein ID only model")
    choice = input("Enter choice (1 or 2): ").strip()
    if choice == '1':
        main_full_model()
    elif choice == '2':
        main_proteinid_only()
    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()
