import tensorflow as tf

# Chargement des données
def load_data(filename):
  with open(filename, "r") as f:
    data = f.read()
  return data.split("\n")

# Tokenisation des données
def tokenize(data):
  words = set()
  for line in data:
    for word in line.split():
      words.add(word)
  word_to_index = {word: i for i, word in enumerate(words)}
  return [word_to_index[word] for word in data]

# Découpage des données en séquences
def pad_sequences(data, maxlen):
  sequences = []
  for line in data:
    sequence = line[:maxlen]
    if len(sequence) < maxlen:
      sequence += [0] * (maxlen - len(sequence))
    sequences.append(sequence)
  return sequences

# Définition du modèle
def create_model(num_words, maxlen):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Embedding(num_words, 128, input_length=maxlen))
  model.add(tf.keras.layers.LSTM(128))
  model.add(tf.keras.layers.Dense(num_words, activation="softmax"))
  return model

# Entraînement du modèle
def train_model(model, data, epochs):
  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
  model.fit(data, epochs=epochs)

# Génération d'un avis
def generate_review(model, maxlen):
  start_token = tf.keras.backend.constant([word_to_index["<start>"]])
  tokens = start_token
  for _ in range(maxlen):
    predictions = model.predict(tokens)
    next_word = tf.keras.backend.argmax(predictions)
    tokens = tf.concat([tokens, [next_word]], axis=0)
  return " ".join([word_to_index.get(token, "?") for token in tokens])

# Chargement des données
data = load_data("reviews.txt")

# Tokenisation des données
data = tokenize(data)

# Découpage des données en séquences
data = pad_sequences(data, maxlen)

# Définition du modèle
model = create_model(num_words, maxlen)

# Entraînement du modèle
train_model(model, data, 10)

# Génération d'un avis
review = generate_review(model, maxlen)
print(review)