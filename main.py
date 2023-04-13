import tensorflow as tf
import numpy as np
import os

# путь к файлам с данными
data_dir = "data"

# загрузка данных
with open(os.path.join(data_dir, "bur-rus.bur")) as f:
    buryat_text = f.read().splitlines()

with open(os.path.join(data_dir, "bur-rus.rus")) as f:
    russian_text = f.read().splitlines()

# выведем первые 5 строк данных
# for i in range(5):
#     print("Buryat: ", buryat_text[i])
#     print("Russian: ", russian_text[i])
#     print()

# параметры модели
vocab_size = 10000
embedding_dim = 256
units = 1024
batch_size = 64
epochs = 10

# токенизация текста на каждом языке
tokenizer_bur = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer_bur.fit_on_texts(buryat_text)
sequences_eng = tokenizer_bur.texts_to_sequences(buryat_text)
padded_eng = tf.keras.preprocessing.sequence.pad_sequences(sequences_eng, padding='post')

tokenizer_rus = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer_rus.fit_on_texts(russian_text)
sequences_fr = tokenizer_rus.texts_to_sequences(russian_text)
padded_fr = tf.keras.preprocessing.sequence.pad_sequences(sequences_fr, padding='post')

# создание модели Encoder-Decoder на основе LSTM
encoder_inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
decoder_inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim)
encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embeddings)
encoder_states = [state_h, state_c]
decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embeddings, initial_state=encoder_states)
dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
outputs = dense(decoder_outputs)
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], outputs)

# обучение модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([padded_eng, padded_fr[:, :-1]], padded_fr[:, 1:], batch_size=batch_size, epochs=epochs)

# # сохранение модели в файл
# model.save('my_translation_model.h5')


# создание функции для перевода текста с бурятского на русский
def translate_text(text):
    # токенизация текста на бурятском
    text_sequence = tokenizer_bur.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, padding='post')

    # # загрузка модели из файла
    # loaded_model = tf.keras.models.load_model('my_translation_model.h5')

    # перевод
    translated_sequence = model.predict([padded_sequence, padded_sequence], verbose=0)
    translated_text = ' '.join(tokenizer_rus.sequences_to_texts([range(np.argmax(i)) for i in translated_sequence])[0].split("<OOV>"))
    return translated_text

# тестирование функции на примере предложения на бурятском
print(translate_text("Баяртай! "))
