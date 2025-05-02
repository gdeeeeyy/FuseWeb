# Music Generation Using Generative AI – Implement a model that composes original music pieces using MIDI data.

import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from sklearn.preprocessing import LabelEncoder

# --- Step 1: Parse MIDI Files ---
def get_notes_from_midi(folder_path):
    notes = []
    for file in glob.glob(f"{folder_path}/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

notes = get_notes_from_midi("midi_songs")  #  Replace with your MIDI folder

# --- Step 2: Prepare Sequences ---
sequence_length = 100
le = LabelEncoder()
note_int = le.fit_transform(notes)
n_vocab = len(set(note_int))

network_input = []
network_output = []

for i in range(len(note_int) - sequence_length):
    sequence_in = note_int[i:i + sequence_length]
    sequence_out = note_int[i + sequence_length]
    network_input.append(sequence_in)
    network_output.append(sequence_out)

network_input = np.reshape(network_input, (len(network_input), sequence_length, 1)) / float(n_vocab)
network_output = to_categorical(network_output)

# --- Step 3: Build the LSTM Model ---
model = Sequential([
    LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(512),
    Dense(256),
    Dropout(0.3),
    Dense(n_vocab),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')

# --- Step 4: Train the Model ---
model.fit(network_input, network_output, epochs=50, batch_size=64)

# --- Step 5: Generate Music ---
def generate_notes(model, network_input, n_vocab, num_notes=200):
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = le.inverse_transform([index])[0]
        prediction_output.append(result)
        pattern = np.append(pattern[1:], [[index]], axis=0)

    return prediction_output

# --- Step 6: Convert Notes to MIDI ---
def create_midi(prediction_output, filename="output.mid"):
    output_notes = []
    offset = 0
    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = [note.Note(int(n)) for n in pattern.split('.')]
            new_chord = chord.Chord(notes_in_chord)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

generated = generate_notes(model, network_input, n_vocab)
create_midi(generated)
print("Music generated and saved as 'output.mid'")
