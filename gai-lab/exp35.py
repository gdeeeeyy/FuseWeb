# Music Generation with RNNs: Train an LSTM model to generate music sequences.

# pip install torch numpy music21

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from music21 import converter, instrument, note, chord
import glob
import os

# Step 1: Load and preprocess the MIDI files
def prepare_midi_data(midi_folder):
    all_notes = []
    for midi_file in glob.glob(os.path.join(midi_folder, "*.mid")):
        midi = converter.parse(midi_file)

        notes = []
        for element in midi.flat.notes:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        all_notes.append(notes)

    # Flatten the list of notes and create unique mappings
    all_notes = [note for sublist in all_notes for note in sublist]
    unique_notes = sorted(set(all_notes))
    note_to_int = {note: number for number, note in enumerate(unique_notes)}
    int_to_note = {number: note for number, note in enumerate(unique_notes)}
    return all_notes, note_to_int, int_to_note

# Prepare dataset
midi_folder = 'path_to_your_midi_files'  # Replace with your MIDI files folder
all_notes, note_to_int, int_to_note = prepare_midi_data(midi_folder)

# Step 2: Prepare input sequences and target sequences for training
sequence_length = 100  # Number of notes to consider for each training sequence

def create_sequences(notes, sequence_length=100):
    network_input = []
    network_output = []
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in seq_in])
        network_output.append(note_to_int[seq_out])
    return np.array(network_input), np.array(network_output)

network_input, network_output = create_sequences(all_notes, sequence_length)

# Convert to PyTorch tensors
X_train = torch.Tensor(network_input).long()
y_train = torch.Tensor(network_output).long()

# Step 3: Define the LSTM model
class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(MusicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model, loss function, and optimizer
input_size = len(note_to_int)  # Number of unique notes
hidden_size = 256  # Number of LSTM units in hidden layer
output_size = len(note_to_int)  # Output size is same as number of unique notes

model = MusicLSTM(input_size, hidden_size, output_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

# Step 5: Generate Music
def generate_music(model, int_to_note, sequence_length=100, seed_notes=None, num_notes=500):
    model.eval()
    
    if seed_notes is None:
        seed_notes = np.random.choice(list(int_to_note.values()), sequence_length)
    
    generated_notes = []
    input_sequence = torch.Tensor([seed_notes]).long().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for _ in range(num_notes):
        with torch.no_grad():
            prediction = model(input_sequence)
            predicted_note = torch.argmax(prediction).item()
            generated_notes.append(predicted_note)
            input_sequence = torch.cat((input_sequence[:, 1:], torch.tensor([[predicted_note]]).long().to(input_sequence.device)), dim=1)

    return generated_notes

# Generate music
generated_notes = generate_music(model, int_to_note, num_notes=500)

# Convert generated notes to actual music
def notes_to_stream(generated_notes, int_to_note):
    output_notes = []
    for note_int in generated_notes:
        note_str = int_to_note[note_int]
        if '.' in note_str or note_str.isdigit():
            chord_notes = note_str.split('.')
            chord_notes = [note.Note(int(n)) for n in chord_notes]
            output_notes.append(chord.Chord(chord_notes))
        else:
            output_notes.append(note.Note(note_str))
    stream = stream.Stream(output_notes)
    return stream

# Convert generated notes to music stream and save it as MIDI
output_stream = notes_to_stream(generated_notes, int_to_note)
output_stream.write('midi', fp='generated_music.mid')

print("Generated music saved to 'generated_music.mid'")
