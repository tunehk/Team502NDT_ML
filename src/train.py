import keras
from keras import backend as K
from keras import Input, layers
from keras import Model

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from os import listdir
from os.path import isfile, join
import uuid
import time

w, h = 256, 256  # initial data size
window = 7  # window for the first max-pool operation

run_uuid = uuid.uuid4()  # unique identifier is generated for each run

path = "../data/training/"  # training data path
vpath = "../data/validation/"  # validation data path

'''     The data_generator reads raw binary UT data from the pre-processed files
        and preconditions it for ML training. '''


def data_generator(batch_size=10):
    input_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.bins')]
    np.random.shuffle(input_files)  # we'll take random set from available data files
    input_files = input_files[0:100]  # limit to 100 files per epoch
    xs = np.empty((0), dtype='float32')  # input data
    ys = np.empty((0, 2), dtype='float32')  # label data

    # Track successful and failed files
    loaded_count = 0
    skipped_count = 0

    for i in input_files:
        try:
            # Load binary data
            bxs = np.fromfile(path + i, dtype=np.uint16).astype('float32')

            # Check if file has data
            if bxs.size == 0:
                print(f"⚠️ Skipping {i}: empty file")
                skipped_count += 1
                continue

            # Normalize
            bxs -= bxs.mean()
            bxs /= bxs.std() + 0.00001  # avoid division by zero

            # Load labels
            bys = np.loadtxt(path + i[:-5] + '.labels')

            # Ensure labels are 2D
            if len(bys.shape) == 1:
                bys = bys.reshape(-1, 2)

            # Test reshape on this file's data before concatenating
            test_shape = np.reshape(bxs, (-1, 256, 256, 1), 'C')

            # Validate dimensions
            if test_shape.shape[1] != 256 or test_shape.shape[2] != 256 or test_shape.shape[3] != 1:
                print(f"⚠️ Skipping {i}: invalid shape {test_shape.shape}")
                skipped_count += 1
                continue

            # Check if labels match data
            if bys.shape[0] != test_shape.shape[0]:
                print(f"⚠️ Skipping {i}: label count {bys.shape[0]} doesn't match data count {test_shape.shape[0]}")
                skipped_count += 1
                continue

            # All checks passed - concatenate
            xs = np.concatenate((xs, bxs))
            ys = np.concatenate((ys, bys))
            loaded_count += 1

        except Exception as e:
            print(f"❌ Error loading {i}: {e}")
            skipped_count += 1
            continue

    # Summary
    print(f"✅ Loaded {loaded_count} files successfully")
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} problematic files")

    # Check if we have any data at all
    if xs.size == 0:
        raise RuntimeError("No valid training data loaded! All files were skipped.")

    # Reshape all concatenated data
    xs = np.reshape(xs, (-1, 256, 256, 1), 'C')

    rows = xs.shape[0]
    cursor = 0

    while True:
        # FIXED: Check if we need to reset BEFORE calculating slice
        if cursor + batch_size > rows:
            cursor = 0

        start = cursor
        end = cursor + batch_size
        cursor = end

        # Ensure we don't go past the end
        if end > rows:
            end = rows

        # Yield the batch
        yield (xs[start:end, :, :, :], ys[start:end, 0])


input_tensor = Input(shape=(w, h, 1))

# start with max-pool to envelop the UT-data
ib = layers.MaxPooling2D(pool_size=(window, 1), padding='valid')(
    input_tensor)  # MaxPooling1D would work, but we may want to pool adjacent A-scans in the future

# build the network
cb = layers.Conv2D(96, 3, padding='same', activation='relu')(ib)
cb = layers.Conv2D(64, 3, padding='same', activation='relu')(cb)
cb = layers.MaxPooling2D((2, 8), padding='same')(cb)

cb = layers.Conv2D(48, 3, padding='same', activation='relu')(cb)
cb = layers.Conv2D(32, 3, padding='same', activation='relu')(cb)
cb = layers.MaxPooling2D((3, 4), padding='same')(cb)
cb = layers.Flatten()(cb)
cb = layers.Dense(14, activation='relu', name='RNN')(cb)
iscrack = layers.Dense(1, activation='sigmoid', name='output')(cb)

model = Model(input_tensor, iscrack)
opt = keras.optimizers.RMSprop(learning_rate=0.0001, clipnorm=1.)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
model.summary()

test_uuid = "FA4DC2D8-C0D9-4ECB-A319-70F156E3AF31"
rxs = np.fromfile(vpath+test_uuid+".bins", dtype=np.uint16 ).astype('float32')
rxs -= rxs.mean()
rxs /= rxs.std()+0.0001
rxs = np.reshape( rxs, (-1,256,256,1), 'C')
rys = np.loadtxt(vpath+test_uuid+".labels", dtype=np.float32)

validation_uuid = "FA4DC2D8-C0D9-4ECB-A319-70F156E3AF31"
xs = np.fromfile(vpath+validation_uuid+".bins", dtype=np.uint16 ).astype('float32')
xs -= xs.mean()
xs /= xs.std()+0.0001
xs = np.reshape( xs, (-1,256,256,1), 'C')
ys = np.loadtxt(vpath+validation_uuid+".labels", dtype=np.float32)



class DebugCallback(keras.callbacks.Callback):
    #    def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs={}):
        predictions = model.predict(rxs)
        res = np.concatenate((rys, predictions), -1)

        # Create plotly scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res[:, 1],
            y=res[:, 2],
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Predictions'
        ))
        fig.update_layout(
            title=f'Epoch {epoch} - Predictions vs Actual',
            xaxis_title='Actual Crack Size',
            yaxis_title='Predicted Probability',
            showlegend=True
        )
        fig.show()

        # Optionally save the figure
        # fig.write_html(f"epoch_{epoch}_plot.html")


debug = DebugCallback()

callbacks = [keras.callbacks.TensorBoard(log_dir='log', histogram_freq=1)
    , keras.callbacks.ModelCheckpoint('../models/model-' + str(run_uuid) + '.keras', monitor='val_loss', save_best_only=True)
    , debug]

model.fit(data_generator(100), epochs=100, validation_data=(xs, ys[:, 0]), steps_per_epoch=60,
                    callbacks=callbacks)

# Final predictions plot
predictions = model.predict(rxs)
res = np.concatenate((rys, predictions), -1)

# Create final plotly scatter plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=res[:, 1],
    y=res[:, 2],
    mode='markers',
    marker=dict(color='blue', size=8),
    name='Final Predictions'
))
fig.update_layout(
    title='Final Model Performance - Predictions vs Actual',
    xaxis_title='Actual Crack Size (mm)',
    yaxis_title='Predicted Probability',
    showlegend=True,
    template='plotly_white'
)
fig.show()

# Save the plot as interactive HTML
fig.write_html('results_plot.html')

print(res)
np.savetxt('results.txt', res)