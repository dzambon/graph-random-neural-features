"""
This example shows how to perform graph classification with a synthetic dataset
of Delaunay triangulations, using a graph attention network (Velickovic et al.)
in batch mode.
"""

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import delaunay
from spektral.layers import GraphAttention, GlobalAttentionPool

from grnf.tf import GraphRandomNeuralFeatures

# Load data
A, X, y = delaunay.generate_data(return_type='numpy', classes=[0, 5])

# Parameters
N = X.shape[-2]          # Number of nodes in the graphs
F = X.shape[-1]          # Original feature dimensionality
n_classes = y.shape[-1]  # Number of classes
l2_reg = 5e-4            # Regularization rate for l2
learning_rate = 1e-3     # Learning rate for Adam
epochs = 20000           # Number of training epochs
batch_size = 32          # Batch size
es_patience = 200        # Patience fot early stopping

# Train/test split
A_train, A_test, \
x_train, x_test, \
y_train, y_test = train_test_split(A, X, y, test_size=0.1)

# Model definition
X_in = Input(shape=(N, F))
A_in = Input((N, N))

gc1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, A_in])
gc2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1, A_in])
pool = GlobalAttentionPool(128)(gc2)

output_gat = Dense(n_classes, activation='softmax')(pool)

# Build model
model_gat = Model(inputs=[X_in, A_in], outputs=output_gat, name="GAT")
optimizer = Adam(lr=learning_rate)
model_gat.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model_gat.summary()

# GRNF
psi = GraphRandomNeuralFeatures(256, activation="relu")([X_in, A_in])
d1 = Dense(16, activation="relu")(psi)
output_grnf = Dense(n_classes, activation='softmax')(d1)

# Build model
model_grnf = Model(inputs=[X_in, A_in], outputs=output_grnf, name="GRNF")
model_grnf.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model_grnf.summary()

# Train model
model_gat.fit([x_train, A_train],
              y_train,
              batch_size=batch_size,
              validation_split=0.1,
              epochs=epochs,
              callbacks=[
                  EarlyStopping(patience=es_patience, restore_best_weights=True)
              ])

# Train model
model_grnf.fit([x_train, A_train],
               y_train,
               batch_size=batch_size,
               validation_split=0.1,
               epochs=epochs,
               callbacks=[
                   EarlyStopping(patience=es_patience, restore_best_weights=True)
               ])

# Evaluate model
print('Evaluating model GAT.')
eval_results = model_gat.evaluate([x_test, A_test],
                              y_test,
                              batch_size=batch_size)
print('Done. Test loss: {:.4f}. Test acc: {:.2f}'.format(*eval_results))

# Evaluate model
print('Evaluating model GRNF.')
eval_results = model_grnf.evaluate([x_test, A_test],
                              y_test,
                              batch_size=batch_size)
print('Done. Test loss: {:.4f}. Test acc: {:.2f}'.format(*eval_results))
