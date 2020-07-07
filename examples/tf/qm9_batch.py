"""
This example shows how to perform regression of molecular properties with the
QM9 database, using a GNN based on edge-conditioned convolutions in batch mode.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets import qm9
from spektral.layers import EdgeConditionedConv, GlobalSumPool
from spektral.utils import label_to_one_hot

from grnf.tf import GraphRandomNeuralFeatures

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 10           # Number of training epochs
batch_size = 32           # Batch size

################################################################################
# LOAD DATA
################################################################################
A, X, E, y = qm9.load_data(return_type='numpy',
                           nf_keys='atomic_num',
                           ef_keys='type',
                           self_loops=True,
                           amount=1000)  # Set to None to train on whole dataset
y = y[['cv']].values  # Heat capacity at 298.15K

# Preprocessing
X_uniq = np.unique(X)
X_uniq = X_uniq[X_uniq != 0]
E_uniq = np.unique(E)
E_uniq = E_uniq[E_uniq != 0]

X = label_to_one_hot(X, X_uniq)
E = label_to_one_hot(E, E_uniq)

# Parameters
N = X.shape[-2]       # Number of nodes in the graphs
F = X[0].shape[-1]    # Dimension of node features
S = E[0].shape[-1]    # Dimension of edge features
n_out = y.shape[-1]   # Dimension of the target

# Train/test split
A_train, A_test, \
X_train, X_test, \
E_train, E_test, \
y_train, y_test = train_test_split(A, X, E, y, test_size=0.1, random_state=0)

################################################################################
# BUILD MODELS
################################################################################
X_in = Input(shape=(N, F))
A_in = Input(shape=(N, N))
E_in = Input(shape=(N, N, S))

X_1 = EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
X_2 = EdgeConditionedConv(32, activation='relu')([X_1, A_in, E_in])
X_3 = GlobalSumPool()(X_2)
output_ecc = Dense(n_out)(X_3)

# Build model
model_ecc = Model(inputs=[X_in, A_in, E_in], outputs=output_ecc, name="ECC")
optimizer = Adam(lr=learning_rate)
model_ecc.compile(optimizer=optimizer, loss='mse')
model_ecc.summary()

# GRNF
psi = GraphRandomNeuralFeatures(256, activation="relu")([X_in, A_in, E_in])
d1 = Dense(16, activation="relu")(psi)
output_grnf = Dense(n_out)(d1)

# Build model
model_grnf = Model(inputs=[X_in, A_in, E_in], outputs=output_grnf, name="GRNF")
model_grnf.compile(optimizer=optimizer, loss='mse')
model_grnf.summary()

################################################################################
# FIT MODELS
################################################################################
print("Training model ECC")
model_ecc.fit([X_train, A_train, E_train],
              y_train,
              batch_size=batch_size,
              epochs=epochs)

print("Training model GRNF")
model_grnf.fit([X_train, A_train, E_train],
               y_train,
               batch_size=batch_size,
               epochs=epochs)


################################################################################
# EVALUATE MODELS
################################################################################
print('Testing model ECC')
model_loss_ecc = model_ecc.evaluate([X_test, A_test, E_test],
                                    y_test,
                                    batch_size=batch_size)
print('Done. Test loss: {}'.format(model_loss_ecc))

print('Testing model GRNF')
model_loss_grnf = model_grnf.evaluate([X_test, A_test, E_test],
                                      y_test,
                                      batch_size=batch_size)
print('Done. Test loss: {}'.format(model_loss_grnf))
