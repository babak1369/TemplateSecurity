import numpy as np
def group_degarbling_( encrypted_table, given_keys,circuit,security):
    batch_size = given_keys.shape[0]
    for i in range( circuit.shape[0]):
        key1 = given_keys[:, circuit[i, 0]]
        key2 = given_keys[:,  circuit[i, 1]]
        row = 10 * key1[:,  security] + key2[:,  security]
        n_rows1 = ((encrypted_table[np.arange(batch_size),i,row, 0,  security] ) + key1[:,  security] + key2[:,  security]) % 10
        n_rows2 = ((encrypted_table[np.arange(batch_size),i ,row, 1,  security]) + key1[:,  security] + key2[:,  security]) % 10
        k_out1 = ((encrypted_table[np.arange(batch_size),i,row, 0, 0: security]+ key1[:, 0: security] + key2[:,
                                                                                            0: security]) % 2)
        k_out2 = ((encrypted_table[np.arange(batch_size),i,row, 1, 0: security] + key1[:, 0: security] + key2[:,
                                                                                            0: security]) % 2)
        k_out1 = np.hstack((k_out1, n_rows1.reshape((batch_size, 1))))
        k_out2 = np.hstack((k_out2, n_rows2.reshape((batch_size, 1))))
        given_keys[:,  circuit[i, 2]] = k_out1
        given_keys[:,  circuit[i, 3]] = k_out2
    return given_keys

def degarbling_(encrypted_table, given_keys,circuit,security):
    for i in range( circuit.shape[0]):
        key1 = given_keys[ circuit[i, 0]]
        key2 = given_keys[ circuit[i, 1]]
        row = int(10 * key1[ security] + key2[ security])
        n_rows1 = (encrypted_table[i][row, 0,  security] + key1[ security] + key2[ security]) % 10
        n_rows2 = (encrypted_table[i][row, 1,  security] + key1[ security] + key2[ security]) % 10
        k_out1 = ((encrypted_table[i][row, 0, 0: security] + key1[0: security] + key2[0: security]) % 2)
        k_out2 = ((encrypted_table[i][row, 1, 0: security] + key1[0: security] + key2[0: security]) % 2)
        k_out1 = np.append(k_out1, n_rows1)
        k_out2 = np.append(k_out2, n_rows2)
        given_keys[ circuit[i, 2]] = k_out1
        given_keys[ circuit[i, 3]] = k_out2
    return given_keys