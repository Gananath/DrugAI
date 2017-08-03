
'''
Author: Gananath R
DrugAI-WGAN: Some more experiments with GAN for drug like molecule generation
Contact: https://github.com/gananath

# shamelessly copied codes for wgan with some modifications from
# https://github.com/eriklindernoren/Keras-GAN/tree/master/wgan
# https://github.com/farizrahman4u/keras-contrib/blob/master/examples/improved_wgan.py

#trained for one hour in GPU using floydhub.com
'''

from drugai import *
# import os.path
np.random.seed(2017)


def Gan(PATH="GAN.h5", lr=0.00001):
    GAN = Sequential()
    GAN.add(G)
    D.trainable = False
    GAN.add(D)
    checkpoint_GAN = ModelCheckpoint(
        filepath=PATH, verbose=1, save_best_only=True)
    GAN.compile(
        loss=wasserstein_loss,
        optimizer=Adam(
            lr,
            beta_1=0.5,
            beta_2=0.9))
    return GAN, checkpoint_GAN


# read csv file
data = pd.read_csv('stahl.csv')
data = data.reindex(np.random.permutation(data.index))
# data=data.head(30)
Y = data.SMILES
Y.head()
X = data.ix[:, 1:7]
X = X.values
X = X.astype('int')
type(X)

# padding smiles to same length by adding "|" at the end of smiles
maxY = Y.str.len().max() + 11
y = Y.str.ljust(maxY, fillchar='|')
ts = y.str.len().max()
print ("ts={0}".format(ts))
# CharToIndex and IndexToChar functions
chars = sorted(list(set("".join(y.values.flatten()))))
print('total chars:', len(chars))

char_idx = dict((c, i) for i, c in enumerate(chars))
idx_char = dict((i, c) for i, c in enumerate(chars))

y_dash = dimY(y, ts, char_idx, chars)
x_dash = X
print("Shape\n X={0} Y={1}".format(x_dash.shape, y_dash.shape))


BATCH_SIZE = 32
HALF_BATCH = BATCH_SIZE / 2
CLIP = 0.01
epochs = 100000
n_critic = 5

G, checkG = Generator(x_dash, y_dash)
D, checkD = Discriminator(y_dash)
GAN, checkGAN = Gan()
# enable training in discrimator
D.trainable = True

for epoch in range(epochs):
    for _ in range(n_critic):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of images
        idx = np.random.randint(0, y_dash.shape[0], HALF_BATCH)
        imgs = y_dash[idx]

        # noise = np.random.normal(0, 1, (HALF_BATCH, 100))
        noise = x_dash[0:HALF_BATCH]
        # Generate a half batch of new images
        gen_imgs = G.predict(noise)

        # Train the discriminator
        d_loss_real = D.train_on_batch(
            imgs, -np.ones((HALF_BATCH, 1)))  # linear activation
        d_loss_fake = D.train_on_batch(gen_imgs, np.ones((HALF_BATCH, 1)))
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

        # Clip discriminator weights
        for l in D.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -CLIP, CLIP) for w in weights]
            l.set_weights(weights)

        # ---------------------
        #  Train Generator
        # ---------------------

        # noise = np.random.normal(0, 1, (BATCH_SIZE, 100))
        noise = x_dash[0:BATCH_SIZE]
        # Train the generator
        g_loss = GAN.train_on_batch(
            noise, -np.ones((BATCH_SIZE, 1)))  # linear activation

    print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss))
    if epoch % (epochs / 1000) == 0:

        # 
        # for saving files in floydhub output directory
        G.save("/output/Gen.h5")
        D.save("/output/Dis.h5")
        GAN.save("/output/Gan.h5")
        # G.save(os.getcwd()+'/output/Gen.h5')
        # D.save(os.getcwd()+'/output/Dis.h5')
        # GAN.save(os.getcwd()+'/output/Gan.h5')

        # For Prediction
        # start Prediction

        Ghash, checkG = Generator(x_dash, y_dash)
        print("Prediction")
        Ghash.load_weights('/output/Gen.h5')
        x_pred = [[0, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1]]

        preds = Ghash.predict(x_pred)
        y_pred = prediction(preds)
        y_pred = seq_txt(y_pred, idx_char)
        s = smiles_output(y_pred)
        print(s)
        # end prediction'''
