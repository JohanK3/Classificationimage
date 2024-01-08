# Classificationimage

Ce projet implémente une solution de reconnaissance d'émotion en utilisant des images et un modèle de réseau de neurone convolutionnel.

Construire un classificateur d'images en utilisant un réseau de neurone convolutionnel (CNN).

Classifier les images selon deux émotions "happy" et "Sad".

## Prérequis

- Python 3.8 
-  Bibliothèques Python : numpy, matplotlib, tensorflow, cv2

## Réalisation du Travail


-- Entrainement du modèle 

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
```

-- Utilisation


```
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result(), re.result(), acc.result())
```

## Auteur 

- Johan Karl KASSA KASSA

