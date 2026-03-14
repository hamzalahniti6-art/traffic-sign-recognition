Verkehrszeichenerkennung mit Deep Learning

Dieses Projekt implementiert ein Convolutional Neural Network (CNN) zur Klassifikation von Verkehrszeichen unter Verwendung des Datensatzes German Traffic Sign Recognition Benchmark (GTSRB).

Voraussetzungen

Python 3.12

Benötigte Bibliotheken:

tensorflow

numpy

pandas

opencv-python

matplotlib

scikit-learn

Installieren Sie die benötigten Bibliotheken mit folgendem Befehl:

pip install tensorflow numpy pandas opencv-python matplotlib scikit-learn

Datensatz

Der in diesem Projekt verwendete Datensatz kann unter folgendem Link heruntergeladen werden:

https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed

Nach dem Herunterladen sollte der Datensatz in den folgenden Ordner entpackt werden:

traffic_sign_project/dataset

Ausführen des Modells

Führen Sie folgenden Befehl aus:

python train_model.py

Das Skript lädt den Datensatz, trainiert das CNN-Modell und bewertet anschließend dessen Genauigkeit.
