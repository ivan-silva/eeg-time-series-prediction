# Analisi di eeg come serie temporali

## Dati

I dati sono stati registrati durante un ciclo di terapia effettuato durante 78 sedute, su 7 soggetti. Per ognuna delle quali sono stati registrati i valori medi relativi a 10 parametri:

0. Alfa1
1. Alfa2
2. Beta1
3. Beta2
4. Delta
5. Gamma1
6. Gamma2
7. Theta
8. Meditazione (*calcolato*)
9. Attenzione (*calcolato*)

Questi sono stati estratti dall’onda eeg originale, isolando range di frequenze specifiche, tramite trasformata di Fourier.

> **Nota.** I valori calcolati non dovrebbero aggiungere informazioni interessanti e quindi in un primo momento non li consideriamo.
> **Nota.** In questo momento i dati sono relativi unicamente a un paziente

### Plot

Per fare i primi ragionamenti sui dati e cominciare a "capirli", si è scelto casualmente un paziente e si è fatto un grafico dei valori.

![parameters_plot](images/parameters_plot.png)

Dato che notiamo degli spike anomali proviamo a farne un plot sovrapposto

![raw_parameters](images/raw_parameters.png)

Vediamo che gli spike sono diffusi sui vari canali, in questo momento può venire in mente siano causati da giornate particolari, errori di taratura degli strumenti ecc. va indagato ed è necessario capire se escluderli o attenuarli.

## Matrice di correlazione

Successivamente si è fatta una matrice di correlazione per iniziare a indagare i rapporti tra i valori e valutare se escludere parametri tra loro dipendenti.

![feature_correlation_heatmap](images/feature_correlation_heatmap.png)

## Scelta e preparazione dei parametri

Per queste prime prove è stato deciso di rimuovere i parametri calcolati, *attenzione* e *meditazione*, rimangono quindi

0. Alfa1
1. Alfa2
2. Beta1
3. Beta2
4. Delta
5. Gamma1
6. Gamma2
7. Theta

I parametri sono stati poi stati portati all’interno di un intervallo <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/acf5ce819219b95070be2dbeb8a671e9.svg?invert_in_darkmode" align=middle width=32.87674500000001pt height=24.65759999999998pt/> tramite `MinMaxScaler((-1, 1))`.

![normalized_parameters](images/normalized_parameters.png)

Si è provato anche ad applicare uno *Standard score* ma per ora sembra peggiorare i valori di training, verranno fatte ulteriori prove durante il fine tuning.

## Model

Di seguito `input_shape`, `target_shape` e model summary

```
Input shape: (1, 1, 8)
Target shape: (1, 1)

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 1, 8)]            0
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5248
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,281
Trainable params: 5,281
Non-trainable params: 0
_________________________________________________________________
```

## Training

I dati di training sono stati espansi tramite [`timeseries_dataset_from_array`](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/timeseries_dataset_from_array), utilizzando i seguenti parametri

```
sequence_length = 1
sampling_rate = 1
sequence_stride = 1
```

Sono perplesso sull’utilità di utilizzare una sliding window con questi parametri, mi aspetto il risultato finale sia equivalente alla sequenza iniziale. Qualsiasi altro parametro peggiorava notevolmente i risultati.

I parametri del training invece sono

```
 learning_rate = 0.001
 batch_size = 1
 epochs = 100
```



## Categorizzazione e risoluzione di problemi

### Data una sequenza monodimensionale predirne il continuo

#### Dati

Array colonna di dati float
<p align="center"><img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/38eb638562e89df43865efc029e1b4db.svg?invert_in_darkmode" align=middle width=428.96534999999994pt height=118.102875pt/></p>

dove <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/16543fa87f68ed614d2a31ead6a8e985.svg?invert_in_darkmode" align=middle width=98.64937499999999pt height=24.65759999999998pt/>.

#### Model e train

Prendiamo i dati e scegliamo un indice <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663295000000005pt height=21.683310000000006pt/> che divida tra quelli che useremo per l’allenamento e quelli invece che useremo per la validazione
<p align="center"><img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/0d8f6f79f9d5d0c426ad432f328c9bb8.svg?invert_in_darkmode" align=middle width=203.14469999999997pt height=41.947125pt/></p>
Sia per i dati di allenamento che per i dati di validazione, generiamo le variabili <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/f3fce9efeb06bb9860c2ae7824f1b5ea.svg?invert_in_darkmode" align=middle width=71.1678pt height=24.65759999999998pt/> e il <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/9100cff2b7c6723c7fe41d214a78b617.svg?invert_in_darkmode" align=middle width=75.79374pt height=24.65759999999998pt/>, dove <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.187330000000003pt height=22.46574pt/> è la variabile di *look back* che ci dice quanti elementi vengono tenuti in considerazione per indovinare l’elemento <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/a87c614260ec1f63d9fad08c6c6896eb.svg?invert_in_darkmode" align=middle width=34.351515pt height=14.155350000000013pt/>-esimo
<p align="center"><img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/53af44ee1b73d92ab49902230f89224e.svg?invert_in_darkmode" align=middle width=525.26595pt height=198.3399pt/></p>
in forma matriciale
<p align="center"><img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/1711f86741b6e59c47a875a6e5197708.svg?invert_in_darkmode" align=middle width=454.15425pt height=257.51714999999996pt/></p>

Esempio per <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/a34bf513818fb6b7feb806c09a7be0ec.svg?invert_in_darkmode" align=middle width=235.814205pt height=24.65759999999998pt/>, <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/e0fa8739d51886edf01d4354e14daca2.svg?invert_in_darkmode" align=middle width=61.00842pt height=21.18732pt/>, <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/365c83bbd9bde4ad1973bd65896b2410.svg?invert_in_darkmode" align=middle width=49.543395pt height=22.46574pt/>
<p align="center"><img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/c94bd6ade074478cbab06617a7dc4b70.svg?invert_in_darkmode" align=middle width=406.2564pt height=257.51714999999996pt/></p>

Prendiamo un valore di split <img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/998349a36ffd1b307d7e3fd4c956cda5.svg?invert_in_darkmode" align=middle width=61.130025pt height=21.18732pt/> abbiamo che le predizioni saranno
<p align="center"><img src="https://rawgit.com/ivan-silva/eeg-time-series-prediction/None/svgs/17bfcf58a49e0393bd39f755fdaefff7.svg?invert_in_darkmode" align=middle width=481.27694999999994pt height=65.75349pt/></p>

## Data una sequenza bidimensionale predirne il continuo

TODO


    Samples. One sequence is one sample. A batch is comprised of one or more samples.
    Time Steps. One time step is one point of observation in the sample.
    Features. One feature is one observation at a time step.
Il modello utilizzato è il seguente


```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 4)                 176
_________________________________________________________________
dense (Dense)                (None, 6)                 30
=================================================================
Total params: 206
Trainable params: 206
Non-trainable params: 0
_________________________________________________________________
```

## Sperimentazione per un parametro su più soggetti

![Previsioni](images/eeg_multiple_subjects_predictions_1_100.png)

![RMSE](images/eeg_multiple_subjects_RMSE_1_100.png)

## Conclusioni

### 9/02/21

Alcune parti del processo non mi sono ancora del tutto chiare ma la cosa che più mi preoccupa sono alcune predizioni eccessivamente corrette. Temo possa essere stato impostato male il problema.

Prossimi passi:

- Mischiare dati di soggetti differenti
- Migliorare la consapevolezza sull’esperimento
- Automatizzare per ogni canale in ingresso
  - La produzione dei modelli e delle predizioni
  - La produzione dei plot
- Creare multiplot