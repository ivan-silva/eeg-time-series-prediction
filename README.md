# Analisi di eeg come serie temporali

Lo scopo di questo progetto è analizzare tracciati elettroencefalografici (eeg) ed predirne il comportamento tramite reti neurali.

Le procedure vengono descritte di seguito

Analisi e preparazione dei dati

1. [Analisi e preparazione dei dati medi](https://colab.research.google.com/github/ivan-silva/eeg-time-series-prediction/blob/master/docs/average_data_analysis.ipynb)
2. [Analisi e preparazione dei dati grezzi](https://colab.research.google.com/github/ivan-silva/eeg-time-series-prediction/blob/master/docs/raw_data_analysis.ipynb)

Categorizzazione e risoluzione di problemi

1. [Data una sequenza monodimensionale predirne il continuo](https://colab.research.google.com/github/ivan-silva/eeg-time-series-prediction/blob/master/docs/problem_monodimensional_data_serie_lookback.ipynb)
2. [Data una sequenza multidimensionale predirne il continuo](https://colab.research.google.com/github/ivan-silva/eeg-time-series-prediction/blob/master/docs/problem_multidimensional_data_serie_lookback.ipynb)

## Testbench

Qui vengono raccolte le applicazioni delle tecniche descritte nelle procedure:

  - LSTM con lookback
    1. [Canale Alfa1 (valori medi) su multipli soggetti](https://colab.research.google.com/github/ivan-silva/eeg-time-series-prediction/blob/master/testbench_1.ipynb)
    2. [Canali multipli (raw data) su singolo soggetto](https://colab.research.google.com/github/ivan-silva/eeg-time-series-prediction/blob/master/testbench_2.ipynb)
    

## Diario

### 22/02/21

Il problema mono e bidimensionale con lookback è stato indagato a fondo. È stato creato un modulo riutilizzabile e applicato su un soggetto (più parametri), su più soggetti (singolo parametro), sia su dati medi che su dati grezzi.

Il prossimi passi saranno:
-  Scegliere una proprieta' dei test fatti prima (P1) e dopo il ciclo di sessioni (P2)
e fare un training con
   
   (P1, alfa1, beta1, ..., alfa2, beta2, ....) -> P2 
   
   dove XXX1 sono i dati eeg della prima sessione e XXX2 i dati eeg della seconda sessione. In pratica non guardiamo tutta l'evoluzione nel tempo, ma solo l'inizio e la fine. 
   Pensare di fare la media delle prime 5 sessioni e ultime 5 sessioni,
per ogni canale eeg. 
   La domanda a cui vogliamo rispondere e': data la situazione iniziale
e l'andamento dell'eeg, riusciamo a prevedere il valore finale.
- Usare piu' pazienti (es training su n-1 e validation su 1)


### 9/02/21

Alcune parti del processo non mi sono ancora del tutto chiare ma la cosa che più mi preoccupa sono alcune predizioni eccessivamente corrette. Temo possa essere stato impostato male il problema.

Prossimi passi:

- Mischiare dati di soggetti differenti
- Migliorare la consapevolezza sull’esperimento
- Automatizzare per ogni canale in ingresso
  - La produzione dei modelli e delle predizioni
  - La produzione dei plot
- Creare multiplot
