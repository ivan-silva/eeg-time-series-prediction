# Analisi di eeg come serie temporali

Lo scopo di questo progetto è analizzare tracciati elettroencefalografici (eeg) ed predirne il comportamento tramite 
reti neurali.

Le procedure vengono descritte nel dettaglio nei seguenti files:

- [Procedura completa](https://colab.research.google.com/github/ivan-silva/eeg-time-series-prediction/blob/master/docs/procedure.ipynb)

## Testbench

Qui vengono raccolte le applicazioni delle tecniche descritte nelle procedure:

  - LSTM con lookback
    1. [Canale Alfa1 (valori medi) su multipli soggetti](testbench_1.ipynb)
    2. [Canali multipli (raw data) su singolo soggetto](testbench_2.ipynb)