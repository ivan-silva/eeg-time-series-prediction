# Analisi di eeg come serie temporali

Lo scopo di questo progetto è analizzare tracciati elettroencefalografici (eeg) e predirne il comportamento tramite 
reti neurali.

Le procedure vengono descritte nel dettaglio nei seguenti files:

- [Procedura completa](docs/procedure.ipynb)

## Testbench

Qui vengono raccolte le applicazioni delle tecniche descritte nelle procedure:

  - LSTM con lookback
    1. [Canale Alfa1 (valori medi) su multipli soggetti](testbench_1.ipynb)
    2. [Canali multipli (raw data) su singolo soggetto](testbench_2.ipynb)
