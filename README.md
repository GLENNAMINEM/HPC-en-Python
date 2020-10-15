##### install Dependencies from requirements
```
  pip install -r requirementsUtilisation.txt

```

## utilisation

Ce notebook presente l'utilisation l'utilisation de certaines librairies parllèles Python pour le calcul parallèle.
Un exemple applicatif du traitement d'une grande grille est utilisé, suivi des tests et comparaisons des performances.

Dans ce notebooke, l'utilisation de mpi4py nécessite  la configuration d'un cluster de calcul.
Ceci est effectué par la librairie IPyparallel (interactive Python parallel).

Pour la configuration du cluster ipy:
```
0:assurez vous d'avoir MPI installé sur votre machine
```
```
1:$ ipython profile create --parallel --profile=mpi

```
```
2:$ vim  ~/.ipython/profile_mpi/ipcluster_config.py 
  (on ajoute la ligne: c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher')
  ```
```
3:$ vim  ~/.ipython/profile_mpi/ipengine_config.py 
  (on ajoute la ligne: c.MPI.use = 'mpi4py')
```
```
4:On démarre le cluster dans le jupyter notebook à l'angle ' IPython Clusters '
```
