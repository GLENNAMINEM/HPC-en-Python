# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# #                 Plan
# <ol>
#     
# ## 0 Introduction générale
# ## 1 Utilisation de mpi4py
# ## 2 Utilisation de multiprocessing
# ## 3 Utilisation de multithreading
# ## 4 Utilisation de dask
# ## 5 Utilisation de Cython
# ## 6 Profilage
# ## 8 Optimisation
# ## 9 Performance
#
# </ol>

# %% [markdown]
# # 0 Introduction générale

# %% [markdown]
#
#
#
#
#
#
#
#
#
#
#
#

# %% [markdown]
# Ce notebook présente un exemple d'utilisation de certaines librairies python pour le calcul parallèle et les differents diagnostics.
# Dans l'ensemble de ce tuto, toutes les librairies exécututent un même exemple, recherche des chiffres nons nuls dans un tableau donné(rempli aléatoirement).
#
#
# ##### Définitions pour comprendre au mieux ce tuto,
#
# * Python est un langage interpreté, c'est à dire que la traduction du code en langage machine se fait par un interpreteur, l'exécution du code se fait de manière dynamique au fur et à mesure que les instructions sont interprées.
#
# * Un processus est une instance d'un programme, c'est un programme en cours d'exécution.C'est un ensemble d'instructions à exécuter. Un processus contient une mémoire.</li>
#
#
# * Un thread est l'unité d'exécution d'un processus.Un processus contient plusieurs threads.Un thread vit au sein d'un processus. Tout les threads d'un processus ont accès à la même mémoire
# </li>
#
#
# * Un noeud de calcul est un ensemble processeurs (matériel physique qui permet d'exécuter un programme) et un système d'exploitation, exemple un pc.
# </li>
#
# * Un Clusters un ensemble de noeuds de caluls organisés en réseau,exemple d'un supercalculateur.
# </li>
#
#
# * On parle de parallélisme des données, quand les données sont partagées entre processus,à l'opposé du parallélisme de tâches.  
#

# %% [markdown]
# # 1 Utilisation de mpi4py

# %% [markdown]
# * MPI4PY est la version python de la librairie mpi, elle permet la gestion des processus(communication entre processus, indexation des processus...).
# * Dans mpi, COMM_WORLD est le communicateur qui contient l'ensemble des processus, c'est le plus grand communicateur. Mpi permet la création des communicateurs plus pétits. Un communicateur un ensemble de processus pouvant communiquer entre eux.</li>
#
#
# * Dans mpi, on distingue les communications point à point (entre procesus 'd'un même communicateur') et les communications collectives ( qui font intervenir tout les processus d'un communicateur).Elles sont dites bloquantes:un processus ne participe pas à la communication, il y a deadlock</li>
#
# * Dans mpi, si une instruction ou donnée n'est pas précédée d'une condition, alors elle est detenue par tout les processus. </li>
#
# * Pour l'envoie des données mpi4py ne prend pas en charge la parallélisation des objets supérieurs à 2^31 octets.La librairie bigmpi4py peut être utiliée,c'est une librairie qui optimise les échanges de gros volumes données(jusqu'à la limite de RAM de l'ordinateur).
# * <<BigMPI4py détermine automatiquement, en tenant compte du type de données, la stratégie optimale de division d'objet pour la parallélisation et utilise des méthodes vectorisées pour les tableaux de types numériques, ce qui permet une plus grande efficacité de parallélisation...>>_[source](https://www.biorxiv.org/content/10.1101/517441v1)_  </li>
#
#
# * L'utilisation de mpi4py sur notebook jupyter nécessite l'utilisation de l'interactive python (ipy) qui est permet de gérer des clusters. _[plus d'informations](https://ipyparallel.readthedocs.io/en/latest)_  </li>
#
#
#

# %% [markdown]
# * ###  connexion au cluster ipy

# %%
import ipyparallel as ipp    
c = ipp.Client(profile='MPI')   #connexion au cluster 'MPI' à changer !!!!
#c.block = True
c[:]

# %% [markdown]
# L'instruction `%%px`  est une commande magic, elle demande au cluster d'exécuter toute la cellule.A l'appel de cette commande, le contrôleur et les clients l'exécutent.On peut les distinguer par la tailles du nombre de processus qu'ils contiennent.
#

# %% [markdown]
# * ### Résolution avec la parallélisation des données

# %% [markdown]
# Le processus 0 remplit un tableau aléatoitement et le partage à tout les processus du communicateur.Chaque processus, reçoit les données et appelle la fonction qui dénombre le nombre de choffre non nuls. En fin chaque processus envoie son résultat au processus 0 (qui collecte).
#

# %%
# %%px --local

from mpi4py import MPI
import numpy as np
import bigmpi4py as BM     #importation de BigMpi4py

import time
import resource

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def funcDenombre(Array):
    """ Fonction qui denombre le nombre de chiffres non nuls contenus dans un tableau, implémentation naïve
       * input : 2D Array 
       * output: int count
        """
    count = 0
    for i in range(len(Array)):
        for j in range(len(Array[i])):
            if (Array[i ][ j] < 10) & (Array[i][j] > 0):
                count+=1
    return count    


def funcDenombr(Array):
    """
    Fonction  qui dénombre le nombre de chiffres non nuls dans un tableau en utilisant, une fonction de numpy
    * input : 2D Array 
    * output: int count
    """
    count = 0
    count = np.count_nonzero(Array<10)
    return count    


if size > 1 : #Pour éliminer le contrôleur dont le nombre de processus est égal à 1
 
    if rank == 0:
        start_par = time.time()
        totalcount = 0
        Array = np.random.randint(1,25,(4000,4000))  #Remplissange de Array aléatoirment des nombres de 0 à 25
        ListArray = np.split(Array,size)
    else :
        ListArray=0

    #SubArray = comm.scatter(ListArray,root=0)      #partage de données à tout les processus de communicateur
    SubArray = BM.scatter(ListArray,comm)           #partage de données avec bigmpi
    
    count = funcDenombr(np.asarray(SubArray))       #apple de funcDenombr par tout les processus
    totalcount = 0
    
    totalcount = comm.reduce(count,op = MPI.SUM,root = 0)   #la processus 0 fait la somme des résultats
    
    print ('Mémoire utilisée KB: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )

    
    if rank == 0:
        end_par = time.time()
        print('je suis le processus = {} la somme totale des chiffres est de {} en {}'.format(rank,totalcount,end_par-start_par))


# %% [markdown]
# # 2 Utilisation Multiprocessing

# %% [markdown]
# Multiprocessing est une librairie qui permet de manipuler des processus.Un processus main crée un ou plusierus processus. 
# * Plusieurs manière de créations de processus existent:`spawn` les processus enfants nouvellement créés contiennent les données necessaires à l'exécution de sa tâche(mode par défault), `fork` les processus enfants contiennent tous les mêmes données que ceux du père au démarage...
# * Il est possible créer une zone mémoire tampon où tout les processus ont accès.
# * L'utilisation de Multiprocessing est limité à un noeud de calcul.

# %% [markdown]
# * ### Résolution en utilisant pool

# %% [markdown]
# Pool est une méthode de multiprocessing qui permet de 'manager' les processus.Elle distribue les tâches aux processus de manière optimale.

# %%
import time
import resource

# %%
import multiprocessing
import numpy as np

def funcDenombre(Array):
    """ Fonction qui denombre le nombre de chiffres non nuls contenus dans un tableau, implémentation naïve
       * input : 2D Array 
       * output: int count
        """
    count = 0
    for i in range(len(Array)):
        for j in range(len(Array[i])):
            if (Array[i ][ j] < 10) & (Array[i][j] > 0):
                count+=1
    return count 

def funcDenombr(Array):
    """
    Fonction  qui dénombre le nombre de chiffres non nuls dans un tableau en utilisant, une fonction de numpy
    * input : 2D Array 
    * output: int count
    """
    count = 0
    count = np.count_nonzero(Array<10)
    print ('Mémoire utilisée: (Kb)', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  )
    return count       


# %%
#if __name__ == '__main__':
    
#multiprocessing.set_start_method('fork', force=True)
Sum = 0
#totalproc = multiprocessing.cpu_count()         #cette fonction renvoie le nombre de processus sur le noeud de calcul
totalproc = 2
Array = np.random.randint(1,25,(4000,4000))     #numpy.random.randint(min_val,max_val,(<num_rows>,<num_cols>))
ListArray = np.split(Array,totalproc)
pool = multiprocessing.Pool(totalproc)          #On reserve le nombre processus avec Pool
start = time.time()

resultat = pool.map(funcDenombr,ListArray)        #map() prend en argument la fonction à exécuter par les processus
                                                    #la liste d'arguments de chaque processus à passer à la fonction

end = time.time()

pool.close() 
pool.join() 
    
for i in range(len(resultat)):
    Sum +=resultat[i]
    

print('la somme =',Sum)
print("temps utilisé",(end - start))
print ('Mémoire utilisée KB: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )





# %%
clear

# %% [markdown]
# * ### Résolution en utilisant Queue

# %% [markdown]
# La méthode queue de multiprocessing petmet de créer une file d'attente, où les processus peuvent écrire.Les données sont ordonnées selon le mode first in first out.

# %%
import time
import resource

# %%
from multiprocessing import  Queue,Process
import numpy as np

def funcDenombre(Array,queu):
    """ Fonction qui denombre le nombre de chiffres non nuls contenus dans un tableau, implémentation naïve
       * input : 2D Array 
                 queu file
        """
    count = 0
    for i in range(len(Array)):
        for j in range(len(Array[i])):
            if (Array[i ][ j] < 10) & (Array[i][j] > 0):
                count+=1
    queu.put(count)                   #écriture dans la file
    print ('Mémoire utilisée: (Kb)',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
    
def funcDenombr(Array,queu):
    """ Fonction qui denombre le nombre de chiffres contenu dans un tableau, utilisation de numpy
       * input : 2D Array 
                 queu file
        """
    count = 0
    count = np.count_nonzero(Array<10)
    queu.put(count)                   #écriture dans la file
    print ('Mémoire utilisée: (Kb)',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )



# %%
totalproc = 2
Sum = 0
queu = Queue(totalproc)
threads=[]
#Array = np.random.randint(1,25,(4000,4000))     #numpy.random.randint(min_val,max_val,(<num_rows>,<num_cols>))
ListArray = np.split(Array,totalproc)

start = time.time()
for i in range(totalproc):
    p = Process(target=funcDenombre,args=(ListArray[i],queu))          #On reserve le nombre processus avec Pool
    p.start()
    p.join() 
end = time.time()

for i in range(totalproc):
    Sum += queu.get()             #lecture de la file
print("la somme = ",Sum)
print('temps passé',(end - start))
print ('Mémoire utilisée KB: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )


# %%
clear

# %% [markdown]
# # 3 Utilisation de Multithreading

# %% [markdown]
# Multithreading est une librairie python qui permet l'utilisation des threads.
# Les threads sont crées à partir du thread principal et contiennent la même mémoire.Pour éviter la corruption des données, que les threading écrivent sur les mêmes cases mémoires, l'interpreteur python utilise le global lock interpreter (gil) qui verrouille la mémoire.Par conséquent, les threads sont exécutés un par un.
# Les threads sont éliminés automiquement à la fin du calcul.

# %%
import time
import resource

# %%
import threading,queue

def funcDenombre(Array,queu):
    """ Fonction qui denombre le nombre de chiffres non nuls contenus dans un tableau, implémentation naïve
       * input : 2D Array 
                 queu file
        """
    count = 0
    for i in range(len(Array)):
        for j in range(len(Array[i])):
            if (Array[i ][ j] < 10) & (Array[i][j] > 0):
                count+=1
    queu.put(count)                   #écriture dans la file
    print ('Mémoire utilisée KB: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
  
def funcDenombr(Array,queu):
    """ Fonction qui denombre le nombre de chiffres contenu dans un tableau, utilisation de numpy
       * input : 2D Array 
                 queu file
        """
    count = 0
    count = np.count_nonzero(Array<10)
    queu.put(count)                   #écriture dans la file
    print ('Mémoire utilisée KB: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )


# %%
N = 4
Sum = 0
queu = queue.Queue(N)          #création d'une file de  éléments 
#Array = np.random.randint(0,25,(4000,4000))     #numpy.random.randint(min_val,max_val,(<num_rows>,<num_cols>))
ListArray = np.split(Array,N)

start = time.time()

for i in range(N):
    t=threading.Thread(target=funcDenombre,args=(ListArray[i],queu)) 
    t.start()
    
end = time.time()
for i in range(0,N):
    Sum += queu.get()             #lecture de la file
print("la somme",Sum)
print("le temps",(end - start))
print ('Mémoire utilisée KB: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )


# %%
clear

# %% [markdown]
# # 4 Utilisation de Dask

# %% [markdown]
# Dask est bibliothèque pour le calcul parallèle en python.Dask est composé de deux parties, gestion dynamique et optimisée des tâches  et manipulation parallèle des tableaux.
# * Il est possible de convertir des objets numpy et pandas en objets dask, et d'y effectuer certaines opérations de bases proposées par ces derniers.
# * Dask est utilisé de manière optimale quand les données sont surpieures à la mémoire.
# * Dask manipule les adresses des objets et affiche le resultat à la demande(l'appel de la méthode .compute()).
# * Dask comme numy sont très lentes sur les opérations d'indexation.
# * Il est possible de paralléliser du code python existant avec le decorateur @delayed placer avant la définition de la fonction.

# %% [markdown]
# * ### Résolution en local 

# %%
import dask.array as da
import numpy as np
import dask
import time

def funcDenombre(Array):
    """ Fonction qui denombre le nombre de chiffres non nuls contenus dans un tableau, implémentation naïve
       * input : 2D Array 
       * output: int count
        """
    count = 0
    for i in range(len(Array)):
        for j in range(len(Array[i])):
            if (Array[i ][ j] < 10) & (Array[i][j] > 0):
                count+=1
    return count            

#@dask.delayed()
def funcDenombr(Array):
    """
    Fonction  qui dénombre le nombre de chiffres non nuls dans un tableau en utilisant, une fonction de numpy
    * input : 2D Array 
    * output: int count
    """
    count = 0
    count = np.count_nonzero(Array<10)
    return count    


# %%
Array = np.random.randint(1,25,(4000,4000))     #céation d'un tableau numpy
x = da.from_array(Array, chunks=(2000, 2000))  #conversion et découpage de Array en morçeaux de 2000 x 200.
#x = da.random.randint(1,25,(4000,4000), chunks=(2000, 2000))  #création directement avec dask

# %%
start = time.time()
totaldask = dask.delayed(funcDenombr)(x).compute()
end = time.time()

print("le temps",(end-start))
print("la somme=",totaldask)

# %%
(dask.delayed(funcDenombr)(x)).visualize(rankdir='LR')#(filename='transpose.svg')

# %%
start = time.time()
totalnumpy = funcDenombr(Array)
end = time.time()

print("temps=",end - start)
totalnumpy


# %% [markdown]
# * ### Résolution en se connectant sur le cluster

# %% [markdown]
# Dask permet de se connecter sur cluster et offre un dashboard pour visualiser le statut du calcul. 

# %%
from dask.distributed import Client, progress
client = Client(  ) # Connection au cluster local de la machine, quand il n'y a pas d'argument dans client
client
#progress(resul)


# %%
start = time.time()

data= client.scatter(x)
result = client.submit(funcDenombr,data).result()
#client.gather(result)

end = time.time()

print("temps=",end-start)
result


# %%
clear

# %% [markdown]
# # 5 Utilisation de Cython

# %% [raw]
# Cython est une librairie qui permet d'écrire du c en python.En python le type d'un objet n'est connu que pendant l'intretation, ce qui empêcher l'interpreteur de faire certaines optimisations.
# En utilisant cython "typera" les objets python et ainsi d'approcher la vitesse d'un code en c.
# Cython nous permert aussi d'utiliser les threads(OpenMp) en contournant l'interpreteur(GIL).

# %%
# %load_ext cython


# %%
import Cython.Compiler.Options as CO
CO.extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ]
CO.extra_link_args = ['-fopenmp']

# %%
# %%cython --compile-args=-fopenmp  --link-args=-fopenmp -a
#utilisation de openmp 

cimport numpy as np
import numpy as np
from cython.parallel import prange
import resource
import cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def funcDenombres(int size,long[:,:] Array):
    """   Fonction dénombre le nombre de chiffre dans un tableau de façon parallèle
          *input: Array, tableau 2 D  carré
                  size, taille d'Array sur une dimension
          *output: count, nombre de chiffre dans le tableau
    """
    cdef int count
    cdef int i
    cdef int j
    for i in prange(size, nogil=True, num_threads=12, schedule='dynamic'): 
        """ parcours parallèle de la boucle par des threads, nogil=true désactive le gil python pour
        les threads en parallèles  """
        for j in range(size):
            if (Array[i ][ j] < 10) & (Array[i][j] > 0):
                count+=1         #l'écriture dans count est fait de manière séquentielle 

    return count 


# %%
size = 4000 
Array  = np.random.randint(1,25, (4000, 4000))

start = time.time()
count = funcDenombres(size,Array)
end = time.time()
print("le temps",end - start)
print ('Mémoire utilisée KB: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
count

# %% [markdown]
# # 6 Profilage du code

# %% [markdown]
# Le profilage permet de connaitre les points chauds, où le code passe le plus de temps ou la consommation des ressources.
#
# * La complexité temporelle, c'est le temps total d'exécution pour résoudre le problème.
#
# * La complexité spatiale, c'est la mémoire totale occupée par le programme.

# %%
# %%prun -D <filename>

funcDenombres(size,Array)

# %% [markdown]
# * ncalls - pour le nombre d'appels
# * tottime- pour le temps total passé dans la fonction donnée (et hors temps passé dans les appels aux sous-fonctions)
# * percall - est le quotient du temps de temps divisé par ncalls
# * cumtime- est le temps cumulé passé dans cette fonction et dans toutes les sous-fonctions (de l'appel à la sortie). Ce chiffre est précis même pour les fonctions récursives
# * percall - est le quotient de cumtime divisé par les appels primitifs
# * filename:lineno(function) - fournit les données respectives de chaque fonction

# %%
# %whos

funcDenombres(size,Array)

# %% [markdown]
# # 7 Optimisations

# %% [markdown]
# " Premature optimization is the root of all evil "
#
# * sortir d'une boucle FOR des calculs ou une allocation mémoire, ce qui permet d'obtenir des gains significatifs.
#
#
# * Eviter l'utilisation des tableaux numpy dans vos boucles.
#
# * Utilisez des vues à la place  des copies  tableaux .
#
# * accéder à un tableau de manière continue est plus rapide que de manière inopinée. Cela implique surtout que de plus petits traitements sont plus rapides (numpy.ascontiguousarray()).
#
# * Utilisation des variables local
#
# * Utilisation map filter reduce 
#
# * Utilisation de numpy car déjà optimisés.
#

# %% [markdown]
# # 8 Performance

# %% [markdown]
# * Le speddup représente le gain en rapidité d'exécution obtenu par son exécution sur plusieurs coeurs de calcul. Sp(c) = T(1)/T(c), c=1,2...
#
#
# * La scalabilité forte, on fixe la taille du problème et on augmente le nombres de coeurs. 
# Si on a une hyperbole:scalabilité forte parfaite.
# On augmente le nombre de coeurs pour calculer plus vite.
#
#
# * La scalabilité faible, on augmente la taille du problème avec le nombre de coeurs.
# Si le temps de calcul est constant:scalabilité forte parfaite.
# On augmente le nombre de coeurs pour résoudre des problèmes de plus grandes tailles.
#

# %%
