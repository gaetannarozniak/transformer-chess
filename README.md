###

Pour gérer les librairies python, j'ai demande a chat gpt et askip le mieux c'est d'utiliser un truc qui s'appelle **uv**.
Il faut l'installer au début, puis askip faire ```bash uv sync``` dans le main folder du projet et ça devrait installer les packages bien comme il faut.
Pour ajouter un package il faut faire ```bash uv add numpy pandas```.

### Organisation du projet

On peut separer le code en trois parties :
1. Le transformer
2. L'environnement de RL
3. L'algorithme de RL

Je pensais commencer par definir une classe pour les differents moves possibles.
Il faut choisir d'abord quel format on utilise pour les mouvements.

### Idées

Peut etre avoir une option ou on filtre a la sortie du transformer les coups impossibles.
J'ai peur que sinon a force de tester des coups qui marchent tres rarement (par exemple transformer un pion en dame) soient fortement penalises par le modele car en general interdits, et que donc le modele ne les decouvre jamais. 
Alors que si on filtre, pas besoin pour le modele de les penaliser tant qu'il n'a pas eu l'occasion de les jouer.


On peut peut etre faire un transformer avec trois "tetes": apres l'avant dernier layer, on met trois differentes couches: une pour la position de depart (genre e2) une pour la position d'arrivee (genre f4) et un pour la promotion de piece(genre q, k, ou null). L'objectif est de reduire la taille de l'espace de sortie. En ecrivant je me rends compte que ca marche pas bien parce que la case de depart et d'arrivee sont pas independantes, et donc tirer les deux independamment avec un softmax va mal marcher, a investiguer.
