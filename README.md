###

Pour gérer les librairies python, j'ai demande a chat gpt et askip le mieux c'est d'utiliser un truc qui s'appelle **uv**.
Il faut l'installer au début, puis askip faire ```bash uv sync``` dans le main folder du projet et ça devrait installer les packages bien comme il faut.
Pour ajouter un package il faut faire ```bash uv add numpy pandas```.

### Organisation du projet

On peut separer le code en trois parties :
1. Le transformer
2. L'environnement de RL
3. L'algorithme de RL

Je pensais commencer par definir une classe 

```python 
class 
