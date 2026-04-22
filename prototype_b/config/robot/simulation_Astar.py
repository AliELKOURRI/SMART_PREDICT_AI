# ---------------------------------------------------------
# IMPORTATION DES BIBLIOTHÈQUES
# ---------------------------------------------------------
import numpy as np                      # NumPy est utilisé pour créer et manipuler la grille de l'entrepôt (matrice 2D) de manière optimisée.
import matplotlib.pyplot as plt         # Matplotlib gère la création de la fenêtre d'affichage et le rendu visuel.
import matplotlib.colors as mcolors     # Permet de créer une palette de couleurs personnalisée pour différencier le robot, les murs, le chemin, etc.
import matplotlib.animation as animation # Permet de créer la boucle temporelle qui va faire avancer le robot pas à pas sans bloquer le programme.
import heapq                            # Fournit une "file de priorité". C'est crucial pour A* : ça permet de toujours extraire instantanément la case avec le score F le plus bas.

# ---------------------------------------------------------
# 1. LE MOTEUR MATHÉMATIQUE (A*)
# ---------------------------------------------------------

def heuristique(a, b):
    # Calcule la Distance de Manhattan (déplacement en grille, sans diagonales).
    # 'a' est la case actuelle, 'b' est la cible.
    # C'est la valeur "H" (Heuristique) de notre équation F = G + H.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_etoile(grille, depart, cible):
    # Les 4 vecteurs de direction possibles : (Y, X). Droite, Gauche, Bas, Haut.
    mouvements = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Dictionnaire qui mémorise "de quelle case je viens". Utile pour retracer le chemin à la fin.
    chemin_parent = {}
    
    # Dictionnaire des coûts G (Énergie dépensée depuis le départ). Au départ, le coût est de 0.
    cout_g = {depart: 0}
    
    # Dictionnaire des scores totaux F (F = G + H).
    cout_f = {depart: heuristique(depart, cible)}
    
    # La Liste Ouverte : contient les cases découvertes mais pas encore explorées.
    cases_a_explorer = []
    # On ajoute la case de départ dans la file de priorité avec son score F.
    heapq.heappush(cases_a_explorer, (cout_f[depart], depart))
    
    while cases_a_explorer:
        # heapq.heappop sort TOUJOURS la case qui a le plus petit score F (le chemin le plus prometteur).
        _, case_actuelle = heapq.heappop(cases_a_explorer)
        
        # CONDITION DE VICTOIRE : Si on est sur le produit, on arrête la recherche.
        if case_actuelle == cible:
            chemin_final = []
            # On remonte l'historique de case en case grâce au dictionnaire chemin_parent.
            while case_actuelle in chemin_parent:
                chemin_final.append(case_actuelle)
                case_actuelle = chemin_parent[case_actuelle]
            # On inverse la liste pour avoir le chemin du départ vers la cible.
            return chemin_final[::-1] 
            
        # EXPLORATION DES VOISINS : On regarde les 4 cases autour de la case actuelle.
        for dy, dx in mouvements:
            voisin = (case_actuelle[0] + dy, case_actuelle[1] + dx)
            
            # Vérification 1 : Est-ce que le voisin est bien à l'intérieur des limites de la matrice ?
            if (0 <= voisin[0] < grille.shape[0]) and (0 <= voisin[1] < grille.shape[1]):
                # Vérification 2 : Est-ce que ce voisin est un rack/mur (valeur 1) ?
                if grille[voisin[0]][voisin[1]] == 1: 
                    continue # C'est un mur, on ignore cette case et on passe à la suivante.
            else:
                continue # C'est hors de la carte, on ignore.
                
            # Calcul du nouveau coût G pour ce voisin (coût actuel + 1 pas).
            nouveau_cout_g = cout_g[case_actuelle] + 1
            
            # Si on n'a jamais visité ce voisin, OU si on a trouvé un raccourci pour y aller (nouveau G plus petit) :
            if voisin not in cout_g or nouveau_cout_g < cout_g[voisin]:
                # On met à jour toutes les informations pour ce voisin.
                chemin_parent[voisin] = case_actuelle
                cout_g[voisin] = nouveau_cout_g
                cout_f[voisin] = nouveau_cout_g + heuristique(voisin, cible) # F = G + H
                
                # On ajoute ce voisin à la liste des cases à explorer plus tard.
                heapq.heappush(cases_a_explorer, (cout_f[voisin], voisin))
                
    # Si la boucle while se termine et qu'on n'a pas trouvé la cible, c'est que le produit est emmuré.
    return None 

# ---------------------------------------------------------
# 2. VARIABLES GLOBALES DE LA SIMULATION
# ---------------------------------------------------------
taille = 15
# Création d'une matrice 15x15 remplie de zéros (0 = allées vides).
carte_entrepot = np.zeros((taille, taille), dtype=int)

point_cible = (14, 14) # Coordonnées fixes du produit à récupérer.

# Position dynamique du robot. C'est une liste [] et non un tuple () pour qu'on puisse la modifier quand il roule.
robot_pos = [0, 0] 
chemin_actuel = []     # Va stocker la liste des coordonnées calculées par A*.

# Création d'une "carte de couleurs" discrète pour le rendu graphique :
# Index 0 (Blanc) -> Case vide
# Index 1 (Vert)  -> Le Robot
# Index 2 (Bleu)  -> Le trajet planifié
# Index 3 (Rouge) -> La cible (Produit)
# Index 4 (Noir)  -> Les murs/racks
cmap = mcolors.ListedColormap(['white', 'limegreen', 'dodgerblue', 'crimson', 'black'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5] # Définit les frontières numériques pour chaque couleur.
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Initialisation de la fenêtre graphique (Figure et Axes) avec Matplotlib.
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.1) # Laisse une petite marge en bas.

# ---------------------------------------------------------
# 3. LOGIQUE D'AFFICHAGE ET D'ANIMATION
# ---------------------------------------------------------

def recalculer_chemin():
    # Déclare qu'on va modifier la variable globale chemin_actuel.
    global chemin_actuel
    
    # Fait appel à notre algorithme A* depuis la position *actuelle* du robot.
    chemin = a_etoile(carte_entrepot, tuple(robot_pos), point_cible)
    
    if chemin is not None:
        chemin_actuel = chemin # Mise à jour du trajet.
    else:
        chemin_actuel = None   # Cas où l'entrepôt est bloqué.

def actualiser_affichage():
    ax.clear() # Efface le dessin précédent pour préparer la nouvelle frame.
    
    # On fait une copie de la matrice physique pour pouvoir dessiner dessus sans altérer les vraies données.
    grille_visuelle = np.copy(carte_entrepot)
    
    # Tous les 1 (murs) de la carte deviennent des 4 (couleur noire) sur le visuel.
    grille_visuelle[grille_visuelle == 1] = 4 
    
    # Si on a un chemin valide, on le dessine sur le visuel avec la valeur 2 (couleur bleue).
    if chemin_actuel:
        for (y, x) in chemin_actuel:
            grille_visuelle[y, x] = 2 
            
    # On superpose enfin la cible (Rouge, 3) et le robot (Vert, 1).
    grille_visuelle[point_cible[0], point_cible[1]] = 3 
    grille_visuelle[robot_pos[0], robot_pos[1]] = 1 
    
    # Rendu effectif de la matrice avec notre palette de couleurs.
    ax.imshow(grille_visuelle, cmap=cmap, norm=norm)
    
    # Ajout du quadrillage gris clair pour bien voir les "cases".
    ax.grid(True, which='both', color='gray', linewidth=0.5)
    # Masque les numéros des axes (x, y) pour un rendu plus propre.
    ax.set_xticks(np.arange(-0.5, taille, 1), [])
    ax.set_yticks(np.arange(-0.5, taille, 1), [])
    
    # Gestion dynamique du titre de la fenêtre.
    if chemin_actuel is None:
        ax.set_title("⚠️ ALERTE : Chemin bloqué ! Calcul impossible.", color='red', weight='bold')
    else:
        ax.set_title("Replanning Dynamique A*\n(Cliquez pour ajouter des racks)", color='black')

def deplacer_robot(frame):
    # Fonction appelée automatiquement à chaque "tic" d'horloge par l'animation.
    global robot_pos, chemin_actuel
    
    if chemin_actuel:
        # On extrait (pop) la toute première case de la liste du chemin...
        prochaine_case = chemin_actuel.pop(0)
        # ...et on y téléporte le robot. C'est l'équivalent de l'ordre d'avancer envoyé aux moteurs.
        robot_pos[0], robot_pos[1] = prochaine_case[0], prochaine_case[1]
        
        # Si le robot vient de se poser sur la cible, la mission est finie.
        if tuple(robot_pos) == point_cible:
            chemin_actuel = [] # On vide le chemin pour arrêter de bouger.
            
    # Une fois la position mise à jour, on redessine l'écran.
    actualiser_affichage()

# ---------------------------------------------------------
# 4. INTERACTIONS (Clics de souris pour les obstacles)
# ---------------------------------------------------------

def lors_du_clic(event):
    # Si l'utilisateur clique en dehors de la zone de la grille, on annule.
    if event.xdata is None or event.ydata is None:
        return
        
    # Arrondit les coordonnées du clic (float) pour trouver l'index exact de la case (int).
    x, y = int(round(event.xdata)), int(round(event.ydata))
    
    # Sécurité : On empêche de poser un mur directement sur la tête du robot ou sur le produit.
    if tuple(robot_pos) == (y, x) or point_cible == (y, x):
        return
        
    # Si le clic est bien à l'intérieur de la grille :
    if 0 <= y < taille and 0 <= x < taille:
        # On modifie l'état de l'entrepôt. Si c'était 0 (vide), ça devient 1 (mur). Inversement.
        carte_entrepot[y, x] = 1 if carte_entrepot[y, x] == 0 else 0
        
        # LE COEUR DU REPLANNING : Le monde vient de changer, on force le recalcul immédiat.
        recalculer_chemin()
        # On met à jour l'écran pour que l'utilisateur voie instantanément le mur et le nouveau chemin.
        actualiser_affichage()
        fig.canvas.draw()

# On "branche" notre fonction de clic sur le gestionnaire d'événements de Matplotlib.
fig.canvas.mpl_connect('button_press_event', lors_du_clic)

# ---------------------------------------------------------
# LANCEMENT DU PROGRAMME
# ---------------------------------------------------------

# Avant même de lancer la fenêtre, on calcule le trajet initial.
recalculer_chemin()

# On lance la boucle d'animation. Elle va appeler `deplacer_robot` toutes les 500 millisecondes.
ani = animation.FuncAnimation(fig, deplacer_robot, interval=500, cache_frame_data=False)

# Affichage effectif de l'interface à l'écran.
plt.show()