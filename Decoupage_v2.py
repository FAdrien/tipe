#!/usr/bin/python3
# -*- coding: utf-8 -*-

########################################################################
# Imports : 
# Pour la précision numérique.
from mpmath import *
mp.dps = 30

# Pour la manipulation d'images.
from PIL import Image, ImageDraw

# Pour la manipulation de fichiers.
import csv
import json
import xml.etree.ElementTree as ET

# Pour la manipulation de tableaux/matrices.
import numpy as np
########################################################################


########################################################################
# Constantes : 
image = "../Data/CentreVille.png"
carte = Image.open(image)
draw = ImageDraw.Draw(carte, 'RGBA') # Pour la transparence.

longueur, largeur = carte.size # Longueur et largeur de la carte.

# Latitude et longitude des coins supérieurs gauche & inférieur droit ainsi que les ratios associés (pour la mise à l'échelle).
longitude_min, longitude_max = -0.60696, -0.50681
longitude_ratio = longueur / (longitude_max - longitude_min)

latitude_min, latitude_max = 44.81924, 44.86563
latitude_ratio = largeur / (latitude_max - latitude_min)

RT = 6370000 # Rayon terrestre approximatif (en mètres).
########################################################################


########################################################################
# Utilitaire extraction/dessin de points, conversion de coordonnées, calcul de distance.
def est_entre(l, a, b):
	'''Tester si le r&el l est entre a et b (i.e. dans [a, b] si a <= b sinon, dans [b, a]).'''
	return a <= l <= b or b <= l <= a

def polaire2pixel(point):
	'''Renvoie les coordonnées du pixel correspondant aux coordonnées polaires fournies.'''
	lat, lon = point
	x, y = int((lon - longitude_min) * longitude_ratio), int((latitude_max - lat) * latitude_ratio)
	return (x, y)

def estDansRectangle(p, c = (0, 0), L = longueur, l = largeur):
	'''Tester si le point p est dans le rectangle de coin supérieur gauche c, de longueur L et de largeur l.'''
	return c[0] <= p[0] < c[0] + L and c[1] <= p[1] < c[1] + l

def construirePoints(fichier, L = 10, l = 11):
	''' L : indice de la longitude, l : indice de la latitude.'''
	liste = []
	with open(fichier, encoding = 'utf-8') as csvfile:
		for ligne in csv.reader(csvfile, delimiter = ','):
			longitude, latitude = float(ligne[L]), float(ligne[l])
			x, y = polaire2pixel((latitude, longitude))

			if estDansRectangle((x, y)):
				liste.append((x, y))
	return liste

def dessinerPoints(points, couleur, rayon = 1, canvas = draw):
	'''Dessiner sur la carte des points selon leur coordonnées (x, y).'''
	for (x, y) in points:
		canvas.ellipse((x - rayon, y - rayon, x + rayon, y + rayon), fill = couleur, outline = couleur)

def distance_reelle(p1, p2, r = RT):
	'''Renvoie la 'distance réelle' entre deux points (latitude, longitude) a une altitude r (typiquement, le rayon de la Terre).
	On utilise la formule de haversine.'''
	lat1, lon1, lat2, lon2 = radians(p1[0]), radians(p1[1]), radians(p2[0]), radians(p2[1]) # Conversion en radian des latitudes et des longitudes des deux points.
	expr = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2 # Formule de haversine (on procède en deux temps).
	# On arrondi à 5 décimales après la virgule (gain de mémoire).
	return round(2 * r * asin(sqrt(expr)), 5)

def dist2(p1, p2 = (0, 0)):
	'''Norme L2 au carré, par défaut (si p2 n'est pas fourni) on renvoie la norme au carrée.'''
	(x1, y1), (x2, y2) = p1, p2
	return (x1 - x2)**2 + (y1 - y2)**2

# Fonctions utiles sur les vecteurs et les matrices (les matrices sont des listes de listes; représentation ligne à ligne).
def vecteur(p, q):
	'''Renvoie le vecteur associé au bipoint pq.'''
	return [qi - pi for (pi, qi) in zip(p, q)]

def normaliser(v):
	'''Normalier un vecteur.'''
	assert v != [0, 0]
	d = norm(v)
	return [v[0] / d, v[1] / d]

def ortho(v):
	'''Renvoie un vecteur du plan orthogonal à v, en sens direct.'''
	assert v != [0, 0], 'Vecteur nul, ambiguïté sur son orthogonal !'
	return [v[1], -v[0]]

def isobarycentre(points):
	'''Renvoie les coordonnées de l'isobarycentre des points du plan fournis.
	Hypothèse : la liste des points est non vide !'''
	X, Y, n = mpf(0), mpf(0), len(points)
	for (x, y) in points:
		X += mpf(x)
		Y += mpf(y)
	return [X / n, Y / n]

def determinantMat2(M):
	'''Calcul de déterminant de la matrice 2 x 2 M'''
	return M[0][0] * M[1][1] - M[0][1] * M[1][0]

def determinantVect2(p1, p2, p3):
	'''Calcul le déterminant du système de vecteurs plans (p1p2, p1p3)'''
	return determinantMat2([vecteur(p1, p2), vecteur(p1, p3)])

def positivementLies(v1, v2, eps = 1e-20):
	'''Tester si deux vecteurs du plan son positivement liés, une précision est fournie.'''
	# Si v2 est nul, 0 * v1 = v2, sinon on teste si v1 et v2 sont colinéaires et si leur produit scalaire est positif (car v2 =/= (0, 0)).
	return v2 == [0, 0] or (abs(determinantMat2([v1, v2])) <= eps and fdot(v1, v2) >= 0)
########################################################################


########################################################################
# Centre du cercle circonscrit, intersection de deux paraboles et de deux droites.
def cercleCirconscrit(p1, p2, p3):
	'''Renvoie le rayon et le centre du cercle circonscrit aux points p1, p2 et p3 (s'il ne sont pas colinéaires).'''
	x, y = intersectionDroites(isobarycentre([p1, p2]), ortho(vecteur(p1, p2)), isobarycentre([p1, p3]), ortho(vecteur(p1, p3)))
	return (x, y), sqrt(dist2(p1, (x, y)))

def intersectionParaboles(p1, p2, y, sens):
	'''Pour deux points, renvoie le point d'intersection (à gauche 'g' ou à droite 'd' suivant le sens) entre les deux paraboles
	de foyers p1, p2 et de directrice la droite horizontale d'ordonnée y.'''
	if p1[1] == p2[1]: # Dans ce cas leur deux breakpoints (p1, p2) et (p2, p1) seront confondus.
		assert p1[0] != p2[0], 'Mêmes foyers !'
		if p1[1] == y: # Cas particulier (rare...).
			return ((p1[0] + p2[0]) / 2, float('-inf'))
		lbd = ((p1[1] - y) / (p2[0] - p1[0]) - (p2[0] - p1[0]) / (4 * (p1[1] - y))) / 2
	else:
		# IMPORTANT : renvoie le point d'intersection à GAUCHE ssi sens = 'g' sinon, c'est le point à DROITE.
		e = -1
		if (p1[1] > p2[1] and sens == 'g') or (p1[1] < p2[1] and sens == 'd'):
			e = 1
		Delta = (y - p1[1]) * (y - p2[1]) * dist2(p1, p2)
		assert Delta >= 0, 'Intersection de paraboles vide !'
		lbd = ((p2[0] - p1[0]) * (y - (p1[1] + p2[1]) / 2) + e * sqrt(Delta)) / (p2[1] - p1[1])**2

	return (lbd * (p2[1] - p1[1]) + (p1[0] + p2[0]) / 2, lbd * (p1[0] - p2[0]) + (p1[1] + p2[1]) / 2)

def intersectionDroites(p1, v1, p2, v2):
	'''Renvoie l'unique point d'intersection entre les deux droites d1 et d2 passant par p1 (resp. p2) de vecteur 
	directeur v1 (resp. v2). Aussi, les droites ne doivent pas être parallèles (ni confondues).'''
	assert v1 != [0, 0] and v2 != [0, 0], 'Vecteur directeur nul !' # Ce test peut être redondant...
	assert determinantMat2([v1, v2]) != 0, 'Droites parallèles.' # Idem...
	# On se ramène à la résolution d'un système linéaire 2x2.
	# On cherche x, y tels que p1 + x*v1 = p2 + y*v2 i.e. p2 - p1 = x*v1 - y*v2 = (v1 v2) * (x -y) (matrice de v1, v2).
	mat = matrix([[v1[0], v2[0]], [v1[1], v2[1]]])
	vec = vecteur(p1, p2)
	x, ny = (mat**(-1)) * matrix([[vec[0]], [vec[1]]]) # Produit entre une matrice et un vecteur.
	return matrix([[p1[0]], [p1[1]]]) + x * matrix([[v1[0]], [v1[1]]]) # Le point d'intersection.

def appartientDemiDroite(p, p1, v1):
	'''Tester si le point p appartient à la demi-droite (p1, v1).'''
	assert v1 != [0, 0], 'Pas de demi-droite !'
	# p est sur la demi-droite d'origine p1 et de vecteur directeur v1 ssi il existe x >=0 tel que p = p1 + x*v1 i.e. les vecteurs p - p1 et v1 sont positivement liés.
	return positivementLies(vecteur(p1, p), v1)

def intersecteDemiDroite(p1, v1, p2, v2):
	'''Pour deux demi-droites, renvoie True si les demi-droites passées en argument s'intersectent dans le plan.
	On a p1, p2 deux points du plan et v1, v2 les vecteurs directeurs de ces demi-droites respectives.'''
	assert v1 != [0, 0] and v2 != [0, 0], 'Vecteur directeur nul !'
	if p1 == p2: return True # Même origine.
	if determinantMat2([v1, v2]) == 0: # Les vecteurs v1 et v2 son colinéaires (rare...).
		# On vérifie si : les droites associées aux demi-droites sont confondues et si l'origine de l'une d'elle appartient à l'autre demi-droite.
		return determinantMat2([v1, vecteur(p1, p2)]) == 0 and (appartientDemiDroite(p1, p2, v2) or appartientDemiDroite(p2, p1, v1))
	# Sinon, les droites associées à ces demi-droites ont un unique point d'intersection dont on teste son appartenance aux demi-droites.
	p = intersectionDroites(p1, v1, p2, v2)
	return appartientDemiDroite(p, p1, v1) and appartientDemiDroite(p, p2, v2)
########################################################################


########################################################################
# Tri fusion : 
def fusionner(t, g, m, d, comp):
	'''Fusion ordonnée de t[g:m] et t[m:d] dans t[g:d].'''
	i, j = 0, m - g
	t2 = t[g:d] # On mémorise les cases qui vont être modifiées.
	for k in range(g, d):
		if i < m - g and (j == d - g or comp(t2[i], t2[j])):
			t[k] = t2[i]
			i += 1
		else:
			t[k] = t2[j]
			j += 1

def tri_fusion(t, m = 0, M = None, comp = lambda x, y : x < y):
	'''Tri de t[m:M] par tri fusion sur place.'''
	def tri_aux(g, d):
		'''Trier le sous-tableau t[g:d].'''
		if g < d - 1: # Au moins deux éléments.
			m = (g + d) // 2
			tri_aux(g, m)
			tri_aux(m, d)
			fusionner(t, g, m, d, comp) # À l'issu, t[g:d] sera trié.
	if M == None: M = len(t)
	tri_aux(m, M)
########################################################################


########################################################################
# Implémentation de l'algorithme d'Andrew pour le calcul d'enveloppes convexes : 
# Complexité : temporelle O(n log(n)), spatiale O(n), n est le nombre de points.
def enveloppe_convexe(P):
	'''Renvoie l'enveloppe convexe de P.'''
	tri_fusion(P) # L'ordre lexicographique est naturellement utilisée sur les tuples/listes.
	L = []
	# Calcul de l'enveloppe convexe supérieure.
	for p in P:
		while len(L) >= 2 and determinantVect2(L[-2], L[-1], p) <= 0: L.pop()
		L.append(p)
	l = len(L)
	# Calcul de l'enveloppe convexe inférieure.
	for i in range(len(P) - 2, -1, -1):
		while len(L) >= l + 1 and determinantVect2(L[-2], L[-1], P[i]) <= 0: L.pop()
		L.append(P[i])
	return L # Le dernier point est en double (c'est pour le dessin).

def dessiner_enveloppe_convexe(env, canvas = draw):
	'''Dessine les enveloppes convexes des groupes sur la carte.'''
	for grp in env:
		for i in range(len(grp) - 1):
			canvas.line((grp[i], grp[i + 1]), fill = 'green')
########################################################################


########################################################################
# Implémentation d'un tas (binaire) min (pour une file de priorité).
class TasMin:
	def __init__(self, n, l = [], comp = lambda x, y: x < y, indexation = False):
		'''Tas binaire max à n éléments.'''
		self.t = [None] * n # Taille du tas-min.
		self.idx = 0 # Indice de la position "libre".
		self.comp = comp # Fonction de comparaison.
		self.indexation = indexation # Pour mémoriser l'indice des objets qui se trouve dans le tableau (facilite leur retrait, sans parcourir tout le tas).

		for e in l:
			self.ajouter(e) # Si des éléments sont passés en argument, on les ajoute dans le tas.

	def estPlein(self): return self.idx == len(self.t) # Tester si le tas est plein.
	def estVide(self): return self.idx == 0 # Tester si le tas est vide.

	def parent(self, i): return (i - 1) // 2 # Renvoie l'indice du nœud parent.
	def filsG(self, i): return 2 * i + 1 # Renvoie l'indice du fils gauche.
	def filsD(self, i): return 2 * i + 2 # Renvoie l'indice du fils droit.

	def echanger(self, i, j):
		'''Échanger deux éléments dans un tableau.'''
		self.t[i], self.t[j] = self.t[j], self.t[i]
		if self.indexation: # Si il y a indexation sur les éléments du tas, on met à jour les indices.
			if self.t[i] != None: self.t[i].idx = i
			if self.t[j] != None: self.t[j].idx = j

	def percole_haut(self, i):
		'''Faire "remonter" la clé d'indice i si besoin.'''
		j = self.parent(i)
		while 0 <= j and self.comp(self.t[i], self.t[j]): # comp(x, y), si x < y
			self.echanger(i, j)
			i, j = j, self.parent(j)

	def percole_bas(self, i):
		'''Faire "descendre" la clé d'indice i si besoin.'''
		g, d = self.filsG(i), self.filsD(i)
		while d < self.idx and (self.comp(self.t[g], self.t[i]) or self.comp(self.t[d], self.t[i])): # Si les deux fils sont définis et si l'invariant du tas n'est pas satisfait.
			m = g if self.comp(self.t[g], self.t[d]) else d
			self.echanger(i, m)
			g, d, i = self.filsG(m), self.filsD(m), m

		if d == self.idx and self.comp(self.t[g], self.t[i]): # Si le fils droit n'est pas défini mais, que le fils gauche l'est et que l'invariant du tas n'est pas vérifié.
			self.echanger(i, g)

	def ajouter(self, e):
		'''Ajouter une clé.'''
		if self.estPlein(): raise Exception('Tas plein !')
		self.t[self.idx] = e
		if self.indexation: e.idx = self.idx
		self.percole_haut(self.idx)
		self.idx += 1

	def extraire(self, i):
		'''Extraire la clé d'indice i.'''
		if self.estVide(): raise Exception('Tas vide !')
		v, self.t[i] = self.t[i], None
		self.idx -= 1
		self.echanger(i, self.idx)
		self.percole_bas(i)
		return v

	def minimum(self):
		return self.extraire(0)
########################################################################


########################################################################
# Implémentation d'un arbre binaire AVL (arbre binaire de recherche équilibré).
# Il est primordial (cf. fonction ajouter_arc) que les nœuds les plus profond de 
# l'arbre soient des feuilles : une feuille n'ayant aucun fils.
class AVL:
	class Noeud:
		def __init__(self, etiquette, filsG = None, filsD = None):
			'''Nœud (nœud interne ou feuille).'''
			self.etiquette = etiquette
			self.filsG, self.filsD = filsG, filsD
			self.hauteur = 1

		def estFeuille(self):
			'''Tester si un nœud est une feuille.'''
			return self.filsG == self.filsD == None

		def filsDroit(self, fils):
			'''Mettre à jour le fils droit de l'arbre.'''
			self.filsD = fils
			self.mettre_hauteur_a_jour()

		def filsGauche(self, fils):
			'''Mettre à jour le fils gauche de l'arbre.'''
			self.filsG = fils
			self.mettre_hauteur_a_jour()

		def mettre_hauteur_a_jour(self):
			'''Met à jour la hauteur du noeud courant (normalement, .'''
			self.hauteur = 1 + max(self.filsG.hauteur, self.filsD.hauteur)

	def __init__(self, comp = lambda x, y: x < y):
		'''Arbre binaire AVL.'''
		self.comp = comp

	def hauteur(self, arb):
		'''Calcul de la hauteur de l'arbre.'''
		if arb == None: return 0
		return arb.hauteur

	def balancement(self, arb):
		'''Facteur de balancement de l'arbre.'''
		if arb == None: return 0
		return self.hauteur(arb.filsG) - self.hauteur(arb.filsD)

	def rotationG(self, arb):
		'''Rotation à gauche de l'arbre, arb n'est pas une feuille.'''
		g, d = arb.filsG, arb.filsD
		if d != None:
			arb.filsDroit(d.filsG)
			d.filsGauche(arb)
			return d
		return arb

	def rotationD(self, arb):
		'''Rotation à droite de l'arbre, arb n'est pas une feuille.'''
		g, d = arb.filsG, arb.filsD
		if g != None:
			arb.filsGauche(g.filsD)
			g.filsDroit(arb)
			return g
		return arb

	def reequilibrer(self, arb):
		'''Ré-équilibrer l'arbre binaire après une insertion/suppression (on fait l'hypothèse que, si un ré-équilibrage est 
		nécessaire alors, le balancement vaut ou bien 2 ou bien -2 : ceci garantit que ré-équilibrer un sous-arbre se fasse en 
		temps contant).'''
		if arb != None:
			b = self.balancement(arb)
			if b > 1: # Si le sous-arbre gauche est plus "profond" que le droit.
				if self.balancement(arb.filsG) < 0:
					arb.filsGauche(self.rotationG(arb.filsG))
				return self.rotationD(arb)
			if b < -1: # Si le sous-arbre droit est plus "profond" que le gauche.
				if self.balancement(arb.filsD) > 0:
					arb.filsDroit(self.rotationD(arb.filsD))
				return self.rotationG(arb)
		return arb

	def rechercher(self, arb, etiquette):
		'''Recherche de la feuille de l'arbre la "plus proche" de l'étiquette.'''
		if arb == None or arb.estFeuille():
			return arb

		if self.comp(etiquette, arb.etiquette):
			return self.rechercher(arb.filsG, etiquette)
		else:
			return self.rechercher(arb.filsD, etiquette)

	def ajouter_arc(self, arb, nouveau_arc):
		'''Permet d'ajouter un arc de parabole dans l'arbre.'''
		if arb == None:
			return self.Noeud(nouveau_arc)
		elif arb.estFeuille():
			assert isinstance(arb.etiquette, Arc), 'Erreur de feuille !' # On doit garantir que les feuilles de l'AVL sont toujours des arcs.
			p1, p2 = arb.etiquette.coords(), nouveau_arc.coords() # Les sites concernés.

			b1, b2 = BreakPoint(p1, p2, 'g', Arete()), BreakPoint(p1, p2, 'd', Arete()) # On construit les breakpoints (chaque breakpoint "trace" une demi-arête).
			b1.jumeau, b2.jumeau = b2, b1 # Gestion des breakpoints jumeaux.
			b1.arete.arete_soeur, b2.arete.arete_soeur = b2.arete, b1.arete # Ces deux breakpoints tracent des demi-arêtes sœurs.

			voisin_gauche, voisin_droit = arb.etiquette.voisin_gauche, arb.etiquette.voisin_droit # Gestion des pointeurs vers les feuilles voisines, on 'recycle' l'arc arc.etiquette.
			a2, a3 = nouveau_arc, Arc(p1, None, voisin_droit, None, arb.etiquette.breakpoint_droit)
			if voisin_droit != None:
				voisin_droit.voisin_gauche = a3
			arb.etiquette.voisin_droit, a3.voisin_gauche = a2, a2
			a2.voisin_gauche, a2.voisin_droit = arb.etiquette, a3

			n1 = self.Noeud(b1, arb, self.Noeud(a2))
			n2 = self.Noeud(b2, n1, self.Noeud(a3))

			arb.etiquette.breakpoint_droit = b1
			a3.breakpoint_gauche = b2
			a2.breakpoint_gauche, a2.breakpoint_droit = b1, b2
			return n2

		elif self.comp(nouveau_arc, arb.etiquette): # Descente dans l'AVL.
			arb.filsGauche(self.ajouter_arc(arb.filsG, nouveau_arc))
		else:
			arb.filsDroit(self.ajouter_arc(arb.filsD, nouveau_arc))

		return self.reequilibrer(arb) # Ré-équilibrage de l'AVL.

	def supprime_arc(self, arb, etiquette, noeud_g = None, noeud_d = None):
		'''Permet de supprimer un arc de l'AVL : parmi deux breakpoints qui fusionnent, l'un sera supprimé, l'autre substitué 
		(il nous faut donc les repérer lors de la descente dans l'AVL).'''
		assert arb != None, 'Arbre binaire équilibré invalide !'
		if arb.estFeuille():
			assert arb.etiquette.id == etiquette.id, 'Erreur : arc invalide !'
			return None # Suppression de l'arc (si l'AVL n'était constitué que d'un seul arc).

		elif self.comp(etiquette, arb.etiquette): # Descente dans l'AVL (à gauche).
			if arb.filsG.estFeuille(): # On doit supprimer cette feuille ainsi que le nœud courant.
				suppression_arc(arb.filsG.etiquette, noeud_g.etiquette, arb.etiquette, noeud_g)
				return arb.filsD # Le fils droit est déjà équilibré.
			else:
				arb.filsGauche(self.supprime_arc(arb.filsG, etiquette, noeud_g, arb)) # On descend dans le fils gauche en gardant une trace du dernier BreakPoint 'droit'.
		else:
			if arb.filsD.estFeuille(): # On doit supprimer cette feuille ainsi que le nœud courant.
				suppression_arc(arb.filsD.etiquette, arb.etiquette, noeud_d.etiquette, noeud_d)
				return arb.filsG # Ce sous-arbre est déjà équilibré.
			else:
				arb.filsDroit(self.supprime_arc(arb.filsD, etiquette, arb, noeud_d)) # On descend dans le fils droit en gardant une trace du dernier BreakPoint 'gauche'.

		return self.reequilibrer(arb) # Ré-équilibrage de l'AVL.

	def parcours_prefixe(self, arb, f):
		'''Parcours préfixe de l'arbre avec traitement de chaque nœud.'''
		if arb != None:
			f(arb)
			self.parcours_prefixe(arb.filsG, f)
			self.parcours_prefixe(arb.filsD, f)

def suppression_arc(arc, bg, bd, noeud):
	'''Cette fonction se charge des formalités lors de la suppression d'un arc.'''
	nouveau_breakpoint = breakpoint_convergeant(bg, bd, arc.voisin_gauche.p, arc.voisin_droit.p) # Le nouveau breakpoint à "substituer" dans l'arbre AVL.
	noeud.etiquette = nouveau_breakpoint
	# On met à jour les voisins pour les arcs.
	arc.voisin_gauche.voisin_droit = arc.voisin_droit
	arc.voisin_droit.voisin_gauche = arc.voisin_gauche
	arc.voisin_gauche.breakpoint_droit, arc.voisin_droit.breakpoint_gauche = nouveau_breakpoint, nouveau_breakpoint # On met à jour les breakpoints voisins pour les arcs.
########################################################################


########################################################################
# Implémentation d'une géométrie dite en "demi-arêtes".
# Complexité (spatiale) : linéaire en le nombre de composantes de la géométrie.
ID_SOMMET = 0 # Identifiant pour les sommets.
ID_ARETES = 0 # Identifiant pour les arêtes.
ID_FACE = 0 # Identifiant pour les faces.

# Liste des composantes du diagrammes de Voronoï : 
liste_sommets = []
liste_aretes = []
liste_faces = []

class Sommet:
	def __init__(self, x, y, couleur = 'black'):
		global ID_SOMMET
		self.x, self.y = x, y

		self.couleur = couleur # Couleur du sommet.

		self.id = ID_SOMMET # À chaque sommet du diagramme de Voronoï est associé un unique identifiant qui servira pour convertir la géométrie dans un format "adéquat" (cf. travail de mon camarade).
		ID_SOMMET += 1
		liste_sommets.append(self)

	def coords(self):
		'''Renvoyer les coordonnées du sommets.'''
		return self.x, self.y

	def dessiner(self, rayon = 1, canvas = draw):
		'''Pour dessiner un sommet sur le convas fourni.'''
		dessinerPoints([(self.x, self.y)], self.couleur, rayon, canvas)

class Arete:
	def __init__(self, arete_soeur = None, arete_suivante = None, arete_precedente = None, sommet_but = None, face = None, couleur = 'black'):
		global ID_ARETES
		self.arete_soeur = arete_soeur
		self.arete_suivante = arete_suivante
		self.arete_precedente = arete_precedente

		self.sommet_but = sommet_but
		self.face = face

		self.couleur = couleur # Couleur de l'arête.

		self.id = ID_ARETES
		ID_ARETES += 1
		liste_aretes.append(self)

	def parcours_devant(self, f):
		'''Parcourir les arêtes d'une même face en suivant le sens trigonométrique.'''
		arete = self
		while arete.arete_suivante != None and arete.arete_suivante.id != self.id:
			f(arete)
			arete = arete.arete_suivante
		f(arete)

	def dessiner(self, canvas = draw):
		'''Pour dessiner une arête sur le canvas fourni.'''
		if self.sommet_but != None and self.arete_soeur.sommet_but != None:
			canvas.line((self.sommet_but.coords(), self.arete_soeur.sommet_but.coords()), fill = self.couleur)

class Face:
	def __init__(self, arete_associee):
		global ID_FACE
		self.arete_associee = arete_associee

		self.id = ID_FACE
		ID_FACE += 1
		liste_faces.append(self)

	def parcours_profondeur(self, f):
		'''Parcours en profondeur de la géométrie depuis la face courante.'''
		faces_visitees = [False] * ID_FACE
		pile = [self]

		def ajouter_nouvelles_faces(arete):
			# Permet d'ajouter les faces encore non visitées dans la pile.
			if arete.arete_soeur.face != None and not faces_visitees[arete.arete_soeur.face.id]:
				pile.append(arete.arete_soeur.face)

		while pile != []:
			face = pile.pop()
			f(face) # Retrait d'une face et traitement de cette dernière.

			faces_visitees[face.id] = True # Marquer la face courante comme visitée.
			face.arete_associee.parcours_devant(ajouter_nouvelles_faces) # Pour ajouter les potentielles nouvelles faces.

	def regrouper(self, face):
		'''Permet d'incorporer les aretes de la face donnée dans la face courante.'''
		arete_1, arete_2 = face.arete_associee, face.arete_associee
		while arete_1.arete_suivante != None and arete_1.arete_suivante.id != face.arete_associee.id:
			arete_1.face = self
			arete_1 = arete_1.arete_suivante

		while arete_2.arete_precedente != None and arete_2.arete_suivante.id != face.arete_associee.id:
			arete_2.face = self
			arete_2 = arete_2.arete_precedente

	def dessiner(self, canvas = draw):
		'''Pour dessiner une face sur le canvas fourni.'''
		self.arete_associee.parcours_devant(lambda arete: arete.dessiner(canvas))

	def colorier(self):
		l = []
		arete = self.arete_associee
		while arete.arete_suivante != None and arete.arete_suivante.id != self.arete_associee.id:
			l.append(arete.sommet_but.coords())
			arete = arete.arete_suivante
		if arete.arete_suivante != None and arete.arete_suivante.id == self.arete_associee.id:
			l.append(arete.sommet_but.coords())
		else:
			arete = self.arete_associee
			while arete.arete_precedente != None and arete.arete_precedente.id != self.arete_associee.id:
				l = [arete.sommet_but.coords()] + l
				arete = arete.arete_precedente

class Geometrie:
	def __init__(self, faces):
		self.faces = faces

	def dessiner(self, canvas = draw):
		'''Dessiner la géométrie sur le canvas fourni.'''
		for face in self.faces:
			face.dessiner(canvas)
########################################################################


########################################################################
# Fonctions de manipulation de la géométrie & utilitaire pour le clippage des cellules.
def chemin_aretes(liste_points, face_g, face_d, couleur = 'black'):
	'''Pour une liste de point données, séparant deux faces, renvoie le chemin d'arêtes passant par tous ces points, dans l'ordre.
	Attention, les arêtes aux extrémités auront des paramètres incomplets !'''
	assert len(liste_points) >= 2, 'Pas assez de points !'
	sommets = [Sommet(p[0], p[1]) for p in liste_points]

	a_g = [Arete(None, None, None, sommets[1], face_g)]
	a_d = [Arete(None, None, None, sommets[0], face_d)]
	a_g[0].arete_soeur, a_d[0].arete_soeur = a_d[0], a_g[0]

	for i in range(1, len(liste_points) - 1):
		arete_g = Arete(None, None, a_g[-1], sommets[i + 1], face_g)
		arete_d = Arete(None, a_d[-1], None, sommets[i], face_d)
		arete_g.arete_soeur, arete_d.arete_soeur = arete_d, arete_g

		a_g[-1].arete_suivante, a_d[-1].arete_precedente = arete_g, arete_d

		a_g.append(arete_g)
		a_d.append(arete_d)

	return a_g, a_d

def clipper_demi_droite(p, v, c = (0, 0), L = longueur, l = largeur):
	'''Renvoie le point d'intersection entre la demi-droite d'origine p et de vecteur directeur v avec le rectangle
	de coin supérieur gauche c, de longueur L (sens des x croissants: de la gauche vers la droite) et de largeur l 
	(sens des y croissant : du haut vers le bas).'''
	assert v != [0, 0] and estDansRectangle(p), 'Données invalides pour le clippage !'
	x, y = c
	# Cas de demi-droites verticales ou horizontales.
	if v[0] == 0:
		y1 = y if v[1] > 0 else y + l
		return (p[0], y1)
	elif v[1] == 0:
		x1 = x if v[0] > 0 else x + L
		return (x1, p[1])
	else: # Cas de demi-droites 'obliques'.
		if v[1] < 0: # Coin supérieur gauche.
			p_inter = intersectionDroites(p, v, c, (1, 0))
		else: # Coin inférieur gauche.
			p_inter = intersectionDroites(p, v, (x, y + l), (1, 0))

		if 0 <= p_inter[0] <= L:
			return p_inter
		elif v[0] > 0:
			return intersectionDroites(p, v, (x + L, y), (0, 1))
		return intersectionDroites(p, v, c, (0, 1))

def relier_points_suivant_rectangle(p1, p2, face = None, c = (0, 0), L = longueur, l = largeur):
	'''Fonction permettant de relier deux sommets sur les bords d'un rectangle.'''
	x, y = c # Coordonnées du coin supérieur gauche.
	coins = [(0, c), (1, (x, y + l)), (0, (x + L, y + l)), (1, (x + L, y))] # Les 4 coins (sens trigonométrique).
	coin_precedent, i = p1, -4 # Coin précédent et indice du coin suivant.
	# Choix de l'indice de départ i : 
	if p1[0] == 0:
		i = -3
	elif p1[1] == y + l:
		i = -2
	elif p1[0] == x + L:
		i = -2

	sommets_chemin = [p1]
	for k in range(i, i + 4):
		s, p = coins[k]
		if p2[1 - s] == p[1 - s] and est_entre(p2[s], coin_precedent[s], p[s]):
			break
		sommets_chemin.append(p)
		coin_precedent = p
		k += 1
	sommets_chemin.append(p2)
	return chemin_aretes(sommets_chemin, face, None)

def clipper_face(face, L = longueur, l = largeur):
	'''Clipper la face donnée avec la fenêtre rectangulaire de coin supérieur gauche (0,0), de longueur L et de largeur l
	Hypothèse : toutes les arêtes possèdent un sommet d'origine.'''
	p_d, p_g = None, None # Les deux sommets à relier (on les met dans l'ordre suivant le parcourt trigonométrique des arêtes d'une face).
	arete_d, arete_g = face.arete_associee

	# Recherche des deux arêtes qu'il faut relier.
	while arete_g.arete_suivante != None:
		arete_g = arete_g.arete_suivante
	p_g = arete_g.sommet_but

	while arete_d.arete_precedente != None:
		arete_d = arete_d.arete_precedente
	p_d = arete_d.arete_soeur.sommet_but

	# Régler le cas des arêtes en début et fin de liste (pas besoin de le faire pour la liste a_d, ses arêtes ne sont pas utilisées) : 
	a_g[0].arete_precedente, a_g[-1].arete_suivante = arete_g, arete_d
########################################################################


########################################################################
# Implémentation de l'algorithme de Fortune pour le calcul du diagramme de Voronoï : 
# Complexité : temporelle O(n log(n)), spatiale O(n), n est le nombre de points.
# Les composantes de l'arbre binaire équilibré : 
ordonnee = 0 # Ordonnée de la droite de balayage.

def direction(b):
	'''Renvoie la direction vers laquelle se dirige le breakpoint b.'''
	p1, p2 = b.p1, b.p2
	if p1[0] == p2[0]:
		if p1[1] < p2[1]:
			if b.sens == 'g':
				return ortho(vecteur(p2, p1))
			return ortho(vecteur(p1, p2))
		else:
			if b.sens == 'g':
				return ortho(vecteur(p1, p2))
			return ortho(vecteur(p2, p1))
	else:
		if p1[1] > p2[1]:
			if b.sens == 'g':
				return ortho(vecteur(p1, p2))
			return ortho(vecteur(p2, p1))
		else:
			if b.sens == 'g':
				return ortho(vecteur(p2, p1))
			return ortho(vecteur(p1, p2))

class BreakPoint:
	def __init__(self, p1, p2, sens, arete, jumeau = None):
		'''Intersection des paraboles de foyer p1, p2 et de directrice 
		la droite de balayage. Arête est un pointeur vers l'arête que trace le breakpoint.'''
		# On s'assure que le point p1 est le site plus à gauche et que p2 est le site le plus à droite.
		assert p1 != p2, 'BreakPoint invalide !'
		if p1[0] > p2[0]: self.p1, self.p2 = p2, p1 # On place les sites dans l'ordre des x croissants.
		else: self.p1, self.p2 = p1, p2

		self.sens = sens # Savoir s'il s'agit du BreakPoint gauche ou du BreakPoint droit.
		self.jumeau = jumeau # BreakPoint 'jumeau', celui qui trace l'arête sœur du BreakPoint courant (utile pour la construction des arêtes, demi-droites et des droites).
		self.arete = arete # L'arête tracée par le breakpoint.
		self.direction = direction(self) # La direction (sous forme de vecteur) que suit le breakpoint.

	def coords(self):
		'''Renvoie les coordonnées du breakpoint (dépendent de la position de la droite
		de balayage).'''
		return intersectionParaboles(self.p1, self.p2, ordonnee, self.sens)

# Plusieurs arcs attachée à un même site p peuvent apparaître dans l'AVL !
class Arc:
	def __init__(self, p, voisin_gauche = None, voisin_droit = None, breakpoint_gauche = None, breakpoint_droit = None, evtcercle = None):
		'''Représente un arc de parabole défini par le point p(x, y).'''
		self.p = p
		self.x, self.y = p
		self.evtcercle = evtcercle
		self.voisin_gauche, self.voisin_droit = voisin_gauche, voisin_droit # Pointeur vers les feuilles voisines dans l'AVL.
		self.breakpoint_gauche, self.breakpoint_droit = breakpoint_gauche, breakpoint_droit # Pointeur vers les nœuds internes où se trouvent les breakpoints aux extrémités de l'arc.

	def coords(self):
		return self.x, self.y

# Les événements de la file de priorité, chaque événement possède un indice idx, sa position dans la file de priorité : 
class EvtPoint:
	def __init__(self, p):
		'''p est un point du plan de la forme (x, y).'''
		self.x, self.y = p
		self.idx = 0

	def coords(self):
		return self.x, self.y

class EvtCercle:
	def __init__(self, p1, p2, p3, arc):
		'''p1, p2, p3 trois sites deux-à-deux distincts.'''
		self.p1, self.p2, self.p3 = p1, p2, p3
		self.idx = 0

		self.centre, self.r = cercleCirconscrit(p1, p2, p3)
		self.x, self.y = self.centre[0], self.centre[1] + self.r # On prend le point du cercle "le plus bas" (c'est lui qui défini l'événement).
		self.arc = arc

	def coords(self):
		return self.x, self.y

def direction_convergence(bg, bd, p1, p2):
	'''Renvoie le sens dans lequel se dirige le nouveau breakpoint issu de la convergence de bg et de bd. Ce nouveau breakpoint 
	est associé aux sites p1 et p2.'''
	dir_g, dir_d = normaliser(bg.direction), normaliser(bd.direction) # Direction normalisée des deux breakpoints convergents.
	if determinantMat2([[dir_g[0], dir_d[0]], [dir_g[1], dir_d[1]]]) >= 0: # On s'arrange pour avoir dir1 suivi de dir2 dans le sens trigonométrique.
		dir_g, dir_d = dir_d, dir_g

	dir_site = normaliser(ortho(vecteur(p1, p2))) # On prend une direction normalisée, a priori arbitraire, de la médiatrice des sites p1 et p2.
	if fdot(dir_g, dir_site) + fdot(dir_site, dir_d) <= pi: # Si cette direction n'est pas dans l'espace formé par les deux demi-droites (bg, dit1) et (bd, dir2).
		dir_site = ortho(vecteur(p2, p1))

	if dir_site[0] <= 0: return 'g' # On renvoie la direction du breakpoint.
	return 'd'

def breakpoint_convergeant(bg, bd, p1, p2):
	'''Lorsque les breakpoints bg (breakpoint à gauche du sommet) et bd (breakpoint à droite du sommet) convergent, 
	cette fonction assure la bonne construction du diagramme de Voronoï. Elle renvoie le nouveau breakpoint associé aux sites 
	p1 et p2 (et construit sont jumeau, qui ne figurera pas dans l'arbre AVL).'''
	# Le sommet où il y a convergence : 
	x, y = bg.coords() # Ces coordonnées ne sont pas exactement celle du sommet du diagramme de Voronoï, mais elles sont très proches.
	s = Sommet(x, y)

	# Les deux nouvelles demi-arêtes que tracera le nouveau breakpoint issu de la fusion de b1 et de b2.
	arete_1, arete_2 = Arete(), Arete()
	arete_1.arete_soeur, arete_2.arete_soeur = arete_2, arete_1

	arete_1.sommet_but = s
	bg.arete.sommet_but, bd.arete.sommet_but = s, s # On fixe l'origine des arêtes tracées par les deux breakpoints qui convergent.

	bg.arete.arete_suivante, bd.arete.arete_suivante = bd.jumeau.arete, arete_2 # On fixe l'arête suivante pour le cas des breakpoints convergeant.
	bd.jumeau.arete.arete_precedente, arete_2.arete_precedente = bg.arete, bd.arete # On fixe l'arête précédente.
	bg.jumeau.arete.arete_precedente, arete_1.arete_suivante = arete_1, bg.jumeau.arete # Idem pour les arêtes manquantes.

	# Construction du nouveau BreakPoint et de son jumeau (il ne sera pas intégré dans l'arbre AVL).
	sens = direction_convergence(bg, bd, p1, p2)
	nouveau_b = BreakPoint(p1, p2, sens, arete_2)
	jumeau_b = BreakPoint(p1, p2, sens, arete_1)
	nouveau_b.jumeau, jumeau_b.jumeau = jumeau_b, nouveau_b

	# Gestion des faces : 
	if bg.jumeau.arete.face == None:
		f1 = Face(arete_1)
		arete_1.face, bg.jumeau.arete.face = f1, f1
	else:
		arete_1.face = bg.jumeau.arete.face

	if bd.arete.face == None:
		f2 = Face(arete_2)
		arete_2.face, bd.arete.face = f2, f2
	else:
		arete_2.face = bd.arete.face

	if bd.jumeau.arete.face == None:
		if bg.arete.face == None:
			f3 = Face(bg.arete)
			bg.arete.face, bd.jumeau.arete.face = f3, f3
		else:
			bd.jumeau.arete.face = bg.arete.face
	else:
		if bg.arete.face == None:
			bg.arete.face = bd.jumeau.arete.face
		elif bg.arete.face.id != bd.jumeau.arete.face.id:
			bg.arete.face.regrouper(bd.jumeau.arete.face) # Mise en commun des deux faces.

	return nouveau_b

def verifier_convergence(arc_g, arc_d, tas):
	'''Gestion de la convergence/divergence de breakpoints (on donne deux arcs : un arc gauche pour l'étude de la convergence à gauche, 
	idem pour l'arc droit (mais à droite)).'''
	# Triplet gauche, s'il y a : 
	g1 = arc_g.voisin_gauche
	if g1 != None:
		g2 = g1.voisin_gauche
		if g2 != None:
			ajouter_evenement_circulaire(g2, g1, arc_g, tas)

	# Triplet droit, s'il y a : 
	d1 = arc_d.voisin_droit
	if d1 != None:
		d2 = d1.voisin_droit
		if d2 != None:
			ajouter_evenement_circulaire(arc_d, d1, d2, tas)

def ajouter_evenement_circulaire(arc_g, arc_m, arc_d, tas):
	'''Ajouter, si nécessaire, un événement circulaire dans le tas.'''
	p1, p2, p3 = arc_g.coords(), arc_m.coords(), arc_d.coords() # Les sites (normalement distincts).
	if p1 != p3: # Dans le cas contraire, les deux breakpoints associés à ce triplet d'arcs divergent.
		b1, b2 = arc_m.breakpoint_gauche, arc_m.breakpoint_droit
		if intersecteDemiDroite(b1.coords(), b1.direction, b2.coords(), b2.direction):
			evt = EvtCercle(p1, p2, p3, arc_m)
			arc_m.evtcercle = evt # On marque l'arc avec ce nouvel événement circulaire.
			tas.ajouter(evt) # On ajoute cet événement au tas.

def retirer_evenement_circulaire(arc, tas):
	'''Retire l'événement circulaire associé à l'arc donné du tas.'''
	if arc.evtcercle != None:
		tas.extraire(arc.evtcercle.idx) # Retrait de l'événement circulaire (en général, c'est une 'fausse alerte').
		arc.evtcercle = None # On retire l'événement circulaire.

# Gestion des événements : 
def evenement_ponctuel(tas, arb, racine, evt):
	'''Prise en charge d'un événement ponctuel.'''
	nouvel_arc = Arc(evt.coords())
	if racine == None:
		racine = arb.ajouter_arc(racine, nouvel_arc)
	else:
		arc = arb.rechercher(racine, evt).etiquette
		retirer_evenement_circulaire(arc, tas) # Suppression de la "fausse alerte".

		racine = arb.ajouter_arc(racine, nouvel_arc) # Ajout de l'arc de parabole.
		verifier_convergence(nouvel_arc, nouvel_arc, tas) # Vérifier la convergence des nouveaux breakpoints (ajout d'événements circulaires si besoin).
	return racine

def evenement_circulaire(tas, arb, racine, evt):
	'''Prise en charge d'un événement circulaire.'''
	arc = evt.arc
	voisin_gauche, voisin_droit = arc.voisin_gauche, arc.voisin_droit
	# On supprime les potentiels événements circulaires qui auraient pu se produire avec les arcs voisins de l'arc concerné.
	retirer_evenement_circulaire(voisin_gauche, tas)
	retirer_evenement_circulaire(voisin_droit, tas)

	# Coordonnées du point au milieu du segment joignant les deux breakpoints (pourr retrouver l'arc à supprimer).
	x, y = (arc.breakpoint_gauche.coords()[0] + arc.breakpoint_droit.coords()[0]) / 2, evt.coords()[1]
	racine = arb.supprime_arc(racine, Arc((x, y))) # Suppression de l'arc.
	verifier_convergence(voisin_droit, voisin_gauche, tas) # Vérifier la convergence des breakpoints (ajout d'événements circulaires si besoin).
	return racine

# La fonction principale : 
def voronoi(P):
	'''On suppose les points deux-à-deux distincts.'''
	global ordonnee
	n = len(P)
	# Cas particuliers : 
	# if n == 0:
		# return Geometrie([])
	# elif n == 1:
		# f = Face(None)
		# a_g, a_d = chemin_aretes([(0, 0), (0, l - 1), (L - 1, l - 1), (L - 1, 0), (0, 0)], f, None)
		# a_g[0].arete_precedente, a_g[-1].arete_suivante = a_g[-1], a_g[0]
		# f.arete_associee = a_g[0]
		# return Geometrie([f])

	compTas = lambda p1, p2 : p1.coords()[::-1] < p2.coords()[::-1] # Comparaison lexicographique suivant la coordonnée y (balayage de haut en bas) puis x (si besoin).
	compArb = lambda p1, p2 : p1.coords() < p2.coords() # Comparaison lexicographique suivant la coordonnée x (ligne de front, de gauche à droite) puis y (si besoin).

	tas = TasMin(3*n - 5, [EvtPoint(p) for p in P], compTas, True)
	arb, racine = AVL(compArb), None

	# On balaye le plan du haut vers le bas, étant donné le repérage, on descend dans le sens des y croissants.
	while not tas.estVide():
		evt = tas.minimum()
		ordonnee = evt.y # On met à jour l'ordonnée de la droite de balayage.
		if isinstance(evt, EvtPoint):
			racine = evenement_ponctuel(tas, arb, racine, evt) # Événement ponctuel.
		else:
			ordonnee -= 1e-20 # Pour éviter les risques d'erreurs liés aux imprécisions numériques (valeur expérimentale).
			racine = evenement_circulaire(tas, arb, racine, evt) # Événement circulaire.

	# À l'issu, les breakpoints qui restent dans l'arbre AVL correspondent à des demi-droites, on procède en deux temps : 
	# -> d'abord on va clipper ces demi-droites avec les bords de la fenêtre et finaliser la construction des arêtes correspondantes (via un parcours en profondeur de l'arbre AVL), 
	# -> puis, on parcourt la géométrie (grâce à un parcours en profondeur à l'aide des faces et des arêtes) afin de :
	#   i) finaliser la construction des faces (en fermant les cellules non bornées avec les bords de la fenêtre), 
	#   ii) recenser toutes les faces.
	def traitement_demi_droites(noeud):
		b = noeud.etiquette
		if isinstance(b, BreakPoint):
			c, d = b.coords(), b.jumeau.arete.sommet_but.coords()
			if estDansRectangle(c) or ((not estDansRectangle(c)) and estDansRectangle(d)):
				p = clipper_demi_droite(d, b.direction)
				s = Sommet(p[0], p[1])
				b.arete.sommet_but = s

	arb.parcours_prefixe(racine, traitement_demi_droites)
	faces = []
	if ID_FACE == 0: # Ceci atteste que le diagramme de Voronoï est dégénéré et ne contient que des droites (i.e. tous les sites sont alignés).
		pass
	else: # Le diagramme de Voronoï est non dégénéré (il contient seulement des segments et des demi-droites commes arêtes).
		def traitement_faces(face):
			#clipper_face(face)
			faces.append(face)
			face.colorier()
		# Comme n >=2, le diagramme de Voronoï, dans ce cas, contient au moins une demi-droite donc, la racine de l'arbre AVL est un BreakPoint.
		racine.etiquette.arete.face.parcours_profondeur(traitement_faces)

	return Geometrie(faces)
########################################################################


########################################################################
# Algorithme pour associer à chaque point d'un premier ensemble (p1) les points les plus proches d'un second ensemble (p2) : 
# Approche naïve : 
# -> on construit un premier groupe en associant tous les points de p2 au premier point de p1,
# -> pour le second point de p2, pour chaque point de p1, on compare les distances (avec <) et, on ne conserve que les plus courtes,
# -> on réitère pour chaque point de p1.
#
# * problèmes : des points de p1 peuvent être non associés : on regarde le premier point de p2 le plus proche, on le classe dans le groupe de ce dernier.
# * complexité : temporelle O(p1 * p2), spatiale O(p1 * p2) : 
def calculer_distance(p1, p2):
	'''Calcul des distances entre deux ensembles de points.'''
	# Complexité : temporelle O(p1 * p2), spatiale O(p1 * p2).
	dist = np.zeros((len(p1), len(p2)))
	for (i, s) in enumerate(p1):
		xs, ys = s
		for (j, a) in enumerate(p2):
			xa, ya = a
			dist[i, j] = dist2((xs, ys), (xa, ya))
	return dist

def regrouper1(p1, p2):
	'''Regrouper chaque point de p2 au plus proche point de p1 (on suppose p1 et p2 non vides).'''
	dist = calculer_distance(p1, p2)
	L, Lmin = [0] * len(p2), [0] * len(p1)
	for i in range(len(p1)):
		k, d_min = 0, dist[i, 0]
		for j in range(len(p2)):
			if dist[i, j] < dist[L[j], j]:
				L[j] = i
			if dist[i, j] < d_min:
				k, d_min = j, dist[i, j]
		Lmin[i] = k

	# Construction des groupes : 
	grp = [[e] for e in p1]
	for (j, i) in enumerate(L):
		grp[i].append(p2[j])

	# Gestion des points de p1 seuls : 
	for (i, point) in enumerate(grp):
		if len(point) == 1:
			grp[L[Lmin[i]]].append(point[0])

	return [e for e in grp if len(e) > 1]
########################################################################


########################################################################
# Implémentation d'un arbre 2-d (statique - donc pas d'insertion/retrait) afin d'améliorer 
# la recherche du plus proche voisin dans le plan.
class Arbre2D:
	class Noeud:
		def __init__(self, etiquette, filsG = None, filsD = None):
			'''Nœud (nœud interne ou feuille).'''
			self.etiquette = etiquette
			self.filsG, self.filsD = filsG, filsD

		def estFeuille(self):
			'''Tester si un nœud est une feuille.'''
			return self.filsG == self.filsD == None

	def __init__(self, points):
		'''Arbre (binaire) 2-d.'''
		self.points = points

	def construire(self, m, M, axe):
		'''Construit récursivement un arbre 2-d.'''
		def comp_coordonnee(p1, p2):
			'''Fonction pour comparer deux points suivant leur coordonnée en x ou en y.'''
			if axe == 1: return p1 < p2
			return p1[::-1] < p2[::-1] # On renverse les couples pour les comparer.

		if m >= M:
			return None
		tri_fusion(self.points, m, M, comp_coordonnee) # Sélection du point médian : O(n log(n)) (O(n) est possible...).
		milieu = (m + M) // 2
		return self.Noeud(self.points[milieu], self.construire(m, milieu, 1 - axe), self.construire(milieu + 1, M, 1 - axe))

	def plusProcheVoisin(self, arb, point, axe):
		'''Recherche du plus proche voisin parmi les points de l'arbre.'''
		meilleur_point, meilleure_distance = None, float('inf')

		def mettre_a_jour(p1):
			'''Mettre à jour, si nécessaire, le meilleur point.'''
			nonlocal meilleure_distance, meilleur_point
			d = dist2(p1, point)
			if d < meilleure_distance:
				meilleur_point, meilleure_distance = p1, d

		def basculer(noeud, fils, direction):
			'''Tester s'il faut parcourir ("basculer" dans) le fils du nœud.'''
			if fils == None:
				return False
			d1, d2 = abs(point[direction] - noeud.etiquette[direction]), abs(point[1 - direction] - fils.etiquette[1 - direction])
			return (min(d1, d2) < meilleure_distance) or (d1**2 + d2**2 < meilleure_distance)

		def recherche(arbre, direction):
			# Cas de base (arbre vide ou feuille).
			if arbre == None:
				return None
			elif arbre.estFeuille(): # On met à jour le plus proche voisin trouvé puis, on remonte.
				mettre_a_jour(arbre.etiquette)
				return None
			# Descente dans l'arbre.
			if point[direction] < arbre.etiquette[direction]: # Descente à gauche.
				recherche(arbre.filsG, 1 - direction)
				mettre_a_jour(arbre.etiquette)
				if basculer(arbre, arbre.filsD, direction): # Pour basculer dans l'autre fils si besoin.
					recherche(arbre.filsD, 1 - direction)
			else: # Descente à droite.
				recherche(arbre.filsD, 1 - direction)
				mettre_a_jour(arbre.etiquette)
				if basculer(arbre, arbre.filsG, direction): # Pour basculer dans l'autre fils si besoin.
					recherche(arbre.filsG, 1 - direction)
		recherche(arb, axe)
		return meilleur_point

def regrouper2(pharmacies, supermarches):
	'''Regrouper supermarche à la pharmacie la plus proche (on suppose p1 et p2 non vides).'''
	grp = {p: [p] for p in pharmacies} # Groupe associé aux pharmacies.
	voisin_supermarche = {p: None for p in supermarches} # Pour chasue supermarché est mémorisé la pharmacie la plus proche.

	# Construction des deux arbres 2-D : 
	arb_p, arb_s = Arbre2D(pharmacies), Arbre2D(supermarches)
	racine_p, racine_s = arb_p.construire(0, len(pharmacies), 0), arb_s.construire(0, len(supermarches), 0)

	# On associe chaque supermarché à la pharmacie la plus proche.
	for p in supermarches:
		voisin = arb_p.plusProcheVoisin(racine_p, p, 0)
		grp[voisin].append(p)
		voisin_supermarche[p] = voisin

	# Pour les pharmacies dont leur groupe est réduit à elle-même, on les met dans 
	# le même groupe que celui du supermarché le plus proche.
	for (p, l) in grp.items():
		if len(l) == 1: # Pharmacie seule dans son groupe.
			voisin = arb_s.plusProcheVoisin(racine_s, p, 0)
			pharmacie_associee = voisin_supermarche[voisin]
			grp[pharmacie_associee].append(p)

	# On ne renvoie que les groupes contenant au moins deux magasins (i.e. au moins un 
	# supermarché et une pharmacie : 
	return [e for e in grp.values() if len(e) > 1]

# Implémentation des KDTree (très rapide (<= 2 sec pour <= 25000 points !), langage C ?)
from scipy.spatial import KDTree
def regrouper3(p1, p2):
	'''Regrouper les points de p2 au plus proche point de p1 (on suppose p1 et p2 non vides).'''
	grp1, grp2 = {p: [p] for p in p1}, {p: None for p in p2}  # Groupe associé aux pharmacies.
	arb1, arb2 = KDTree(p1), KDTree(p2)

	for p in p2:
		voisin = p1[arb1.query(p)[1]]
		grp1[voisin].append(p)
		grp2[p] = voisin

	for (p, l) in grp1.items():
		if len(l) == 1:
			grp1[grp2[p2[arb2.query(p)[1]]]].append(p)

	return [e for e in grp1.values() if len(e) > 1]
########################################################################


########################################################################
alimentaire = construirePoints('../Data/Alimentation.csv')
sante = construirePoints( '../Data/Sante.csv')

def decoupage():
	grp = regrouper2(sante, alimentaire)
	evn = list(map(enveloppe_convexe, grp)) # Calcul de la liste des enveloppes convexes (deux-à-deux disjointes).
	sites = list(map(isobarycentre, evn)) # Les sites.

	dessiner_enveloppe_convexe(evn) # Dessin des enveloppes convexes.
	dessinerPoints(alimentaire, 'red', 2)
	dessinerPoints(sante, 'green', 2)
	dessinerPoints(sites, 'blue', 3)

	vor =  voronoi(sites) # Construction du diagramme de Voronoï avec les isobarycentres des enveloppes convexes.



	# 'food' -> 0, 'health' -> 1
	#for _ in germes:
	#	Cellule(id_cell, [0.6, 0.1, 0, 0, 0], [0.01, 0.7, 0.01, 50*random(), 0.005, 0.1, 0.005], 0)
	#	id_cell += 1

	# Association cellules-régions : 
	#for i, r in enumerate(voronoi.point_region):
	#	d_cellules[i].poly = voronoi.regions[r]

	# Construction voisins : 
	#for g1, g2 in voronoi.ridge_points:
	#	d_cellules[g1].ajout_voisin(d_cellules[g2])

	return vor
########################################################################


########################################################################
# Construction du graphe routier.
def extraire():
	'''Extraire les nœuds et routes du fichier de donnée XML.'''
	fichier_xml = "../Data/CentreVilleMap.xml"
	arbre = ET.iterparse(fichier_xml)
	noeuds_full, noeuds, routes = {}, {}, [] # On ne garde que les nœuds "utiles".

	# Parcours du fichier XML.
	# Le fichier XML est structuré grosso modo en deux blocs : 
	# -> le premier contenant tous les nœuds, 
	# -> le second contient toutes les routes.
	for (evt, elm) in arbre:
		if elm.tag == 'node': # Gestion d'un nœud.
			attributs = elm.attrib
			noeuds_full[attributs['id']] = (float(attributs['lat']), float(attributs['lon']))
		elif elm.tag == 'way': # Gestion d'une route.
			if any([n.attrib['k'] == 'highway' and n.attrib['v'] not in ['path', 'cycleway', 'footway', 'steps', 'elevator'] for n in elm.findall('tag')]):
				lst = []
				for n in elm.findall('nd'):
					ref = n.attrib['ref']
					lst.append(ref)
					noeuds[ref] = noeuds_full[ref]
				routes.append(lst)
	# On renvoie les nœuds et les routes.
	return noeuds, routes

# Pour charger les composantes géographiques du centre-ville (noeuds et routes) ainsi que le graphe.
def charger_composantes():
	with open('routes.json', 'r') as fichier: routes = json.loads(fichier.read())
	with open('noeuds.json', 'r') as fichier: noeuds = json.loads(fichier.read())
	return noeuds, routes

def charger_graphe():
	with open('graphe.json', 'r') as fichier: graphe = json.loads(fichier.read())
	return graphe

def construireGraphe(noeuds, routes):
	'''Construit le graphe pondéré à partir des fichiers "nœuds.txt" et "routes.txt".'''
	# On adopte une représentation sous forme de dictionnaire d'adjacence : 
	# -> on conserve les identifiants des nœuds enregistrés.
	# -> les poids correspondent à la distance réelle entre les points.
	graphe = dict()

	# Construction du graphe : 
	for r in routes:
		for i in range(len(r) - 1):
			n1, n2 = r[i], r[i + 1]
			d = distance_reelle(noeuds[n1], noeuds[n2]) # Distance entre les deux points : le poids de l'arête.

			if graphe.get(n1) == None: graphe[n1] = {n2 : d}
			else: graphe[n1][n2] = d

			if graphe.get(n2) == None: graphe[n2] = {n1 : d}
			else: graphe[n2][n1] = d
	# On renvoie le graphe.
	return graphe

def dessinerGraphe(noeuds, routes, couleur_sommet = 'blue', couleur_arete = 'black', canvas = draw):
	'''Dessine le graphe des routes sur la carte.'''
	for r in routes:
		for i in range(len(r) - 1):
			n1, n2 = r[i], r[i + 1]
			p1, p2 = polaire2pixel(noeuds[n1]), polaire2pixel(noeuds[n2])

			canvas.line((p1, p2), fill = couleur_arete)
			dessinerPoints([p2], couleur_sommet, 1, canvas)

		dessinerPoints([polaire2pixel(noeuds[r[0]])], couleur_sommet, 1, canvas)
########################################################################


########################################################################
# Fonctions pour la manipulation de graphes.
def composantesConnexes(graphe):
	'''Renvoie le liste des composantes connexes du graphe donnée (sous forme de dictionnaire).
	La complexité de cette fonction est : temporelle O(a + s), spatiale O(a + s) pour s sommets et a arêtes.'''
	composantes_connexes, deja_vus = [], dict()

	def parcourt_profondeur(sommet):
		'''Parcours en profondeur du graphe depuis un sommet donné.'''
		composante, pile = dict(), [sommet]
		while len(pile) > 0:
			s = pile.pop() # On prend le sommet au dessus de la pile.
			if not deja_vus.get(s, False): # S'il n'a pas encore été visité.
				deja_vus[s] = True # On marque s comme sommet visité.
				composante[s] = graphe[s] # On met à jour la composante connexe en construction.
				pile.extend(graphe[s].keys()) # On met tous les voisins de s dans la pile.
		# On renvoie la composante connexe, elle est vide (i.e. égale à dict()) lorsque sommet à déjà été visité.
		return composante

	for sommet in graphe.keys(): # Pour chaque sommet du graphe.
		composante = parcourt_profondeur(sommet) # On extrait la composante connexe dont il fait parti.
		if composante != dict(): # Si le sommet en question n'a pas déjà été visité, on ajoute la composante connexe dans la liste.
			composantes_connexes.append(composante)
	return composantes_connexes # On renvoie la liste des composantes connexes.

def dijkstra(debut, graphe, fin = None):
	'''Mise en œuvre de l'algorithme de Dijkstra pour la recherche d'un plus court chemin
	du sommet 'debut' au sommet 'fin' dans le graphe fourni.
	Le graphe est un dictionnaire : {id nœud : {id voisin : distance, ...}}.
	Renvoie le dictionnaire des distances et des prédécesseurs ; une condition d'arrêt est 
	possible si le sommet 'fin' est précisé.
	Sa complexité est : temporelle O((a + s) * log(s)), spatiale O(s) avec s sommets et a arêtes.'''
	# Cette classe permet "d'accélérer" l'algorithme de Dijkstra en facilitant la mise à jour du tas binaire.
	class Sommet:
		def __init__(self, sommet):
			self.sommet = sommet # Identifiant du sommet.
			self.idx = 0 # Position dans le tas-min.

	distance, predecesseur, correspondance = dict(), dict(), dict() # Dictionnaire de correspondance (identifiant du sommet : objet Sommet).
	sommets = graphe.keys() # Les sommets du graphe.
	n = len(sommets) # Nombre de sommets du graphe.
	tas = TasMin(n, [], lambda s1, s2 : distance[s1.sommet] < distance[s2.sommet], True) # On autorise l'indexation sur les objets du tas pour faciliter leur manipulation.

	for s in sommets: # Initialisation des structures de données.
		distance[s], correspondance[s] = graphe[debut].get(s, float('inf')), Sommet(s)
		if s != debut:
			tas.ajouter(correspondance[s]) # La file de priorité.

	for e in range(n - 2):
		S = tas.minimum() # On retire du tas le sommet S dont on est sûr de connaître un plus court chemin de 'debut' à S.
		if S.sommet == fin: # Pour terminer dès que le sommet 'fin' a été rencontré.
			return distance, predecesseur

		for v in graphe[S.sommet].keys(): # On restreint la recherche aux voisins de S seulement.
			d = distance[S.sommet] + graphe[S.sommet][v]
			if d < distance[v]: # On regarde si la nouvelle distance du sommet 'debut' au sommet v est plus courte.
				distance[v], predecesseur[v] = d, S.sommet # On met à jour, pour le sommet v, ses données.
				tas.percole_haut(correspondance[v].idx) # Préserver l'invariant grâce à sa position dans le tas, mémorisée.
	# On renvoie le dictionnaire des distances et des prédécesseurs.
	return distance, predecesseur

def construireChemin(debut, fin, predecesseur, noeuds):
	'''Étant donné deux sommets debut et fin ainsi qu'un "dictionnaire des prédécesseurs", 
	renvoie la liste ordonnée des identifiants des sommets pour aller de "debut" à "fin".
	Attention : on fait ici l'hypothèse qu'un chemin entre "debut" et "fin" existe !'''
	liste, sommet = [], fin
	while sommet != None: # Tant qu'il existe des prédécesseurs.
		liste.append(polaire2pixel(noeuds[sommet])) # On ajoute le prédécesseur dans la liste.
		sommet = predecesseur.get(sommet, None) # On regarde s'il possède lui-aussi un prédécesseur ('None' sinon).
	# On renvoie le chemin ordonnée, du début à la fin.
	return [polaire2pixel(noeuds[debut])] + liste[::-1]
########################################################################


########################################################################
# Ajuster les arêtes du diagramme de Voronoï aux routes.
# Dictionnaire de correspondance entre les identidiants des sommets du diagramme de Voronoï et les identifiants des sommets du graphe (après ajustement des premiers).
sommets_graphe = dict()

def ajusterSommet(sommets, graphe, noeuds):
	'''Cette fonction altère les coordonnées des sommets du diagramme de Voronoï en 
	remplaçant chaque sommet par le sommet du graphe routier le plus proche.'''
	global sommets_graphe

	p = {polaire2pixel(noeuds[nom]) : nom for nom in graphe.keys()}
	l = [polaire2pixel(noeuds[nom]) for nom in graphe.keys()]
	arb = Arbre2D(list(p.keys()))
	racine = arb.construire(0, len(p), 0)

	for sommet in sommets:
		voisin = arb.plusProcheVoisin(racine, sommet.coords(), 0)
		sommet.x, sommet.y = voisin[0], voisin[1]
		sommets_graphe[sommet.id] = p[voisin]

def ajusterAretes(aretes, graphe, noeuds):
	'''Cette fonction altère les arêtes du diagramme de Voronoï afin de les 
	faire coïncider avec les routes.
	Attention : on fait ici l'hypothèse que les sommets du diagramme de Voronoï 
	coïncident avec des sommets du graphe.'''
	arete_ajustees = [False] * ID_ARETES # Mémoriser les arêtes déjà traitées.
	for i in range(ID_ARETES):
		arete = aretes[i]
		if not arete_ajustees[arete.id]:
			if arete.sommet_but != None and arete.arete_soeur.sommet_but != None:
				# Marquer l'arête et sa soeur comme ajustées.
				arete_ajustees[arete.id], arete_ajustees[arete.arete_soeur.id] = True, True

				n1, n2 = sommets_graphe[arete.sommet_but.id], sommets_graphe[arete.arete_soeur.sommet_but.id]
				distance, predecesseur = dijkstra(n1, graphe, n2)
				if distance[n2] <= 1.5 * distance_reelle(noeuds[n1], noeuds[n2]):
					chemin = construireChemin(n1, n2, predecesseur, noeuds)
					chemin_g, chemin_d = chemin_aretes(chemin, arete.arete_soeur.face, arete.face)

					# # On recole les nouveaux chemins aux arêtes déjà présentes.
					chemin_d[-1].arete_precedente, arete.arete_precedente.arete_suivante = arete.arete_precedente, chemin_d[-1]
					chemin_d[0].arete_suivante, arete.arete_suivante.arete_precedente =  arete.arete_suivante, chemin_d[0]

					chemin_g[0].arete_precedente, arete.arete_soeur.arete_precedente.arete_suivante = arete.arete_soeur.arete_precedente, chemin_g[0]
					chemin_g[-1].arete_suivante, arete.arete_soeur.arete_suivante.arete_precedente =  arete.arete_soeur.arete_suivante, chemin_g[-1]
				else:
					arete.couleur, arete.arete_soeur.couleur = 'red', 'red'

def ajusterVoronoi(graphe, noeuds):
	'''Ajuster le diagramme de Voronoï au graphe routier.'''
	ajusterSommet(liste_sommets, graphe, noeuds) # Ajuster les sommets du diagramme de Voronoï.
	ajusterAretes(liste_aretes, graphe, noeuds) # Ajuster les arêtes du diagramme de Voronoï.
########################################################################

# Extraction et mémorisation des routes et des noeuds : 
noeuds, routes = charger_composantes() # extraire()
# with open('noeuds.json', 'w+') as fichier: fichier.write(json.dumps(noeuds))
# with open('routes.json', 'w+') as fichier: fichier.write(json.dumps(routes))

# Construction, extraction de la plus grosse composante connexe et mémorisation du graphe des routes : 
graphe = charger_graphe() # construireGraphe(noeuds, routes)
# liste_composantes = composantesConnexes(graphe)
# tri_fusion(liste_composantes, 0, None, lambda x, y : len(x.keys()) < len(y.keys()))
# graphe = liste_composantes[-1]
# with open('graphe.json', 'w+') as fichier: fichier.write(json.dumps(graphe))

# dessinerGraphe()

# Construction du diagramme de Voronoï et ajustement aux routes : 
voronoi = decoupage()
#ajusterVoronoi(graphe, noeuds) -> dressiner() reste bloqué dans une boucle infinie si in ajuste les arêtes...
voronoi.dessiner()
carte.save('decoupage.png')
#######################################################################
