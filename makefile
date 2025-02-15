PYTHON = python3
PIP = pip
VENV_DIR = venv
MAIN_SCRIPT = main.py

# Installation des dépendances
install:
	$(PIP) install -r requirements.txt

# Création d'un environnement virtuel et installation des dépendances
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r requirements.txt

format:
	$(VENV_DIR)/bin/black .

lint:
	$(VENV_DIR)/bin/flake8 .

security:
	$(VENV_DIR)/bin/bandit -r .

# Préparation des données
prepare:
	$(PYTHON) $(MAIN_SCRIPT) --prepare

# Exécution de toutes les étapes (préparation, entraînement, évaluation, amélioration)
run:
	$(PYTHON) $(MAIN_SCRIPT) --prepare --train --evaluate --improve

# Exécuter uniquement l'entraînement du modèle
train:
	$(PYTHON) $(MAIN_SCRIPT) --train

# Évaluation du modèle
evaluate:
	$(PYTHON) $(MAIN_SCRIPT) --evaluate

# Amélioration du modèle
improve:
	$(PYTHON) $(MAIN_SCRIPT) --improve

# Sauvegarde du modèle entraîné
save:
	$(PYTHON) $(MAIN_SCRIPT) --save

# Chargement du modèle sauvegardé
load:
	$(PYTHON) $(MAIN_SCRIPT) --load

# Aide : affiche les commandes disponibles
help:
	@echo "Commandes disponibles dans ce Makefile :"
	@echo "  install   - Installer les dépendances"
	@echo "  venv      - Créer un environnement virtuel et installer les dépendances"
	@echo "  prepare   - Préparer les données et les sauvegarder"
	@echo "  run       - Exécuter toutes les étapes (prepare, train, evaluate, improve, save, load)"
	@echo "  train     - Entraîner le modèle"
	@echo "  evaluate  - Évaluer le modèle"
	@echo "  improve   - Améliorer le modèle"
	@echo "  save      - Sauvegarder le modèle"
	@echo "  load      - Charger le modèle"
	@echo "  lint      - Vérifier la qualité du code"
	@echo "  format    - Formatter le code"
	@echo "  security  - Vérifier les vulnérabilités"
	@echo "  clean     - Supprimer les fichiers temporaires et modèles"
	@echo "  ci        - Exécuter toutes les vérifications CI"

