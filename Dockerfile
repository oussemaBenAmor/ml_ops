# Utiliser une image Python officielle comme base
FROM python:3

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier tous les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer les ports pour Flask (5000) et MLflow (5001)
EXPOSE 5000 5001

# Définir la commande de démarrage pour exécuter Flask et MLflow en parallèle
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5001 & exec python app.py"]

