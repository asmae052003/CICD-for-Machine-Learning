# ===========================
# Installation des dépendances
# ===========================
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt


# ===========================
# Formatage du code avec Black
# ===========================
format:
	python -m black *.py


# ===========================
# Entraînement du modèle
# ===========================
train:
	python train.py


# ===========================
# Évaluation + génération du rapport CML
# ===========================
eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md

	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md

	cml comment create report.md


# ===========================
# Mise à jour de la branche "update"
# ===========================
update-branch:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git diff --quiet || git commit -am "Update with new results"
	git push --force origin HEAD:update


# ===========================
# Connexion à Hugging Face
# ===========================
hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token "$(HF)" --add-to-git-credential


# ===========================
# Déploiement sur Hugging Face Space
# ===========================
push-hub:
	# Remplacer par ton identifiant HF réel
	huggingface-cli upload your-username/Heart-Classification ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload your-username/Heart-Classification ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload your-username/Heart-Classification ./Results /Metrics --repo-type=space --commit-message="Sync Metrics"


# ===========================
# Pipeline complet de déploiement
# ===========================
deploy: hf-login push-hub
