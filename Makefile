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
# Connexion à Hugging Face (préparation)
# ===========================
hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub"

# ===========================
# Déploiement sur Hugging Face Space (sans CLI, via API Python)
# ===========================
push-hub:
	python - << 'EOF'
	import os
	from huggingface_hub import upload_folder

	repo_id = "asmaegr50/Heart_Disease_Classification"
	token = os.environ["HF"]

	# Upload du dossier App (app Gradio)
	upload_folder(
	    repo_id=repo_id,
	    repo_type="space",
	    folder_path="App",
	    path_in_repo=".",
	    token=token,
	    commit_message="Sync App files",
	)

	# Upload du modèle
	upload_folder(
	    repo_id=repo_id,
	    repo_type="space",
	    folder_path="Model",
	    path_in_repo="Model",
	    token=token,
	    commit_message="Sync Model",
	)

	# Upload des résultats (metrics / plots)
	upload_folder(
	    repo_id=repo_id,
	    repo_type="space",
	    folder_path="Results",
	    path_in_repo="Metrics",
	    token=token,
	    commit_message="Sync Metrics",
	)
	EOF

# ===========================
# Pipeline complet de déploiement
# ===========================
deploy: hf-login push-hub

