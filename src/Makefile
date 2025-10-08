# ======================================================
# 🧩 Makefile - Proyecto MLOps: Billetes
# ======================================================

IMAGE_NAME = billetes-mlops
IMAGE_TAG = latest

GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m

.PHONY: help build train mlflow predict dev clean stop logs

help:
	@echo "$(GREEN)Comandos disponibles:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

build: ## Construir la imagen Docker
	@echo "$(GREEN)Construyendo imagen Docker...$(NC)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

train: build ## Ejecutar entrenamiento (usa MLflow)
	@echo "$(GREEN)Iniciando entrenamiento...$(NC)"
	docker-compose up train

mlflow: ## Iniciar servidor MLflow
	@echo "$(GREEN)Iniciando MLflow Server en http://localhost:5001$(NC)"
	docker-compose up -d mlflow-server

predict: build ## Ejecutar servicio de predicciones batch
	@echo "$(GREEN)Iniciando batch de predicciones...$(NC)"
	docker-compose up predict

dev: ## Entrar al contenedor en modo desarrollo
	@echo "$(YELLOW)Entrando al contenedor...$(NC)"
	docker-compose run --rm train /bin/bash

stop: ## Detener todos los contenedores
	docker-compose down

clean: stop ## Limpiar recursos y caché
	@echo "$(RED)Limpiando contenedores e imágenes...$(NC)"
	docker system prune -af

logs: ## Ver logs en tiempo real
	docker-compose logs -f
