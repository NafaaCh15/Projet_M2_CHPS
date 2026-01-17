# =============================================================================
# PROJET M2 - MAKEFILE POUR PARTIE 2 (SMA C/C++)
# =============================================================================

# Compilateurs et options
CC = gcc
CXX = g++
CFLAGS = -std=c99 -O3 -Wall -Wextra -march=native
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

# Fichiers sources et exécutables
SRC_C = SMA_C_SEIRS.c
SRC_CPP = SMA_CPP_SEIRS.cpp
TARGET_C = sma_c
TARGET_CPP = sma_cpp

# =============================================================================
# RÈGLES DE COMPILATION
# =============================================================================

.PHONY: all clean help c cpp python run-c run-cpp run-python run-all

all: c cpp
	@echo ""
	@echo "✅ Compilation terminée avec succès"
	@echo ""
	@echo "Exécuter les simulations :"
	@echo "  make run-c      : Lancer SMA C (30 réplications)"
	@echo "  make run-cpp    : Lancer SMA C++ (30 réplications)"
	@echo "  make run-python : Lancer SMA Python (30 réplications)"
	@echo "  make run-all    : Lancer TOUS les SMA (90 fichiers CSV)"

# Compilation C
c: $(TARGET_C)

$(TARGET_C): $(SRC_C)
	@echo "🔨 Compilation C..."
	$(CC) $(CFLAGS) -o $(TARGET_C) $(SRC_C) $(LDFLAGS)
	@echo "✅ $(TARGET_C) compilé"

# Compilation C++
cpp: $(TARGET_CPP)

$(TARGET_CPP): $(SRC_CPP)
	@echo "🔨 Compilation C++..."
	$(CXX) $(CXXFLAGS) -o $(TARGET_CPP) $(SRC_CPP)
	@echo "✅ $(TARGET_CPP) compilé"

# Exécution C
run-c: $(TARGET_C)
	@echo "▶️  Lancement SMA C (30 réplications)..."
	@echo ""
	./$(TARGET_C)

# Exécution C++
run-cpp: $(TARGET_CPP)
	@echo "▶️  Lancement SMA C++ (30 réplications)..."
	@echo ""
	./$(TARGET_CPP)

# Exécution Python
run-python:
	@echo "▶️  Lancement SMA Python (30 réplications)..."
	@echo ""
	python3 SMA_Python_SEIRS.py

# Exécution de tous les SMA (C, C++, Python)
run-all: all
	@echo "🚀 Lancement de TOUTES les simulations (90 fichiers CSV)"
	@echo ""
	@echo "==================== SMA C ===================="
	./$(TARGET_C)
	@echo ""
	@echo "==================== SMA C++ ===================="
	./$(TARGET_CPP)
	@echo ""
	@echo "==================== SMA PYTHON ===================="
	python3 SMA_Python_SEIRS.py
	@echo ""
	@echo "✅ TOUTES LES SIMULATIONS TERMINÉES"
	@echo ""
	@echo "Fichiers générés :"
	@echo "  - results_c/       : 30 fichiers CSV (C)"
	@echo "  - results_cpp/     : 30 fichiers CSV (C++)"
	@echo "  - results_python/  : 30 fichiers CSV (Python)"
	@echo ""
	@echo "Prochaine étape : Analyse statistique (Jupyter Notebook)"

# Nettoyage
clean:
	@echo "🧹 Nettoyage..."
	rm -f $(TARGET_C) $(TARGET_CPP) *.o
	rm -rf results_c/ results_cpp/ results_python/
	@echo "✅ Nettoyage terminé"

# Nettoyage partiel (garde les résultats)
clean-bin:
	@echo "🧹 Nettoyage des exécutables..."
	rm -f $(TARGET_C) $(TARGET_CPP) *.o
	@echo "✅ Exécutables supprimés"

# Aide
help:
	@echo "==================================================="
	@echo "MAKEFILE PROJET M2 - PARTIE 2 (SMA)"
	@echo "==================================================="
	@echo ""
	@echo "COMPILATION :"
	@echo "  make           : Compiler C et C++"
	@echo "  make c         : Compiler uniquement C"
	@echo "  make cpp       : Compiler uniquement C++"
	@echo ""
	@echo "EXÉCUTION :"
	@echo "  make run-c     : Lancer SMA C (30 réplications)"
	@echo "  make run-cpp   : Lancer SMA C++ (30 réplications)"
	@echo "  make run-python: Lancer SMA Python (30 réplications)"
	@echo "  make run-all   : Lancer TOUS (C + C++ + Python)"
	@echo ""
	@echo "NETTOYAGE :"
	@echo "  make clean     : Supprimer tout (binaires + résultats)"
	@echo "  make clean-bin : Supprimer uniquement les binaires"
	@echo ""
	@echo "AIDE :"
	@echo "  make help      : Afficher cette aide"
	@echo ""
	@echo "TEMPS ESTIMÉ (30 réplications par langage) :"
	@echo "  - Python : ~30-60 minutes"
	@echo "  - C      : ~5-10 minutes"
	@echo "  - C++    : ~5-10 minutes"
	@echo "  - TOTAL  : ~40-80 minutes pour 90 fichiers CSV"
	@echo ""

# Stats rapides sur les fichiers générés
stats:
	@echo "📊 Statistiques des fichiers générés :"
	@echo ""
	@echo "C :"
	@ls -lh results_c/*.csv 2>/dev/null | wc -l || echo "  Aucun fichier"
	@echo ""
	@echo "C++ :"
	@ls -lh results_cpp/*.csv 2>/dev/null | wc -l || echo "  Aucun fichier"
	@echo ""
	@echo "Python :"
	@ls -lh results_python/*.csv 2>/dev/null | wc -l || echo "  Aucun fichier"
	@echo ""
	@echo "TOTAL :"
	@find results_* -name "*.csv" 2>/dev/null | wc -l || echo "0"