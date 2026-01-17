# =============================================================================
# PROJET M2 - MAKEFILE POUR PARTIE 1 (C++)
# =============================================================================

# Compilateur et options
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

# Fichiers sources et exécutables
SRC = seirs_cpp_rk4.cpp
TARGET = seirs_cpp_rk4

# =============================================================================
# RÈGLES DE COMPILATION
# =============================================================================

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "🔨 Compilation de $(SRC)..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)
	@echo "✅ Compilation réussie : $(TARGET)"

# Exécution
run: $(TARGET)
	@echo "▶️  Exécution de $(TARGET)..."
	./$(TARGET)

# Compilation et exécution
build-run: all run

# Nettoyage
clean:
	@echo "🧹 Nettoyage..."
	rm -f $(TARGET) *.o *.csv
	@echo "✅ Nettoyage terminé"

# Aide
help:
	@echo "==================================================="
	@echo "MAKEFILE PROJET M2 - PARTIE 1"
	@echo "==================================================="
	@echo ""
	@echo "Commandes disponibles :"
	@echo "  make          : Compiler le programme C++"
	@echo "  make run      : Exécuter le programme"
	@echo "  make build-run: Compiler et exécuter"
	@echo "  make clean    : Nettoyer les fichiers générés"
	@echo "  make help     : Afficher cette aide"
	@echo ""

.PHONY: all run build-run clean help