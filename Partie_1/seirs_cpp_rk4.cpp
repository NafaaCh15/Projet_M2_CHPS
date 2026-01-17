#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <array>

// =============================================================================
// PARAMÈTRES GLOBAUX DU MODÈLE SEIRS
// =============================================================================

// Paramètres épidémiologiques (CONSTANTS)
const double BETA  = 0.5;           // Taux d'infection
const double SIGMA = 1.0 / 3.0;     // Taux inverse incubation (≈ 0.3333)
const double GAMMA = 1.0 / 7.0;     // Taux inverse infectieux (≈ 0.1429)
const double RHO   = 1.0 / 365.0;   // Taux inverse immunité (≈ 0.00274)

// Conditions initiales
const double S0 = 0.999;            // 99.9% susceptibles
const double E0 = 0.0;              // 0% exposés
const double I0 = 0.001;            // 0.1% infectés
const double R0 = 0.0;              // 0% remis

// Domaine d'intégration
const double T_INITIAL = 0.0;       // Temps initial (jours)
const double T_FINAL   = 730.0;     // Temps final (2 ans = 730 jours)

// =============================================================================
// STRUCTURE POUR STOCKER L'ÉTAT DU SYSTÈME
// =============================================================================

struct State {
    double S;  // Susceptibles
    double E;  // Exposés
    double I;  // Infectés
    double R;  // Remis
    
    // Constructeur
    State(double s = 0.0, double e = 0.0, double i = 0.0, double r = 0.0)
        : S(s), E(e), I(i), R(r) {}
    
    // Opérateur d'addition pour faciliter les calculs RK4
    State operator+(const State& other) const {
        return State(S + other.S, E + other.E, I + other.I, R + other.R);
    }
    
    // Multiplication par un scalaire
    State operator*(double scalar) const {
        return State(S * scalar, E * scalar, I * scalar, R * scalar);
    }
    
    // Division par un scalaire
    State operator/(double scalar) const {
        return State(S / scalar, E / scalar, I / scalar, R / scalar);
    }
};

// =============================================================================
// SYSTÈME D'ÉQUATIONS DIFFÉRENTIELLES SEIRS
// =============================================================================

/**
 * @brief Calcule les dérivées du système SEIRS
 * 
 * @param t Temps (non utilisé, système autonome)
 * @param y État actuel [S, E, I, R]
 * @return State Dérivées [dS/dt, dE/dt, dI/dt, dR/dt]
 */
State seirs_ode(double t, const State& y) {
    State dydt;
    
    dydt.S = -BETA * y.S * y.I + RHO * y.R;
    dydt.E = BETA * y.S * y.I - SIGMA * y.E;
    dydt.I = SIGMA * y.E - GAMMA * y.I;
    dydt.R = GAMMA * y.I - RHO * y.R;
    
    return dydt;
}

// =============================================================================
// MÉTHODE RUNGE-KUTTA D'ORDRE 4 (RK4)
// =============================================================================

/**
 * @brief Résolution du système ODE par méthode RK4
 * 
 * Schéma :
 *   k1 = dt * f(t, y)
 *   k2 = dt * f(t + dt/2, y + k1/2)
 *   k3 = dt * f(t + dt/2, y + k2/2)
 *   k4 = dt * f(t + dt, y + k3)
 *   y(t+dt) = y(t) + (k1 + 2*k2 + 2*k3 + k4) / 6
 * 
 * @param y0 Conditions initiales
 * @param t_initial Temps initial
 * @param t_final Temps final
 * @param dt Pas de temps
 * @param t_array Vecteur temps (sortie)
 * @param y_array Vecteur solutions (sortie)
 */
void rk4_method(const State& y0, double t_initial, double t_final, double dt,
                std::vector<double>& t_array, std::vector<State>& y_array) {
    
    // Calcul du nombre de pas
    int n_steps = static_cast<int>((t_final - t_initial) / dt) + 1;
    
    // Réservation de mémoire
    t_array.reserve(n_steps);
    y_array.reserve(n_steps);
    
    // Initialisation
    double t = t_initial;
    State y = y0;
    
    t_array.push_back(t);
    y_array.push_back(y);
    
    // Boucle d'intégration RK4
    for (int i = 1; i < n_steps; ++i) {
        // Calcul des 4 étapes de Runge-Kutta
        State k1 = seirs_ode(t, y) * dt;
        State k2 = seirs_ode(t + dt / 2.0, y + k1 / 2.0) * dt;
        State k3 = seirs_ode(t + dt / 2.0, y + k2 / 2.0) * dt;
        State k4 = seirs_ode(t + dt, y + k3) * dt;
        
        // Mise à jour
        y = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
        t = t_initial + i * dt;
        
        t_array.push_back(t);
        y_array.push_back(y);
    }
}

// =============================================================================
// EXPORT CSV
// =============================================================================

/**
 * @brief Exporte les résultats dans un fichier CSV
 * 
 * Format : temps,S,E,I,R
 * 
 * @param filename Nom du fichier
 * @param t_array Vecteur temps
 * @param y_array Vecteur solutions
 */
void export_to_csv(const std::string& filename, 
                   const std::vector<double>& t_array,
                   const std::vector<State>& y_array) {
    
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << std::endl;
        return;
    }
    
    // En-tête
    file << "temps,S,E,I,R\n";
    
    // Données (haute précision)
    file << std::fixed << std::setprecision(10);
    
    for (size_t i = 0; i < t_array.size(); ++i) {
        file << t_array[i] << ","
             << y_array[i].S << ","
             << y_array[i].E << ","
             << y_array[i].I << ","
             << y_array[i].R << "\n";
    }
    
    file.close();
    std::cout << "✓ Données exportées : " << filename << std::endl;
}

// =============================================================================
// STATISTIQUES DE BASE
// =============================================================================

/**
 * @brief Affiche des statistiques de base sur la simulation
 */
void print_statistics(const std::vector<double>& t_array,
                      const std::vector<State>& y_array) {
    
    // Trouver le pic d'infection
    double max_infected = 0.0;
    double time_max_infected = 0.0;
    
    for (size_t i = 0; i < y_array.size(); ++i) {
        if (y_array[i].I > max_infected) {
            max_infected = y_array[i].I;
            time_max_infected = t_array[i];
        }
    }
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n--- STATISTIQUES DE LA SIMULATION ---" << std::endl;
    std::cout << "Pic d'infection : I_max = " << max_infected 
              << " à t = " << time_max_infected << " jours" << std::endl;
    
    // État final
    const State& final_state = y_array.back();
    std::cout << "État final (t = " << T_FINAL << " jours) :" << std::endl;
    std::cout << "  S = " << final_state.S << std::endl;
    std::cout << "  E = " << final_state.E << std::endl;
    std::cout << "  I = " << final_state.I << std::endl;
    std::cout << "  R = " << final_state.R << std::endl;
    std::cout << "  Somme S+E+I+R = " 
              << (final_state.S + final_state.E + final_state.I + final_state.R) 
              << std::endl;
}

// =============================================================================
// PROGRAMME PRINCIPAL
// =============================================================================

int main() {
    std::cout << "=============================================================================" << std::endl;
    std::cout << "PROJET M2 - PARTIE 1 : MODÈLE SEIRS - RÉSOLUTION NUMÉRIQUE (C++)" << std::endl;
    std::cout << "=============================================================================" << std::endl;
    std::cout << std::endl;
    
    // Conditions initiales
    State y0(S0, E0, I0, R0);
    
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Conditions initiales : S=" << S0 << ", E=" << E0 
              << ", I=" << I0 << ", R=" << R0 << std::endl;
    std::cout << "Vérification : S+E+I+R = " << (S0 + E0 + I0 + R0) << " ✓" << std::endl;
    std::cout << std::endl;
    
    // Paramètres de simulation
    double dt = 0.1;  // Pas de temps (jours)
    
    std::cout << "Pas de temps dt = " << dt << " jours" << std::endl;
    std::cout << "Domaine : [" << T_INITIAL << ", " << T_FINAL << "] jours" << std::endl;
    std::cout << std::endl;
    
    // Vecteurs de stockage
    std::vector<double> t_array;
    std::vector<State> y_array;
    
    // -------------------------------------------------------------------------
    // RÉSOLUTION PAR RK4
    // -------------------------------------------------------------------------
    std::cout << "--- MÉTHODE RUNGE-KUTTA 4 ---" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    rk4_method(y0, T_INITIAL, T_FINAL, dt, t_array, y_array);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "✓ Résolution terminée en " 
              << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "  Nombre de pas de temps : " << t_array.size() << std::endl;
    std::cout << std::endl;
    
    // -------------------------------------------------------------------------
    // EXPORT CSV
    // -------------------------------------------------------------------------
    export_to_csv("seirs_cpp_rk4.csv", t_array, y_array);
    
    // -------------------------------------------------------------------------
    // STATISTIQUES
    // -------------------------------------------------------------------------
    print_statistics(t_array, y_array);
    
    std::cout << std::endl;
    std::cout << "=============================================================================" << std::endl;
    std::cout << "✓ SIMULATION C++ TERMINÉE AVEC SUCCÈS" << std::endl;
    std::cout << "=============================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Fichier généré : seirs_cpp_rk4.csv" << std::endl;
    std::cout << std::endl;
    std::cout << "Prochaine étape : Comparer avec les résultats Python (Jupyter Notebook)" << std::endl;
    
    return 0;
}