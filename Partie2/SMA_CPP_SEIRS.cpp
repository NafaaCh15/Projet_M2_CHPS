/*
=============================================================================
PROJET M2 - PARTIE 2 : MODÈLE MULTI-AGENT SEIRS (C++ OPTIMISÉ)
=============================================================================
Optimisations principales :
- Grille d’occupation des infectés (voisinage en O(1) au lieu de O(N))
- Distributions aléatoires créées une seule fois
- Réserves de mémoire
- Prêt pour -O3 -march=native -flto
=============================================================================
COMPILATION :
g++ -std=c++17 -O3 -march=native -flto -DNDEBUG -Wall -o sma_cpp_opt SMA_CPP_SEIRS_OPT.cpp

EXÉCUTION :
./sma_cpp_opt
=============================================================================
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <numeric>
#include <sys/stat.h>

// =============================================================================
// PARAMÈTRES GLOBAUX
// =============================================================================

constexpr int N_AGENTS = 20000;
constexpr int N_INITIAL_INFECTED = 20;
constexpr int GRID_SIZE = 300;
constexpr int N_ITERATIONS = 730;
constexpr int N_REPLICATIONS = 30;

constexpr double MEAN_EXPOSED_DURATION = 3.0;
constexpr double MEAN_INFECTED_DURATION = 7.0;
constexpr double MEAN_RECOVERED_DURATION = 365.0;
constexpr double INFECTION_FORCE = 0.5;

const std::string OUTPUT_DIR = "results_cpp";

// =============================================================================
// ÉTATS
// =============================================================================

enum class Status
{
    S, // Susceptible
    E, // Exposé
    I, // Infecté
    R  // Remis
};

// =============================================================================
// CLASSE AGENT
// =============================================================================

class Agent
{
public:
    int id;
    Status status;
    int x, y;
    int cell_index; // index linéaire dans la grille
    int time_in_status;
    double dE, dI, dR;

    Agent(int id_, Status status_, int x_, int y_,
          double dE_, double dI_, double dR_, int cell_idx)
        : id(id_), status(status_), x(x_), y(y_),
          cell_index(cell_idx), time_in_status(0),
          dE(dE_), dI(dI_), dR(dR_) {}

    // Renvoie le nouvel index de cellule après déplacement
    int move_random(std::mt19937 &rng, int grid_size,
                    std::uniform_int_distribution<int> &pos_dist)
    {
        x = pos_dist(rng);
        y = pos_dist(rng);
        return y * grid_size + x;
    }

    void update_status()
    {
        if (status == Status::E && time_in_status >= static_cast<int>(dE))
        {
            status = Status::I;
            time_in_status = 0;
        }
        else if (status == Status::I && time_in_status >= static_cast<int>(dI))
        {
            status = Status::R;
            time_in_status = 0;
        }
        else if (status == Status::R && time_in_status >= static_cast<int>(dR))
        {
            status = Status::S;
            time_in_status = 0;
        }
    }
};

// =============================================================================
// CLASSE SIMULATION OPTIMISÉE
// =============================================================================

class SEIRSSimulation
{
private:
    std::vector<Agent> agents;
    std::mt19937 rng;
    unsigned int seed;
    int grid_size;

    // Grille d’occupation des infectés : taille GRID_SIZE * GRID_SIZE
    std::vector<int> infected_grid;

    // Historique
    std::vector<int> history_S;
    std::vector<int> history_E;
    std::vector<int> history_I;
    std::vector<int> history_R;

    // Distributions réutilisées
    std::uniform_real_distribution<double> uni01;
    std::uniform_int_distribution<int> pos_dist;

    // -------------------------------------------------------------------------
    // Tirage exponentiel
    // -------------------------------------------------------------------------
    double neg_exp(double mean)
    {
        double u = uni01(rng);
        if (u == 0.0)
            u = 1e-10;
        return -mean * std::log(1.0 - u);
    }

    // -------------------------------------------------------------------------
    // Initialisation
    // -------------------------------------------------------------------------
    void init_agents()
    {
        std::cout << "  Initialisation des " << N_AGENTS
                  << " agents (seed=" << seed << ")..." << std::endl;

        for (int i = 0; i < N_AGENTS; ++i)
        {
            Status status = (i < N_INITIAL_INFECTED) ? Status::I : Status::S;
            int x = pos_dist(rng);
            int y = pos_dist(rng);
            int cell_idx = y * grid_size + x;

            double dE = neg_exp(MEAN_EXPOSED_DURATION);
            double dI = neg_exp(MEAN_INFECTED_DURATION);
            double dR = neg_exp(MEAN_RECOVERED_DURATION);

            agents.emplace_back(i, status, x, y, dE, dI, dR, cell_idx);

            if (status == Status::I)
            {
                ++infected_grid[cell_idx];
            }
        }

        std::cout << "    ✓ " << N_INITIAL_INFECTED << " infectés initiaux\n";
        std::cout << "    ✓ " << (N_AGENTS - N_INITIAL_INFECTED)
                  << " susceptibles initiaux\n";
    }

    // -------------------------------------------------------------------------
    // Comptage infectés dans voisinage via grille
    // -------------------------------------------------------------------------
    int count_infected_in_neighborhood(int x, int y) const
    {
        int count = 0;

        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = (x + dx + grid_size) % grid_size;
            for (int dy = -1; dy <= 1; ++dy)
            {
                int ny = (y + dy + grid_size) % grid_size;
                int idx = ny * grid_size + nx;
                count += infected_grid[idx];
            }
        }
        return count;
    }

    // -------------------------------------------------------------------------
    // Infection
    // -------------------------------------------------------------------------
    void try_infection(Agent &agent)
    {
        if (agent.status != Status::S)
            return;

        int Ni = count_infected_in_neighborhood(agent.x, agent.y);

        if (Ni > 0)
        {
            double p = 1.0 - std::exp(-INFECTION_FORCE * Ni);
            double u = uni01(rng);
            if (u < p)
            {
                agent.status = Status::E;
                agent.time_in_status = 0;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Enregistrement stats
    // -------------------------------------------------------------------------
    void record_statistics()
    {
        int counts[4] = {0, 0, 0, 0};

        for (const auto &agent : agents)
        {
            ++counts[static_cast<int>(agent.status)];
        }

        history_S.push_back(counts[0]);
        history_E.push_back(counts[1]);
        history_I.push_back(counts[2]);
        history_R.push_back(counts[3]);
    }

    // -------------------------------------------------------------------------
    // Un pas de simulation
    // -------------------------------------------------------------------------
    void simulation_step()
    {
        // Ordre aléatoire
        std::vector<int> indices(agents.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int idx : indices)
        {
            Agent &agent = agents[idx];

            // Si actuellement infecté, retirer temporairement de la grille
            if (agent.status == Status::I)
            {
                --infected_grid[agent.cell_index];
            }

            // Déplacement : retourne la nouvelle cellule
            int new_cell = agent.move_random(rng, grid_size, pos_dist);
            agent.cell_index = new_cell;

            // Infection éventuelle
            try_infection(agent);

            // Mise à jour d'état
            Status old_status = agent.status;
            agent.update_status();

            // Si nouvel état infecté → ajouter dans la grille
            if (agent.status == Status::I)
            {
                ++infected_grid[agent.cell_index];
            }

            // Temps dans l'état
            ++agent.time_in_status;

            // Si l'agent était infecté mais ne l’est plus,
            // il a déjà été retiré avant le déplacement.
        }
    }

public:
    SEIRSSimulation(unsigned int seed_)
        : rng(seed_),
          seed(seed_),
          grid_size(GRID_SIZE),
          infected_grid(GRID_SIZE * GRID_SIZE, 0),
          uni01(0.0, 1.0),
          pos_dist(0, GRID_SIZE - 1)
    {
        agents.reserve(N_AGENTS);
        history_S.reserve(N_ITERATIONS);
        history_E.reserve(N_ITERATIONS);
        history_I.reserve(N_ITERATIONS);
        history_R.reserve(N_ITERATIONS);
    }

    void run()
    {
        init_agents();

        std::cout << "  Simulation en cours (seed=" << seed << ")..."
                  << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        // État initial
        record_statistics();

        for (int iter = 1; iter < N_ITERATIONS; ++iter)
        {
            simulation_step();
            record_statistics();

            if (iter % 100 == 0)
            {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed =
                    std::chrono::duration_cast<std::chrono::seconds>(now - start);

                std::cout << "    Jour " << iter << "/" << N_ITERATIONS
                          << " (S=" << history_S.back()
                          << ", E=" << history_E.back()
                          << ", I=" << history_I.back()
                          << ", R=" << history_R.back()
                          << ") [" << elapsed.count() << "s]\n";
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "  ✓ Simulation terminée en "
                  << duration.count() / 1000.0 << " secondes\n";
    }

    void export_csv(const std::string &filename)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Erreur : impossible d'ouvrir " << filename << "\n";
            return;
        }

        file << "iteration,S,E,I,R\n";
        for (size_t i = 0; i < history_S.size(); ++i)
        {
            file << i << "," << history_S[i] << "," << history_E[i] << ","
                 << history_I[i] << "," << history_R[i] << "\n";
        }

        std::cout << "  ✓ Données exportées : " << filename << "\n";
    }

    void print_peak_info()
    {
        auto max_I_it = std::max_element(history_I.begin(), history_I.end());
        auto max_E_it = std::max_element(history_E.begin(), history_E.end());
        int peak_day = static_cast<int>(
            std::distance(history_I.begin(), max_I_it));

        std::cout << "  Statistiques :\n";
        std::cout << "    - Pic infectés : " << *max_I_it
                  << " au jour " << peak_day << "\n";
        std::cout << "    - Max exposés  : " << *max_E_it << "\n";
    }
};

// =============================================================================
// UTILITAIRES
// =============================================================================

void create_output_dir(const std::string &dir)
{
#ifdef _WIN32
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0777);
#endif
}

std::vector<unsigned int> generate_seeds(int n, unsigned int base_seed = 42)
{
    std::mt19937 rng(base_seed);
    std::uniform_int_distribution<unsigned int> dist(0, 0x7FFFFFFF);

    std::vector<unsigned int> seeds;
    seeds.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        seeds.push_back(dist(rng));
    }
    return seeds;
}

// =============================================================================
// MAIN
// =============================================================================

int main()
{
    std::cout << "================================================================================\n";
    std::cout << "PROJET M2 - PARTIE 2 : MODÈLE MULTI-AGENT SEIRS (C++ OPTIMISÉ)\n";
    std::cout << "================================================================================\n\n";

    std::cout << "Configuration :\n";
    std::cout << "  - Nombre d'agents       : " << N_AGENTS << "\n";
    std::cout << "  - Taille grille         : " << GRID_SIZE << "×" << GRID_SIZE << "\n";
    std::cout << "  - Itérations            : " << N_ITERATIONS << " jours\n";
    std::cout << "  - Réplications          : " << N_REPLICATIONS << "\n";
    std::cout << "  - Infectés initiaux     : " << N_INITIAL_INFECTED << "\n\n";

    create_output_dir(OUTPUT_DIR);
    std::cout << "✓ Dossier de sortie : " << OUTPUT_DIR << "/\n\n";

    auto seeds = generate_seeds(N_REPLICATIONS);
    std::cout << "✓ " << N_REPLICATIONS << " seeds générées\n\n";

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int rep = 0; rep < N_REPLICATIONS; ++rep)
    {
        std::cout << "================================================================================\n";
        std::cout << "RÉPLICATION " << (rep + 1) << "/" << N_REPLICATIONS
                  << " (seed=" << seeds[rep] << ")\n";
        std::cout << "================================================================================\n";

        SEIRSSimulation sim(seeds[rep]);
        sim.run();

        std::ostringstream filename;
        filename << OUTPUT_DIR << "/results_CPP_rep_"
                 << std::setfill('0') << std::setw(2) << rep << ".csv";
        sim.export_csv(filename.str());

        sim.print_peak_info();
        std::cout << "\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration =
        std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);

    std::cout << "================================================================================\n";
    std::cout << "✓ TOUTES LES RÉPLICATIONS TERMINÉES AVEC SUCCÈS\n";
    std::cout << "================================================================================\n";
    std::cout << "Temps total : " << total_duration.count() << " secondes ("
              << total_duration.count() / 60.0 << " minutes)\n";
    std::cout << "Fichiers générés : " << N_REPLICATIONS
              << " fichiers CSV dans " << OUTPUT_DIR << "/\n\n";
    std::cout << "Prochaine étape : Analyse statistique (Jupyter Notebook)\n";

    return 0;
}
