#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

int main() {
    const int NUM_RUNS = 1000000;
    const int d_DISTANCE = 9;
    const int size = d_DISTANCE * d_DISTANCE;
    int idx = 0;

    long long total_duration = 0;

    for (int run = 0; run < NUM_RUNS; ++run) {
        std::vector<char> ancilla(size);
        std::vector<char> pure(size);

        // Initialize random seed
        std::srand(std::time(nullptr));

        // Fill ancilla with random 0 or 1
        for(int i = 0; i < size; ++i) {
            ancilla[i] = std::rand() % 2;
        }

        // Fill pure with random 0
        std::fill(pure.begin(), pure.end(), 0);

        auto start = std::chrono::high_resolution_clock::now();

        for(char p = -1; p < 2; p += 2) {
            unsigned r0 = (d_DISTANCE - 2 + p) / 2;
            for(unsigned c0 = 0; c0 < (d_DISTANCE + 1) / 2; c0++) {
                unsigned tx = 0, tz = 0;
                for(unsigned i = 0; i < (d_DISTANCE - 1) / 2; i++) {
                    unsigned r = r0 + p * i;
                    unsigned ax = r * (d_DISTANCE + 1) / 2 + c0;
                    unsigned az = ax + (size - 1) / 2;
                    unsigned dx = r + 2 * d_DISTANCE * c0 + (1 + p) / 2;
                    unsigned dz = (r + (3 + p) / 2) * d_DISTANCE - 1 - 2 * c0;

                    tx ^= ancilla[ax + idx * (size - 1)] << 1;
                    tz ^= ancilla[az + idx * (size - 1)];

                    pure[dx + idx * size] ^= tx;
                    pure[dz + idx * size] ^= tz;    
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        total_duration += duration.count();
    }

    std::cout << "Average execution time: " << total_duration / NUM_RUNS << " nanoseconds" << std::endl;

    return 0;
}