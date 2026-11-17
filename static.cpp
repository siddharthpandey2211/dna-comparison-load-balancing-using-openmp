
#include "nw_common.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace chrono;

int main() {
    vector<string> seq;

    if (!nw::readSequences("data/sequences.txt", seq)) {
        cout << "Dataset not found!" << endl;
        return 0;
    }

    int total = seq.size() / 2;
    vector<int> scores(total);

    omp_set_num_threads(12);

    auto start = high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; i++) {
        scores[i] = nw::calculate2D(seq[2 * i], seq[2 * i + 1]);
    }

    auto stop = high_resolution_clock::now();

    long long totalScore = 0;
    for (int i = 0; i < total; i++) {
        totalScore += scores[i];
    }

    double time = duration<double, milli>(stop - start).count();

    cout << "Total pairs aligned: " << total << endl;
    cout << "Total alignment score: " << totalScore << endl;
    cout << "Number of threads: " << 12 << endl;
    cout << "Static execution time: " << time << " ms" << endl;

    return 0;
}
