#include "nw_common.h"
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace chrono;

int main() {
    vector<string> seq;

    if (!nw::readSequences("data/sequences.txt", seq)) {
        cout << "Dataset not found!" << endl;
        return 0;
    }

    auto start = high_resolution_clock::now();

    long long totalScore = 0;

    for (int i = 0; i + 1 < seq.size(); i += 2) {
        totalScore += nw::calculate2D(seq[i], seq[i + 1]);
    }

    auto stop = high_resolution_clock::now();

    double time = duration<double, milli>(stop - start).count();

    cout << "Total pairs aligned: " << seq.size() / 2 << endl;
    cout << "Total alignment score: " << totalScore << endl;
    cout << "Sequential execution time: " << time << " ms" << endl;

    return 0;
}
