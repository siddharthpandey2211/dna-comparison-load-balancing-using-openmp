
#include "nw_common.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace chrono;

struct Task {
    int id;
    long long work;
};

int main() {
    vector<string> seq;

    if (!nw::readSequences("data/sequences.txt", seq)) {
        cout << "Dataset not found!" << endl;
        return 0;
    }

    int total = seq.size() / 2;
    const int threads = 12;

    vector<int> scores(total, 0);
    vector<Task> tasks;

    for (int i = 0; i < total; i++) {
        long long work = 1LL * seq[2 * i].size() * seq[2 * i + 1].size();
        tasks.push_back({i, work});
    }

    omp_set_num_threads(threads);

    auto start = high_resolution_clock::now();

    sort(tasks.begin(), tasks.end(),
         [](Task a, Task b) {
             return a.work > b.work;
         });

    vector<vector<int>> initial(threads);
    vector<long long> loads(threads, 0);

    int first = min(total, 2 * threads);

    for (int i = 0; i < first; i++) {
        int t = 0;

        for (int j = 1; j < threads; j++) {
            if (loads[j] < loads[t])
                t = j;
        }

        initial[t].push_back(tasks[i].id);
        loads[t] += tasks[i].work;
    }

    vector<int> pending;

    for (int i = first; i < total; i++)
        pending.push_back(tasks[i].id);

    int next = 0, batch = 0, done = 0;

    #pragma omp parallel num_threads(12) shared(next, batch, done)
    {
        int id = omp_get_thread_num();

        for (int task : initial[id]) {
            scores[task] = nw::calculate2D(
                seq[2 * task], seq[2 * task + 1]);
        }

        #pragma omp barrier

        while (true) {
            #pragma omp single
            {
                int Q = pending.size() - next;
                int I = omp_get_num_threads();

                if (Q == 0)
                    done = 1;
                else {
                    batch = max(1, Q / I);
                    done = 0;
                }
            }

            if (done)
                break;

            vector<int> myTasks;

            #pragma omp critical
            {
                for (int i = 0; i < batch && next < pending.size(); i++) {
                    myTasks.push_back(pending[next]);
                    next++;
                }
            }

            #pragma omp barrier

            for (int task : myTasks) {
                scores[task] = nw::calculate2D(
                    seq[2 * task], seq[2 * task + 1]);
            }

            #pragma omp barrier
        }
    }

    auto stop = high_resolution_clock::now();

    long long totalScore = 0;

    for (int i = 0; i < total; i++)
        totalScore += scores[i];

    double time = duration<double, milli>(stop - start).count();

    cout << "Total pairs aligned: " << total << endl;
    cout << "Total alignment score: " << totalScore << endl;
    cout << "Number of threads: " << threads << endl;
    cout << "Execution time: " << time << " ms" << endl;

    return 0;
}
