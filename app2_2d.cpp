
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
        return 1;
    }

    int total = seq.size() / 2;
    if (total == 0) return 1;

    const int threads = 12;

    auto start = high_resolution_clock::now();

    vector<Task> tasks;

    for (int i = 0; i < total; i++) {
        long long work = 1LL * seq[2 * i].size() * seq[2 * i + 1].size();
        tasks.push_back({i, work});
    }

    sort(tasks.begin(), tasks.end(),
         [](Task a, Task b) {
             return a.work > b.work;
         });

    vector<vector<int>> batches(threads);
    vector<long long> threadWork(threads, 0);
    vector<int> scores(total, 0);

    int next = 0;
    bool done = false;
    double schedulingTime = 0;

    #pragma omp parallel num_threads(12) shared(batches, next, done, schedulingTime, threadWork, scores)
    {
        int id = omp_get_thread_num();

        while (true) {
            #pragma omp single
            {
                auto scheduleStart = high_resolution_clock::now();

                for (int t = 0; t < threads; t++)
                    batches[t].clear();

                if (next >= total) {
                    done = true;
                } else {
                    done = false;

                    long long remainingWork = 0;

                    for (int i = next; i < total; i++)
                        remainingWork += tasks[i].work;

                    long long target = max(1LL, remainingWork / threads);

                    vector<long long> batchWork(threads, 0);
                    vector<bool> reached(threads, false);
                    int reachedCount = 0;

                    while (next < total && reachedCount < threads) {
                        int least = -1;

                        for (int t = 0; t < threads; t++) {
                            if (!reached[t] &&
                                (least == -1 || threadWork[t] < threadWork[least])) {
                                least = t;
                            }
                        }

                        if (least == -1) break;

                        int taskId = tasks[next].id;
                        batches[least].push_back(taskId);

                        threadWork[least] += tasks[next].work;
                        batchWork[least] += tasks[next].work;
                        next++;

                        if (batchWork[least] >= target) {
                            reached[least] = true;
                            reachedCount++;
                        }
                    }

                    while (next < total) {
                        int least = 0;

                        for (int t = 1; t < threads; t++) {
                            if (threadWork[t] < threadWork[least])
                                least = t;
                        }

                        int taskId = tasks[next].id;
                        batches[least].push_back(taskId);
                        threadWork[least] += tasks[next].work;
                        next++;
                    }
                }

                auto scheduleEnd = high_resolution_clock::now();

                schedulingTime += duration<double, milli>(
                    scheduleEnd - scheduleStart).count();
            }

            if (done) break;

            for (int taskId : batches[id]) {
                scores[taskId] = nw::calculate2D(
                    seq[2 * taskId], seq[2 * taskId + 1]);
            }

            #pragma omp barrier
        }
    }

    auto stop = high_resolution_clock::now();

    long long totalScore = 0;
    for (int score : scores)
        totalScore += score;

    long long minWork = threadWork[0];
    long long maxWork = threadWork[0];
    long long totalWork = 0;

    for (int work : threadWork) {
        minWork = min(minWork, (long long)work);
	maxWork = max(maxWork, (long long)work);
        totalWork += work;
    }

    double avgWork = (double)totalWork / threads;
    double imbalance = avgWork > 0
        ? (maxWork - minWork) * 100.0 / avgWork : 0;

    double executionTime = duration<double, milli>(stop - start).count();

    cout << "Total pairs aligned: " << total << endl;
    cout << "Total alignment score: " << totalScore << endl;
    cout << "Number of threads: " << threads << endl;
    cout << "Execution time: " << executionTime << " ms" << endl;
    cout << "Scheduling overhead: " << schedulingTime << " ms" << endl;
    cout << "Load imbalance: " << imbalance << "%" << endl;

    return 0;
}
