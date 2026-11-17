
#include "nw_common.h"
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <climits>
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

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    auto start = high_resolution_clock::now();

    vector<Task> tasks;
    vector<long long> taskWork(total);

    for (int i = 0; i < total; i++) {
        long long work = 1LL * seq[2 * i].size() * seq[2 * i + 1].size();
        tasks.push_back({i, work});
        taskWork[i] = work;
    }

    sort(tasks.begin(), tasks.end(),
         [](Task a, Task b) {
             return a.work > b.work;
         });

    vector<deque<int>> batches(threads);
    vector<long long> threadWork(threads, 0);
    vector<long long> actualWork(threads, 0);
    vector<int> stolenTasks(threads, 0);
    vector<int> scores(total, 0);

    int next = 0;
    int outstanding = 0;
    bool done = false;
    double schedulingTime = 0;

    #pragma omp parallel num_threads(12) shared(batches, next, done, outstanding, schedulingTime, threadWork, actualWork, stolenTasks, scores)
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

                    outstanding = next;
                }

                auto scheduleEnd = high_resolution_clock::now();

                schedulingTime += duration<double, milli>(
                    scheduleEnd - scheduleStart).count();
            }

            if (done) break;

            while (true) {
                int task = -1;

                #pragma omp critical(queue_lock)
                {
                    if (!batches[id].empty()) {
                        task = batches[id].back();
                        batches[id].pop_back();
                    } else {
                        for (int offset = 1; offset < threads; offset++) {
                            int victim = (id + offset) % threads;

                            if (!batches[victim].empty()) {
                                task = batches[victim].front();
                                batches[victim].pop_front();
                                stolenTasks[id]++;
                                break;
                            }
                        }
                    }
                }

                if (task != -1) {
                    scores[task] = nw::calculate2D(
                        seq[2 * task], seq[2 * task + 1]);

                    actualWork[id] += taskWork[task];

                    #pragma omp atomic update
                    outstanding--;
                } else {
                    int remaining;

                    #pragma omp atomic read
                    remaining = outstanding;

                    if (remaining == 0) break;
                }
            }

            #pragma omp barrier
        }
    }

    auto stop = high_resolution_clock::now();

    long long totalScore = 0;
    for (int score : scores)
        totalScore += score;

    long long totalWork = 0;
    long long maxWork = 0;
    long long minWork = LLONG_MAX;
    int totalStolen = 0;

    for (int i = 0; i < threads; i++) {
        totalWork += actualWork[i];
        maxWork = max(maxWork, actualWork[i]);
        minWork = min(minWork, actualWork[i]);
        totalStolen += stolenTasks[i];
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
    cout << "Total stolen tasks: " << totalStolen << endl;
    cout << "Load imbalance: " << imbalance << "%" << endl;

    return 0;
}
