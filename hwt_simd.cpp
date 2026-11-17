
#include "nw_common.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <immintrin.h>
#include <omp.h>
#include <numeric>
using namespace std;
using namespace chrono;

struct Task {
    int index;
    long long work;
};

int calculate_simd(const string& input1, const string& input2) {
    const string *a = &input1, *b = &input2;

    if (a->size() > b->size())
        swap(a, b);

    int m = a->size(), n = b->size();

    const int MATCH = 2, MISMATCH = -1, GAP = -2;

    vector<int> prev2(m + 1), prev1(m + 1), curr(m + 1);

    prev1[0] = 0;

    for (int k = 1; k <= m + n; k++) {
        if (k <= n) curr[0] = k * GAP;
        if (k <= m) curr[k] = k * GAP;

        int low = max(1, k - n);
        int high = min(m, k - 1);
        int i = low;

        for (; i + 7 <= high; i += 8) {
            alignas(32) int scoreValues[8];

            for (int lane = 0; lane < 8; lane++) {
                int row = i + lane;
                int col = k - row;

                scoreValues[lane] =
                    ((*a)[row - 1] == (*b)[col - 1]) ? MATCH : MISMATCH;
            }

            __m256i diagonal = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(prev2.data() + i - 1));
            __m256i up = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(prev1.data() + i - 1));
            __m256i left = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(prev1.data() + i));
            __m256i score = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(scoreValues));

            diagonal = _mm256_add_epi32(diagonal, score);
            up = _mm256_add_epi32(up, _mm256_set1_epi32(GAP));
            left = _mm256_add_epi32(left, _mm256_set1_epi32(GAP));

            __m256i best = _mm256_max_epi32(diagonal, up);
            best = _mm256_max_epi32(best, left);

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(curr.data() + i), best);
        }

        for (; i <= high; i++) {
            int j = k - i;

            int diagonal = prev2[i - 1] +
                (((*a)[i - 1] == (*b)[j - 1]) ? MATCH : MISMATCH);
            int up = prev1[i - 1] + GAP;
            int left = prev1[i] + GAP;

            curr[i] = max(diagonal, max(up, left));
        }

        prev2.swap(prev1);
        prev1.swap(curr);
    }

    return prev1[m];
}

int main(int argc, char* argv[]) {
    const int THREADS = 12;
    double alpha = 0.75;

    if (argc >= 2) {
        try {
            alpha = stod(argv[1]);
        } catch (...) {
            cout << "Invalid alpha value!" << endl;
            return 1;
        }
    }

    if (alpha < 0 || alpha > 1) {
        cout << "Alpha must be between 0 and 1." << endl;
        return 1;
    }

    __builtin_cpu_init();

    if (!__builtin_cpu_supports("avx2")) {
        cout << "AVX2 is not supported by this CPU." << endl;
        return 1;
    }

    vector<string> seq;

    if (!nw::readSequences("data/sequences.txt", seq)) {
        cout << "Dataset not found!" << endl;
        return 1;
    }

    if (seq.size() < 2 || seq.size() % 2 != 0) {
        cout << "Invalid dataset!" << endl;
        return 1;
    }

    int total = seq.size() / 2;
    vector<int> scores(total, 0);
    vector<Task> tasks;

    omp_set_dynamic(0);
    omp_set_num_threads(THREADS);

    auto start = steady_clock::now();

    long long totalWork = 0;

    for (int i = 0; i < total; i++) {
        long long work = 1LL * seq[2 * i].size() *
                         seq[2 * i + 1].size();

        tasks.push_back({i, work});
        totalWork += work;
    }

    vector<Task> sortedTasks = tasks;

    sort(sortedTasks.begin(), sortedTasks.end(),
         [](const Task& a, const Task& b) {
             if (a.work != b.work) return a.work < b.work;
             return a.index < b.index;
         });

    long long target = (long long)((1.0 - alpha) * totalWork);

    vector<bool> isReservoir(total, false);
    long long reservoirWork = 0;
    int reservoirCount = 0;

    for (const Task& task : sortedTasks) {
        if (reservoirWork >= target) break;

        isReservoir[task.index] = true;
        reservoirWork += task.work;
        reservoirCount++;
    }

    vector<Task> reservoir;
    vector<int> head;

    for (const Task& task : tasks) {
        if (isReservoir[task.index])
            reservoir.push_back(task);
        else
            head.push_back(task.index);
    }

    sort(reservoir.begin(), reservoir.end(),
         [](const Task& a, const Task& b) {
             if (a.work != b.work) return a.work > b.work;
             return a.index < b.index;
         });

    vector<long long> prefix(head.size() + 1, 0);

    for (int i = 0; i < (int)head.size(); i++)
        prefix[i + 1] = prefix[i] + tasks[head[i]].work;

    vector<int> rangeStart(THREADS), rangeEnd(THREADS);

    int cursor = 0;
    long long headWork = prefix.back();

    for (int t = 0; t < THREADS; t++) {
        rangeStart[t] = cursor;

        if (t == THREADS - 1) {
            rangeEnd[t] = head.size();
            cursor = rangeEnd[t];
            continue;
        }

        long long targetWork = headWork * (t + 1) / THREADS;
        int maxEnd = head.size() - (THREADS - t - 1);

        int end = cursor;

        while (end < maxEnd && prefix[end] < targetWork)
            end++;

        if (end > cursor && end < (int)head.size()) {
            long long beforeDiff = llabs(prefix[end - 1] - targetWork);
            long long afterDiff = llabs(prefix[end] - targetWork);

            if (beforeDiff < afterDiff)
                end--;
        }

        rangeEnd[t] = end;
        cursor = end;
    }

    atomic<int> nextReservoir(0);
    vector<int> threadTasks(THREADS, 0);

    #pragma omp parallel num_threads(THREADS)
    {
        int tid = omp_get_thread_num();

        for (int p = rangeStart[tid]; p < rangeEnd[tid]; p++) {
            int idx = head[p];

            scores[idx] = calculate_simd(
                seq[2 * idx], seq[2 * idx + 1]);

            threadTasks[tid]++;
        }

        while (true) {
            int r = nextReservoir.fetch_add(1, memory_order_relaxed);

            if (r >= (int)reservoir.size())
                break;

            int idx = reservoir[r].index;

            scores[idx] = calculate_simd(
                seq[2 * idx], seq[2 * idx + 1]);

            threadTasks[tid]++;
        }
    }

    auto stop = steady_clock::now();

    long long totalScore = accumulate(scores.begin(), scores.end(), 0LL);
    double time = duration<double, milli>(stop - start).count();

    cout << "Total pairs aligned: " << total << endl;
    cout << "Total alignment score: " << totalScore << endl;
    cout << "Number of threads: " << THREADS << endl;
    cout << "Alpha: " << alpha << endl;
    cout << "Static head tasks: " << head.size() << endl;
    cout << "Reservoir tasks: " << reservoir.size() << endl;
    cout << "Execution time: " << time << " ms" << endl;

    return 0;
}
