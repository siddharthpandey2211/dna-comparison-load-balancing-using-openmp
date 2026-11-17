
#include "nw_common.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace chrono;

struct Task {
    int index;
    long long work;
};

int myersEditDistance(const string& pattern, const string& text) {
    int m = pattern.size();

    if (m == 0) return text.size();

    int words = (m + 63) / 64;
    int rem = m & 63;

    uint64_t lastMask = (rem == 0) ? UINT64_MAX : ((uint64_t(1) << rem) - 1);
    uint64_t highBit = uint64_t(1) << ((m - 1) & 63);
    int highWord = words - 1;

    vector<uint64_t> peq(256 * words, 0);

    for (int i = 0; i < m; i++) {
        unsigned char ch = pattern[i];
        peq[ch * words + (i >> 6)] |= uint64_t(1) << (i & 63);
    }

    vector<uint64_t> Pv(words, UINT64_MAX), Mv(words, 0);
    vector<uint64_t> Xv(words), Xh(words), Ph(words), Mh(words);

    Pv[highWord] &= lastMask;

    int score = m;

    for (unsigned char ch : text) {
        const uint64_t* Eq = peq.data() + ch * words;

        uint64_t carry = 0;

        for (int w = 0; w < words; w++) {
            uint64_t e = Eq[w];
            uint64_t p = Pv[w];

            Xv[w] = e | Mv[w];

            uint64_t a = e & p;
            uint64_t sum1 = a + p;
            uint64_t c1 = (sum1 < a);

            uint64_t sum2 = sum1 + carry;
            uint64_t c2 = (sum2 < sum1);

            carry = c1 | c2;
            Xh[w] = (sum2 ^ p) | e;
        }

        for (int w = 0; w < words; w++) {
            Ph[w] = Mv[w] | ~(Xh[w] | Pv[w]);
            Mh[w] = Pv[w] & Xh[w];
        }

        if (Ph[highWord] & highBit)
            score++;
        else if (Mh[highWord] & highBit)
            score--;

        uint64_t phCarry = 1, mhCarry = 0;

        for (int w = 0; w < words; w++) {
            uint64_t nextPh = Ph[w] >> 63;
            uint64_t nextMh = Mh[w] >> 63;

            Ph[w] = (Ph[w] << 1) | phCarry;
            Mh[w] = (Mh[w] << 1) | mhCarry;

            phCarry = nextPh;
            mhCarry = nextMh;
        }

        for (int w = 0; w < words; w++) {
            Pv[w] = Mh[w] | ~(Xv[w] | Ph[w]);
            Mv[w] = Ph[w] & Xv[w];
        }

        Pv[highWord] &= lastMask;
        Mv[highWord] &= lastMask;
    }

    return score;
}

int calculateBitParallelNW(const string& a, const string& b) {
    int distance = myersEditDistance(a, b);

    return a.size() + b.size() - 3 * distance;
}

int main() {
    const int THREADS = 12;
    const double alpha = 0.75;

    vector<string> seq;

    if (!nw::readSequences("data/sequences.txt", seq)) {
        cout << "Dataset not found!" << endl;
        return 1;
    }

    if (seq.size() < 2 || seq.size() % 2 != 0)
        return 1;

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

    long long target = (1.0 - alpha) * totalWork;

    vector<bool> isReservoir(total, false);
    long long reservoirWork = 0;

    for (const Task& task : sortedTasks) {
        if (reservoirWork >= target) break;

        isReservoir[task.index] = true;
        reservoirWork += task.work;
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
             return a.work > b.work;
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
            continue;
        }

        long long targetWork = headWork * (t + 1) / THREADS;
        int maxEnd = head.size() - (THREADS - t - 1);

        int end = cursor;

        while (end < maxEnd && prefix[end] < targetWork)
            end++;

        if (end > cursor && end < (int)head.size()) {
            long long before = llabs(prefix[end - 1] - targetWork);
            long long after = llabs(prefix[end] - targetWork);

            if (before < after)
                end--;
        }

        rangeEnd[t] = end;
        cursor = end;
    }

    atomic<int> nextReservoir(0);

    #pragma omp parallel num_threads(THREADS)
    {
        int tid = omp_get_thread_num();

        for (int p = rangeStart[tid]; p < rangeEnd[tid]; p++) {
            int idx = head[p];

            scores[idx] = calculateBitParallelNW(
                seq[2 * idx], seq[2 * idx + 1]);
        }

        while (true) {
            int r = nextReservoir.fetch_add(1, memory_order_relaxed);

            if (r >= (int)reservoir.size())
                break;

            int idx = reservoir[r].index;

            scores[idx] = calculateBitParallelNW(
                seq[2 * idx], seq[2 * idx + 1]);
        }
    }

    auto stop = steady_clock::now();

    long long totalScore = accumulate(scores.begin(), scores.end(), 0LL);
    double time = duration<double, milli>(stop - start).count();

    cout << "Total pairs aligned: " << total << endl;
    cout << "Total alignment score: " << totalScore << endl;
    cout << "Number of threads: " << THREADS << endl;
    cout << "Static head tasks: " << head.size() << endl;
    cout << "Reservoir tasks: " << reservoir.size() << endl;
    cout << "Execution time: " << time << " ms" << endl;

    return 0;
}
