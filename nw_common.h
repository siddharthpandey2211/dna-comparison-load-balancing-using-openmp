
#ifndef NW_COMMON_H
#define NW_COMMON_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

namespace nw {

inline bool readSequences(const std::string& path, std::vector<std::string>& seq) {
    std::ifstream file(path);
    if (!file) return false;

    std::string line, current;

    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == '>') {
            if (!current.empty()) seq.push_back(current);
            current.clear();
        } else {
            current += line;
        }
    }

    if (!current.empty()) seq.push_back(current);

    return true;
}

inline int calculate2D(const std::string& a, const std::string& b) {
    const int MATCH = 2, MISMATCH = -1, GAP = -2;
    int m = a.size(), n = b.size();

    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (int i = 0; i <= m; i++)
        dp[i][0] = i * GAP;

    for (int j = 0; j <= n; j++)
        dp[0][j] = j * GAP;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int score = (a[i - 1] == b[j - 1]) ? MATCH : MISMATCH;

            dp[i][j] = std::max({
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + GAP,
                dp[i][j - 1] + GAP
            });
        }
    }

    return dp[m][n];
}

}

#endif
