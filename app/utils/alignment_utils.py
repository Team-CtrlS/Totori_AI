from typing import TypeVar

T = TypeVar("T")

def levenshtein(s1: list, s2: list) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def get_alignment(original: list, stt: list) -> tuple[list, list]:
    m, n = len(original), len(stt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if original[i - 1] == stt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    align1, align2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and original[i - 1] == stt[j - 1]:
            align1.append(original[i - 1])
            align2.append(stt[j - 1])
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            align1.append(original[i - 1])
            align2.append(stt[j - 1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            align1.append(original[i - 1])
            align2.append("-")
            i -= 1
        else:
            align1.append("-")
            align2.append(stt[j - 1])
            j -= 1
    return align1[::-1], align2[::-1]