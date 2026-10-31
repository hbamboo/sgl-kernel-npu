#pragma once
#include <cstdint>
struct ScatterMlaTiling {
    int64_t tokens;
    int64_t heads;
    int64_t nopeDim;
    int64_t ropeDim;
    int64_t valueDim;
    int64_t capacity;
    int64_t tileTokens;
    int64_t usedCores;
    int64_t tokensPerCore;
    int64_t indexBytes;
    int64_t coalesce;
    int64_t writeDense;
};
