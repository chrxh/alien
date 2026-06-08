#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Singleton.h"

// Accumulates per-kernel wall-clock durations. Only active when enabled; fed by the kernel-call macros in debug
// mode, where each launch is followed by a stream sync so the recorded duration is the kernel's execution time.
class KernelProfiler
{
    MAKE_SINGLETON(KernelProfiler);

public:
    void setEnabled(bool value) { _enabled = value; }
    bool isEnabled() const { return _enabled; }

    void record(char const* name, std::chrono::steady_clock::duration duration);

    void reset();
    std::string getReport() const;

private:
    struct Entry
    {
        uint64_t count = 0;
        double totalNanoseconds = 0.0;
    };

    bool _enabled = false;
    mutable std::mutex _mutex;
    std::unordered_map<std::string, Entry> _entries;
};
