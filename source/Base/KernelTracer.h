#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <cstdio>

#include "Singleton.h"

// Traces kernel calls into a file. Only active in debug mode; fed by the kernel-call macros, where each launch is
// followed by a stream sync. Every call is written when it starts and again when it completes, both into a fixed
// record at the top of the file and into a ring holding the preceding calls, so a kernel that hangs stays visible as
// running. The records are written unbuffered: they reach the file even if the process is killed without unwinding,
// as happens on a driver timeout.
class KernelTracer
{
    MAKE_SINGLETON(KernelTracer);

public:
    ~KernelTracer();

    // Starts a fresh trace; may be called again to restart it
    void init(std::filesystem::path const& filename);
    void close();
    bool isEnabled() const { return _file != nullptr; }

    void setTimestep(uint64_t value);

    void traceBegin(char const* name);
    void traceEnd(std::chrono::steady_clock::duration duration);

private:
    void writeEntry(std::string const& status, bool intoHistory);
    void writeRecord(int recordIndex, std::string const& text);

    std::FILE* _file = nullptr;
    uint64_t _timestep = 0;
    std::mutex _mutex;
    uint64_t _callIndex = 0;
    char const* _pendingName = nullptr;
};
