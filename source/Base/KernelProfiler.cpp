#include "KernelProfiler.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

void KernelProfiler::record(char const* name, std::chrono::steady_clock::duration duration)
{
    if (!_enabled) {
        return;
    }
    auto nanoseconds = std::chrono::duration<double, std::nano>(duration).count();

    std::lock_guard lock(_mutex);
    auto& entry = _entries[name];
    ++entry.count;
    entry.totalNanoseconds += nanoseconds;
}

void KernelProfiler::reset()
{
    std::lock_guard lock(_mutex);
    _entries.clear();
}

std::string KernelProfiler::getReport() const
{
    std::lock_guard lock(_mutex);

    std::vector<std::pair<std::string, Entry>> sorted(_entries.begin(), _entries.end());
    std::ranges::sort(sorted, [](auto const& left, auto const& right) { return left.second.totalNanoseconds > right.second.totalNanoseconds; });

    double totalNanoseconds = 0.0;
    uint64_t totalCount = 0;
    for (auto const& [name, entry] : sorted) {
        totalNanoseconds += entry.totalNanoseconds;
        totalCount += entry.count;
    }

    std::ostringstream stream;
    stream << "Kernel profiling report (debug mode, wall-clock per kernel incl. launch/sync overhead)\n";
    stream << std::left << std::setw(4) << "#" << std::setw(52) << "kernel" << std::right << std::setw(10) << "calls" << std::setw(14) << "total [ms]"
           << std::setw(12) << "avg [us]" << std::setw(9) << "share" << "\n";

    int rank = 1;
    for (auto const& [name, entry] : sorted) {
        auto totalMs = entry.totalNanoseconds / 1.0e6;
        auto avgUs = entry.count != 0 ? entry.totalNanoseconds / 1.0e3 / static_cast<double>(entry.count) : 0.0;
        auto share = totalNanoseconds != 0.0 ? 100.0 * entry.totalNanoseconds / totalNanoseconds : 0.0;
        stream << std::left << std::setw(4) << rank << std::setw(52) << name << std::right << std::setw(10) << entry.count << std::setw(14) << std::fixed
               << std::setprecision(3) << totalMs << std::setw(12) << std::setprecision(1) << avgUs << std::setw(8) << std::setprecision(1) << share << "%\n";
        ++rank;
    }
    stream << std::left << std::setw(4) << "" << std::setw(52) << "total" << std::right << std::setw(10) << totalCount << std::setw(14) << std::fixed
           << std::setprecision(3) << (totalNanoseconds / 1.0e6) << std::setw(12) << "" << std::setw(8) << "100.0" << "%\n";

    return stream.str();
}
