#include "KernelProfiler.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ranges>
#include <sstream>
#include <utility>
#include <vector>

namespace
{
    auto constexpr WriteInterval = std::chrono::seconds(1);

    std::array<KernelCategory, NumKernelCategories> const AllCategories = {KernelCategory::Simulation, KernelCategory::Rendering, KernelCategory::Other};

    std::string getCategoryName(KernelCategory category)
    {
        switch (category) {
        case KernelCategory::Simulation:
            return "Simulation";
        case KernelCategory::Rendering:
            return "Rendering";
        default:
            return "Other";
        }
    }

    double toMilliseconds(double nanoseconds)
    {
        return nanoseconds / 1.0e6;
    }

    double getShare(double nanoseconds, double totalNanoseconds)
    {
        return totalNanoseconds != 0.0 ? 100.0 * nanoseconds / totalNanoseconds : 0.0;
    }
}

thread_local KernelCategory KernelProfiler::_category = KernelCategory::Other;

KernelProfiler::CategoryScope::CategoryScope(KernelCategory category)
    : _previousCategory(_category)
{
    _category = category;
}

KernelProfiler::CategoryScope::~CategoryScope()
{
    _category = _previousCategory;
}

KernelProfiler::~KernelProfiler()
{
    close();
}

void KernelProfiler::init(std::filesystem::path const& filename)
{
    std::lock_guard lock(_mutex);
    _filename = filename;
    _enabled = true;
    for (auto& entries : _entriesByCategory) {
        entries.clear();
    }
    _lastWriteTimepoint = std::chrono::steady_clock::now();
    writeReport();
}

void KernelProfiler::close()
{
    std::lock_guard lock(_mutex);
    if (!_enabled) {
        return;
    }
    writeReport();
    _enabled = false;
}

void KernelProfiler::setContext(std::string const& key, std::string const& value)
{
    if (!_enabled) {
        return;
    }
    std::lock_guard lock(_mutex);
    auto match = std::ranges::find(_context, key, &std::pair<std::string, std::string>::first);
    if (match != _context.end()) {
        match->second = value;
    } else {
        _context.emplace_back(key, value);
    }
}

void KernelProfiler::record(char const* name, std::chrono::steady_clock::duration duration, int numBlocks, int threadsPerBlock)
{
    if (!_enabled) {
        return;
    }
    auto nanoseconds = std::chrono::duration<double, std::nano>(duration).count();

    std::lock_guard lock(_mutex);
    auto& entry = _entriesByCategory.at(static_cast<int>(_category))[name];
    ++entry.count;
    entry.totalNanoseconds += nanoseconds;
    entry.numBlocks = numBlocks;
    entry.threadsPerBlock = threadsPerBlock;

    auto now = std::chrono::steady_clock::now();
    if (now - _lastWriteTimepoint >= WriteInterval) {
        _lastWriteTimepoint = now;
        writeReport();
    }
}

double KernelProfiler::getAverageNanoseconds(std::string const& name) const
{
    std::lock_guard lock(_mutex);
    for (auto const& entries : _entriesByCategory) {
        auto match = entries.find(name);
        if (match != entries.end() && match->second.count != 0) {
            return match->second.totalNanoseconds / static_cast<double>(match->second.count);
        }
    }
    return 0.0;
}

std::string KernelProfiler::getReport() const
{
    std::lock_guard lock(_mutex);
    return createReport();
}

void KernelProfiler::writeReport() const
{
    std::ofstream file(_filename, std::ios_base::trunc);
    if (!file.is_open()) {
        return;
    }
    file << createReport();
}

KernelProfiler::Entry KernelProfiler::sumUp(Entries const& entries)
{
    Entry result;
    for (auto const& entry : entries | std::views::values) {
        result.count += entry.count;
        result.totalNanoseconds += entry.totalNanoseconds;
    }
    return result;
}

std::string KernelProfiler::createReport() const
{
    std::ostringstream stream;
    stream << "Kernel profiling report (debug mode, wall-clock per kernel incl. launch/sync overhead)\n\n";

    for (auto const& [key, value] : _context) {
        stream << std::left << std::setw(32) << key << value << "\n";
    }
    if (!_context.empty()) {
        stream << "\n";
    }

    std::array<Entry, NumKernelCategories> totals;
    for (auto const& [total, entries] : std::views::zip(totals, _entriesByCategory)) {
        total = sumUp(entries);
    }
    auto grandTotal = Entry{};
    for (auto const& total : totals) {
        grandTotal.count += total.count;
        grandTotal.totalNanoseconds += total.totalNanoseconds;
    }

    // Overview of the categories
    stream << std::left << std::setw(56) << "category" << std::right << std::setw(10) << "calls" << std::setw(14) << "total [ms]" << std::setw(21) << "share"
           << "\n";
    for (auto const& [category, total] : std::views::zip(AllCategories, totals)) {
        stream << std::left << std::setw(56) << getCategoryName(category) << std::right << std::setw(10) << total.count << std::setw(14) << std::fixed
               << std::setprecision(3) << toMilliseconds(total.totalNanoseconds) << std::setw(20) << std::setprecision(1)
               << getShare(total.totalNanoseconds, grandTotal.totalNanoseconds) << "%\n";
    }
    stream << std::left << std::setw(56) << "total" << std::right << std::setw(10) << grandTotal.count << std::setw(14) << std::fixed << std::setprecision(3)
           << toMilliseconds(grandTotal.totalNanoseconds) << std::setw(20) << "100.0" << "%\n";

    // One ranking per category, with the shares relative to that category
    for (auto const& [category, total, entries] : std::views::zip(AllCategories, totals, _entriesByCategory)) {
        if (entries.empty()) {
            continue;
        }
        std::vector<std::pair<std::string, Entry>> sorted(entries.begin(), entries.end());
        std::ranges::sort(sorted, [](auto const& left, auto const& right) { return left.second.totalNanoseconds > right.second.totalNanoseconds; });

        stream << "\n" << getCategoryName(category) << "\n";
        stream << std::left << std::setw(4) << "#" << std::setw(52) << "kernel" << std::right << std::setw(10) << "calls" << std::setw(14) << "total [ms]"
               << std::setw(12) << "avg [us]" << std::setw(9) << "share" << std::setw(10) << "blocks" << std::setw(9) << "threads" << "\n";

        int rank = 1;
        for (auto const& [name, entry] : sorted) {
            auto avgUs = entry.count != 0 ? entry.totalNanoseconds / 1.0e3 / static_cast<double>(entry.count) : 0.0;
            stream << std::left << std::setw(4) << rank << std::setw(52) << name << std::right << std::setw(10) << entry.count << std::setw(14) << std::fixed
                   << std::setprecision(3) << toMilliseconds(entry.totalNanoseconds) << std::setw(12) << std::setprecision(1) << avgUs << std::setw(8)
                   << std::setprecision(1) << getShare(entry.totalNanoseconds, total.totalNanoseconds) << "%" << std::setw(9) << entry.numBlocks << std::setw(9)
                   << entry.threadsPerBlock << "\n";
            ++rank;
        }
        stream << std::left << std::setw(4) << "" << std::setw(52) << "total" << std::right << std::setw(10) << total.count << std::setw(14) << std::fixed
               << std::setprecision(3) << toMilliseconds(total.totalNanoseconds) << std::setw(12) << "" << std::setw(8) << "100.0" << "%\n";
    }

    return stream.str();
}
