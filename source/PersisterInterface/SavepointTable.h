#pragma once

#include <cstdint>
#include <deque>
#include <string>

#include "Definitions.h"

using SavepointState = int;
enum SavepointState_
{
    SavepointState_InQueue,
    SavepointState_InProgress,
    SavepointState_Persisted,
    SavepointState_Error
};

struct SavepointEntry
{
    std::string id;
    std::string filename;
    SavepointState state = SavepointState_InQueue;
    std::string timestamp;
    std::string name;
    uint64_t timestep = 0;
};

class SavepointTable
{
    friend SavepointTableService;

public:
    SavepointEntry const& at(int index) const { return _entries.at(index); }
    int getSize() const { return toInt(_entries.size()); }
    std::string const& getFilename() const { return _filename; }

private:
    SavepointTable(std::string const& filename, std::deque<SavepointEntry> const& entries)
        : _filename(filename)
        , _entries(entries)
    {}

    std::string _filename;
    std::deque<SavepointEntry> _entries;
};

