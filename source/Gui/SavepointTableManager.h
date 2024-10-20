#pragma once

#include <cstdint>
#include <deque>
#include <string>

enum class SavepointState
{
    InQueue,
    InProgress,
    Persisted,
    Error
};

struct SavepointEntry
{
    std::string id;
    std::string filename;
    SavepointState state = SavepointState::InQueue;
    std::string timestamp;
    std::string name;
    uint64_t timestep = 0;
};

using SavepointTable = std::deque<SavepointEntry>;

class SavepointTableManager
{
public:
    void loadFromFile(std::string const& filename);

    SavepointTable const& getTable() const;

private:
    SavepointTable _table;
};