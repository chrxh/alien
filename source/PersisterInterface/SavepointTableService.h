#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <variant>

#include "Base/Singleton.h"

#include "Definitions.h"
#include "SavepointTable.h"

class SavepointTableService
{
    MAKE_SINGLETON(SavepointTableService);
public:
    struct Error {};
    std::variant<SavepointTable, Error> loadFromFile(std::string const& filename);

    bool insertEntry(SavepointTable& table, SavepointEntry const& entry) const;
    bool updateEntry(SavepointTable& table, int row, SavepointEntry const& newEntry) const;

private:
    bool writeToFile(SavepointTable& table) const;
};