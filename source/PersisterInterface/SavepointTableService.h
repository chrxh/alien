#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <variant>

#include <boost/property_tree/ptree_fwd.hpp>

#include "Base/JsonParser.h"
#include "Base/Singleton.h"

#include "Definitions.h"
#include "SavepointTable.h"

class SavepointTableService
{
    MAKE_SINGLETON(SavepointTableService);
public:
    struct Error {};
    std::variant<SavepointTable, Error> loadFromFile(std::string const& filename);

    std::vector<SavepointEntry> truncate(SavepointTable& table, int newSize) const; //returns non-persistent entries
    void insertEntryAtFront(SavepointTable& table, SavepointEntry const& entry) const;
    void updateEntry(SavepointTable& table, int row, SavepointEntry const& newEntry) const;

private:
    void writeToFile(SavepointTable& table) const;
    void encodeDecode(boost::property_tree::ptree& tree, SavepointTable& table, ParserTask task) const;
    void encodeDecode(boost::property_tree::ptree& tree, std::deque<SavepointEntry>& entries, ParserTask task) const;
    void encodeDecode(boost::property_tree::ptree& tree, SavepointEntry& entry, ParserTask task) const;
    void encodeDecode(boost::property_tree::ptree& tree, std::filesystem::path& path, std::string const& node, ParserTask task) const;
};