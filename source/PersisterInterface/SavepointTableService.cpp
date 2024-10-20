#include "SavepointTableService.h"

#include <filesystem>
#include <fstream>

#include <boost/property_tree/json_parser.hpp>

#include "ParameterParser.h"


namespace
{
    void encodeDecode(boost::property_tree::ptree& tree, SavepointEntry& entry, ParserTask task)
    {
        JsonParser::encodeDecode(tree, entry.filename, std::string(), "filename", task);
        JsonParser::encodeDecode(tree, entry.state, 0, "state", task);
        JsonParser::encodeDecode(tree, entry.timestamp, std::string(), "timestamp", task);
        JsonParser::encodeDecode(tree, entry.name, std::string(), "name", task);
        JsonParser::encodeDecode(tree, entry.timestep, 0ull, "timestep", task);
    }

    void encodeDecode(boost::property_tree::ptree& tree, std::deque<SavepointEntry>& entries, ParserTask task)
    {
        if (ParserTask::Encode == task) {
            int index = 0;
            for (auto& entry : entries) {
                boost::property_tree::ptree subtree;
                encodeDecode(subtree, entry, task);
                tree.push_back(std::make_pair(std::to_string(index), subtree));
                ++index;
            }
        } else {
            entries.clear();
            for (auto& [key, subtree] : tree) {
                SavepointEntry entry;
                encodeDecode(subtree, entry, task);
                entries.emplace_back(entry);
            }
        }
    }
}

namespace
{
    bool hasWriteAccess(std::filesystem::path const& path)
    {
        std::filesystem::path tempFilePath = path / "temp_test_file.tmp";

        std::ofstream testFile(tempFilePath);
        if (testFile) {
            testFile.close();
            std::filesystem::remove(tempFilePath);
            return true;
        } else {
            return false;
        }
    }
}

auto SavepointTableService::loadFromFile(std::string const& filename) -> std::variant<SavepointTable, Error>
{
    try {
        auto directory = std::filesystem::path(filename).parent_path();

        // directory does not exist
        if (!std::filesystem::exists(directory)) {
            return Error{};
        }

        // no write access
        if (!hasWriteAccess(directory)) {
            return Error{};
        }

        // savepoint file does not exist
        if (!std::filesystem::exists(filename)) {
            return SavepointTable(filename, {});
        }

        std::ifstream stream(filename, std::ios::binary);
        if (!stream) {
            return Error{};
        }

        boost::property_tree::ptree tree;
        boost::property_tree::read_json(stream, tree);
        std::deque<SavepointEntry> entries;
        encodeDecode(tree, entries, ParserTask::Decode);

        return SavepointTable(filename, entries);
    } catch (...) {
        return Error{};
    }
}

bool SavepointTableService::insertEntry(SavepointTable& table, SavepointEntry const& entry) const
{
    table._entries.emplace_back(entry);
    return writeToFile(table);
}

bool SavepointTableService::updateEntry(SavepointTable& table, int row, SavepointEntry const& newEntry) const
{
    table._entries.at(row) = newEntry;
    return writeToFile(table);
}

bool SavepointTableService::writeToFile(SavepointTable& table) const
{
    try {
        std::ofstream stream(table.getFilename(), std::ios::binary);
        if (!stream) {
            return false;
        }
        boost::property_tree::ptree tree;
        encodeDecode(tree, table._entries, ParserTask::Encode);
        boost::property_tree::json_parser::write_json(stream, tree);
        return true;
    } catch (...) {
        return false;
    }
}
