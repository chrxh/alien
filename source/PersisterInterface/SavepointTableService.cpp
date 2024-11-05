#include "SavepointTableService.h"

#include <filesystem>
#include <fstream>
#include <ranges>

#include <boost/property_tree/json_parser.hpp>

#include "ParameterParser.h"
#include "SerializerService.h"

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
        SavepointTable result(filename, std::deque<SavepointEntry>());
        encodeDecode(tree, result, ParserTask::Decode);

        return result;
    } catch (...) {
        return Error{};
    }
}

std::vector<SavepointEntry> SavepointTableService::truncate(SavepointTable& table, int newSize) const
{
    std::vector<SavepointEntry> result;

    auto& entries = table._entries;
    if (entries.size() < newSize) {
        return result;
    }

    for (auto const& entry : entries | std::views::drop(newSize)) {
        if (entry->state == SavepointState_Persisted) {
            SerializerService::get().deleteSimulation(entry->filename);
        } else {
            result.emplace_back(entry);
        }
    }

    entries.erase(entries.begin() + newSize, entries.end());
    updateFile(table);
    return result;
}

void SavepointTableService::insertEntryAtFront(SavepointTable& table, SavepointEntry const& entry) const
{
    table._entries.emplace_front(entry);
    ++table._sequenceNumber;
    updateFile(table);
}

void SavepointTableService::updateEntry(SavepointTable& table, int row, SavepointEntry const& newEntry) const
{
    table._entries.at(row) = newEntry;
    updateFile(table);
}

void SavepointTableService::deleteEntry(SavepointTable& table, SavepointEntry const& entry) const
{
    if (entry->state == SavepointState_Persisted) {
        SerializerService::get().deleteSimulation(entry->filename);
    }

    table._entries.erase(std::remove(table._entries.begin(), table._entries.end(), entry), table._entries.end());
    updateFile(table);
}

void SavepointTableService::updateFile(SavepointTable& table) const
{
    try {
        std::ofstream stream(table.getFilename(), std::ios::binary);
        if (!stream) {
            throw std::runtime_error("Could not access save point table file: " + table.getFilename().string());
        }
        boost::property_tree::ptree tree;
        encodeDecode(tree, table, ParserTask::Encode);
        boost::property_tree::json_parser::write_json(stream, tree);
    } catch (std::exception const& e) {
        throw std::runtime_error(std::string("The following error occurred: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Unknown error.");
    }
}

void SavepointTableService::encodeDecode(boost::property_tree::ptree& tree, SavepointTable& table, ParserTask task) const
{
    JsonParser::encodeDecode(tree, table._sequenceNumber, 0, "sequence number", task);
    encodeDecode(tree, table._entries, task);
}

void SavepointTableService::encodeDecode(boost::property_tree::ptree& tree, std::deque<SavepointEntry>& entries, ParserTask task) const
{
    if (ParserTask::Encode == task) {
        boost::property_tree::ptree subtree;

        int index = 0;
        for (auto& entry : entries) {
            boost::property_tree::ptree subsubtree;
            encodeDecode(subsubtree, entry, task);
            subtree.push_back(std::make_pair(std::to_string(index), subsubtree));
            ++index;
        }
        tree.push_back(std::make_pair("entries", subtree));
    } else {
        entries.clear();
        for (auto& [key, subtree] : tree.get_child("entries")) {
            SavepointEntry entry = std::make_shared<_SavepointEntry>();
            encodeDecode(subtree, entry, task);
            entries.emplace_back(entry);
        }
    }
}

void SavepointTableService::encodeDecode(boost::property_tree::ptree& tree, SavepointEntry& entry, ParserTask task) const
{
    encodeDecode(tree, entry->filename, "filename", task);
    JsonParser::encodeDecode(tree, entry->state, 0, "state", task);
    JsonParser::encodeDecode(tree, entry->timestamp, std::string(), "timestamp", task);
    JsonParser::encodeDecode(tree, entry->name, std::string(), "name", task);
    JsonParser::encodeDecode(tree, entry->timestep, uint64_t(0), "timestep", task);
    JsonParser::encodeDecode(tree, entry->peak, std::string(), "peak", task);
    JsonParser::encodeDecode(tree, entry->peakType, std::string(), "peak type", task);
}

void SavepointTableService::encodeDecode(boost::property_tree::ptree& tree, std::filesystem::path& path, std::string const& node, ParserTask task) const
{
    if (task == ParserTask::Encode) {
        auto pathString = path.string();
        JsonParser::encodeDecode(tree, pathString, std::string(), node, task);
    } else {
        std::string pathString;
        JsonParser::encodeDecode(tree, pathString, std::string(), node, task);
        path = pathString;
    }
}
