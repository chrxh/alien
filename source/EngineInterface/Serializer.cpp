#include "Serializer.h"

#include <sstream>
#include <stdexcept>
#include <filesystem>

#include <optional>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/adaptors.hpp>
#include <zstr.hpp>

#include "Base/Resources.h"
#include "Descriptions.h"
#include "SimulationParameters.h"
#include "Parser.h"

namespace cereal
{

    template <class Archive>
    inline void serialize(Archive& ar, IntVector2D& data)
    {
        ar(data.x, data.y);
    }
    template <class Archive>
    inline void serialize(Archive& ar, RealVector2D& data)
    {
        ar(data.x, data.y);
    }

    template <class Archive>
    inline void save(Archive& ar, CellFeatureDescription const& data)
    {
        ar(data.getType(), data.volatileData, data.constData);
    }
    template <class Archive>
    inline void load(Archive& ar, CellFeatureDescription& data)
    {
        Enums::CellFunction type;
        ar(type, data.volatileData, data.constData);
        data.setType(type);
    }

    template <class Archive>
    inline void serialize(Archive& ar, CellMetadata& data)
    {
        ar(data.computerSourcecode, data.name, data.description, data.color);
    }
    template <class Archive>
    inline void serialize(Archive& ar, ConnectionDescription& data)
    {
        ar(data.cellId, data.distance, data.angleFromPrevious);
    }

    template <class Archive>
    inline void serialize(Archive& ar, ParticleMetadata& data)
    {
        ar(data.color);
    }
    template <class Archive>
    inline void serialize(Archive& ar, TokenDescription& data)
    {
        ar(data.energy, data.data);
    }
    template <class Archive>
    inline void serialize(Archive& ar, CellDescription& data)
    {
        ar(data.id,
           data.pos,
           data.vel,
           data.energy,
           data.maxConnections,
           data.connections,
           data.tokenBlocked,
           data.tokenBranchNumber,
           data.metadata,
           data.cellFeature,
           data.tokens,
           data.tokenUsages,
           data.barrier);
    }
    template <class Archive>
    inline void serialize(Archive& ar, ClusterDescription& data)
    {
        ar(data.id, data.cells);
    }
    template <class Archive>
    inline void serialize(Archive& ar, ParticleDescription& data)
    {
        ar(data.id, data.pos, data.vel, data.energy, data.metadata);
    }
    template <class Archive>
    inline void serialize(Archive& ar, ClusteredDataDescription& data)
    {
        ar(data.clusters, data.particles);
    }

    //DEPRECATED
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_CellDescription& data)
    {
        ar(data.id,
           data.pos,
           data.vel,
           data.energy,
           data.maxConnections,
           data.connections,
           data.tokenBlocked,
           data.tokenBranchNumber,
           data.metadata,
           data.cellFeature,
           data.tokens,
           data.tokenUsages);
    }
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_ClusterDescription& data)
    {
        ar(data.id, data.cells);
    }
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_ClusteredDataDescription& data)
    {
        ar(data.clusters, data.particles);
    }

}

bool Serializer::serializeSimulationToFile(std::string const& filename, DeserializedSimulation const& data)
{
    try {

        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));

        std::filesystem::path symbolsFilename(filename);
        symbolsFilename.replace_extension(std::filesystem::path(".symbols.json"));

        {
            zstr::ofstream fileStream(filename, std::ios::binary);
            if (!fileStream) {
                return false;
            }
            serializeDataDescription(data.content, fileStream);
        }
        {
            std::ofstream stream(settingsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeTimestepAndSettings(data.timestep, data.settings, stream);
            stream.close();
        }
        {
            std::ofstream stream(symbolsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeSymbolMap(data.symbolMap, stream);
            stream.close();
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::deserializeSimulationFromFile(std::string const& filename, DeserializedSimulation& data)
{
    try {
        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));

        std::filesystem::path symbolsFilename(filename);
        symbolsFilename.replace_extension(std::filesystem::path(".symbols.json"));

        if (!deserializeDataDescription(data.content, filename)) {
            return false;
        }
        {
            std::ifstream stream(settingsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeTimestepAndSettings(data.timestep, data.settings, stream);
            stream.close();
        }
        {
            std::ifstream stream(symbolsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeSymbolMap(data.symbolMap, stream);
            stream.close();
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content)
{
    try {
        zstr::ofstream fileStream(filename, std::ios::binary);
        if (!fileStream) {
            return false;
        }
        serializeDataDescription(content, fileStream);

        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::deserializeContentFromFile(std::string const& filename, ClusteredDataDescription& content)
{
    try {
        if (!deserializeDataDescription(content, filename)) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::serializeSymbolsToFile(std::string const& filename, SymbolMap const& symbolMap)
{
    try {
        std::ofstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        serializeSymbolMap(symbolMap, stream);
        stream.close();
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::deserializeSymbolsFromFile(std::string const& filename, SymbolMap& symbolMap)
{
    try {
        std::ifstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        deserializeSymbolMap(symbolMap, stream);
        stream.close();
        return true;
    } catch (...) {
        return false;
    }
}

void Serializer::serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream)
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(Const::ProgramVersion);
    archive(data);
}

void Serializer::serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings, std::ostream& stream)
{
    boost::property_tree::json_parser::write_json(stream, Parser::encode(timestep, generalSettings));
}

void Serializer::serializeSymbolMap(SymbolMap const symbols, std::ostream& stream)
{
    boost::property_tree::ptree tree;
    for (auto const& [key, value] : symbols) {
        tree.add(key, value);
    }

    boost::property_tree::json_parser::write_json(stream, tree);
}

bool Serializer::deserializeDataDescription(ClusteredDataDescription& data, std::string const& filename)
{
    try {
        zstr::ifstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        deserializeDataDescription(data, stream);
    } catch (...) {

        //try reading old unversioned data
        zstr::ifstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        DEPREACATED_deserializeDataDescription(data, stream);
    }
    return true;
}

namespace
{
    bool isVersionValid(std::string const& s)
    {
        std::vector<std::string> versionParts;
        boost::split(versionParts, s, boost::is_any_of("."));
        return versionParts.size() == 3;
    }
}

void Serializer::deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream)
{
    cereal::PortableBinaryInputArchive archive(stream);
    std::string version;
    archive(version);
    if (!isVersionValid(version)) {
        throw std::runtime_error("No version detected.");
    }
    archive(data);
}

void Serializer::DEPREACATED_deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream)
{
    DEPRECATED_ClusteredDataDescription DEPRECATED_data;
    cereal::PortableBinaryInputArchive archive(stream);
    archive(DEPRECATED_data);
    data = DEPRECATED_data.convert();
}

void Serializer::deserializeTimestepAndSettings(uint64_t& timestep, Settings& settings, std::istream& stream)
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    std::tie(timestep, settings) = Parser::decodeTimestepAndSettings(tree);
}

void Serializer::deserializeSymbolMap(SymbolMap& symbolMap, std::istream& stream)
{
    symbolMap.clear();
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    for (auto const& [key, value] : tree) {
        symbolMap.emplace(key.data(), value.data());
    }
}
