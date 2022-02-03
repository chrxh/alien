#include "Serializer.h"

#include <sstream>
#include <regex>
#include <stdexcept>

#include <optional>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/adaptors.hpp>

#include <imgui/stb_compression.h>

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
           data.tokenUsages);
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
}

bool _Serializer::serializeSimulationToFile(std::string const& filename, DeserializedSimulation const& data)
{
    try {
        std::regex fileEndingExpr("\\.\\w+$");
        if (!std::regex_search(filename, fileEndingExpr)) {
            return false;
        }
        auto settingsFilename = std::regex_replace(filename, fileEndingExpr, ".settings.json");
        auto symbolsFilename = std::regex_replace(filename, fileEndingExpr, ".symbols.json");

        {
            std::stringstream stringStream;
            serializeDataDescription(data.content, stringStream);

            std::ofstream fileStream(filename, std::ios::binary);
            if (!fileStream) {
                return false;
            }
            compress(stringStream.str(), fileStream);
            fileStream.close();
        }
        {
            std::ofstream stream(settingsFilename, std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeTimestepAndSettings(data.timestep, data.settings, stream);
            stream.close();
        }
        {
            std::ofstream stream(symbolsFilename, std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeSymbolMap(data.symbolMap, stream);
            stream.close();
        }
        return true;
    } catch (std::exception const& e) {
        throw std::runtime_error(std::string("An error occurred while serializing simulation data: ") + e.what());
    }
}

bool _Serializer::deserializeSimulationFromFile(std::string const& filename, DeserializedSimulation& data)
{
    try {
        std::regex fileEndingExpr("\\.\\w+$");
        if (!std::regex_search(filename, fileEndingExpr)) {
            return false;
        }
        auto settingsFilename = std::regex_replace(filename, fileEndingExpr, ".settings.json");
        auto symbolsFilename = std::regex_replace(filename, fileEndingExpr, ".symbols.json");

        {
            std::ifstream stream(filename, std::ios::binary);
            if (!stream) {
                return false;
            }
            std::string compressedData((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

            std::stringstream stringStream;
            decompress(std::move(compressedData), stringStream);
            stringStream.seekg(0, std::ios::beg);
            deserializeDataDescription(data.content, stringStream);
            stream.close();

        }
        {
            std::ifstream stream(settingsFilename, std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeTimestepAndSettings(data.timestep, data.settings, stream);
            stream.close();
        }
        {
            std::ifstream stream(symbolsFilename, std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeSymbolMap(data.symbolMap, stream);
            stream.close();
        }
        return true;
    } catch (std::exception const& e) {
        throw std::runtime_error("An error occurred while loading the file " + filename + ": " + e.what());
    }
}

bool _Serializer::serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content)
{
    try {
        std::stringstream stringStream;
        serializeDataDescription(content, stringStream);

        std::ofstream fileStream(filename, std::ios::binary);
        if (!fileStream) {
            return false;
        }
        compress(stringStream.str(), fileStream);
        fileStream.close();

        return true;
    } catch (std::exception const& e) {
        throw std::runtime_error(std::string("An error occurred while serializing simulation data: ") + e.what());
    }
}

bool _Serializer::deserializeContentFromFile(std::string const& filename, ClusteredDataDescription& content)
{
    try {
        std::ifstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        std::string compressedData((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

        std::stringstream stringStream;
        decompress(std::move(compressedData), stringStream);
        stringStream.seekg(0, std::ios::beg);
        deserializeDataDescription(content, stringStream);
        stream.close();

        return true;
    } catch (std::exception const& e) {
        throw std::runtime_error("An error occurred while loading the file " + filename + ": " + e.what());
    }
}

bool _Serializer::serializeSymbolsToFile(std::string const& filename, SymbolMap const& symbolMap)
{
    try {
        std::ofstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        serializeSymbolMap(symbolMap, stream);
        stream.close();
        return true;
    } catch (std::exception const& e) {
        throw std::runtime_error(std::string("An error occurred while serializing simulation data: ") + e.what());
    }
}

bool _Serializer::deserializeSymbolsFromFile(std::string const& filename, SymbolMap& symbolMap)
{
    try {
        std::ifstream stream(filename, std::ios::binary);
        if (!stream) {
            return false;
        }
        deserializeSymbolMap(symbolMap, stream);
        stream.close();
        return true;
    } catch (std::exception const& e) {
        throw std::runtime_error("An error occurred while loading the file " + filename + ": " + e.what());
    }
}

void _Serializer::serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream) const
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(data);
}

void _Serializer::serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings, std::ostream& stream)
    const
{
    boost::property_tree::json_parser::write_json(stream, Parser::encode(timestep, generalSettings));
}

void _Serializer::serializeSymbolMap(SymbolMap const symbols, std::ostream& stream) const
{
    boost::property_tree::ptree tree;
    for (auto const& [key, value] : symbols) {
        tree.add(key, value);
    }

    boost::property_tree::json_parser::write_json(stream, tree);
}

void _Serializer::deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream) const
{
    cereal::PortableBinaryInputArchive archive(stream);
    archive(data);

    if (data.clusters.empty() && data.particles.empty()) {
        throw std::runtime_error("no data found");
    }
}

void _Serializer::deserializeTimestepAndSettings(uint64_t& timestep, Settings& settings, std::istream& stream) const
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    std::tie(timestep, settings) = Parser::decodeTimestepAndSettings(tree);
}

void _Serializer::deserializeSymbolMap(SymbolMap& symbolMap, std::istream& stream)
{
    symbolMap.clear();
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    for (auto const& [key, value] : tree) {
        symbolMap.emplace(key.data(), value.data());
    }
}

void _Serializer::compress(std::string&& uncompressedData, std::ostream& stream)
{
    int maxlen = uncompressedData.size() + 512 + (uncompressedData.size() >> 2) + sizeof(int);  // total guess
    char* compressedData = new char[maxlen];
    auto compressedDataLength =
        stb_compress(reinterpret_cast<stb_uchar*>(compressedData), reinterpret_cast<stb_uchar*>(uncompressedData.data()), uncompressedData.size());

    stream.write(compressedData, compressedDataLength);
    delete[] compressedData;
}

void _Serializer::decompress(std::string&& compressedData, std::ostream& stream)
{
    const unsigned int decompressedSize = stb_decompress_length(reinterpret_cast<unsigned char const*>(compressedData.c_str()));
    char* decompressedData = new char[decompressedSize];
    stb_decompress(
        reinterpret_cast<unsigned char*>(decompressedData),
        reinterpret_cast<unsigned char const*>(compressedData.c_str()),
        static_cast<unsigned int>(compressedData.size()));
    stream.write(decompressedData, decompressedSize);
    delete[] decompressedData;
}
