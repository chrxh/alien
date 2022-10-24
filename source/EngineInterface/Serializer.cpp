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
#include <cereal/types/variant.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/adaptors.hpp>
#include <zstr.hpp>

#include "Base/Resources.h"
#include "Descriptions.h"
#include "SimulationParameters.h"
#include "SettingsParser.h"

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
    inline void serialize(Archive& ar, CellMetadataDescription& data)
    {
        ar(data.name, data.description);
    }
    template <class Archive>
    inline void serialize(Archive& ar, ConnectionDescription& data)
    {
        ar(data.cellId, data.distance, data.angleFromPrevious);
    }

    template <class Archive>
    inline void serialize(Archive& ar, NeuronDescription& data)
    {
        ar(data.weights, data.bias);
    }
    template <class Archive>
    inline void serialize(Archive& ar, TransmitterDescription& data)
    {
    }
    template <class Archive>
    inline void serialize(Archive& ar, ConstructorDescription& data)
    {
        ar(data.mode, data.dna);
    }
    template <class Archive>
    inline void serialize(Archive& ar, SensorDescription& data)
    {
        ar(data.mode, data.color);
    }
    template <class Archive>
    inline void serialize(Archive& ar, NerveDescription& data)
    {
    }
    template <class Archive>
    inline void serialize(Archive& ar, AttackerDescription& data)
    {
    }
    template <class Archive>
    inline void serialize(Archive& ar, InjectorDescription& data)
    {
        ar(data.dna);
    }
    template <class Archive>
    inline void serialize(Archive& ar, MuscleDescription& data)
    {
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
           data.executionOrderNumber,
           data.color,
           data.underConstruction,
           data.inputBlocked,
           data.outputBlocked,
           data.cellFunction,
           data.barrier,
           data.age);
    }
    template <class Archive>
    inline void serialize(Archive& ar, ClusterDescription& data)
    {
        ar(data.id, data.cells);
    }
    template <class Archive>
    inline void serialize(Archive& ar, ParticleDescription& data)
    {
        ar(data.id, data.pos, data.vel, data.energy, data.color);
    }
    template <class Archive>
    inline void serialize(Archive& ar, ClusteredDataDescription& data)
    {
        ar(data.clusters, data.particles);
    }
}

/************************************************************************/
/* Support for old file formats                                         */
/************************************************************************/

bool Serializer::serializeSimulationToFiles(std::string const& filename, DeserializedSimulation const& data)
{
    try {

        std::filesystem::path settingsFilename(filename);
        settingsFilename.replace_extension(std::filesystem::path(".settings.json"));

        std::filesystem::path symbolsFilename(filename);
        symbolsFilename.replace_extension(std::filesystem::path(".symbols.json"));

        {
            zstr::ofstream stream(filename, std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeDataDescription(data.content, stream);
        }
        {
            std::ofstream stream(settingsFilename.string(), std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeTimestepAndSettings(data.timestep, data.settings, stream);
            stream.close();
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::deserializeSimulationFromFiles(DeserializedSimulation& data, std::string const& filename)
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
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::serializeSimulationToStrings(
    std::string& content,
    std::string& timestepAndSettings,
    DeserializedSimulation const& data)
{
    try {
        {
            std::stringstream stdStream;
            zstr::ostream stream(stdStream, std::ios::binary);
            if (!stream) {
                return false;
            }
            serializeDataDescription(data.content, stream);
            stream.flush();
            content = stdStream.str();
        }
        {
            std::stringstream stream;
            serializeTimestepAndSettings(data.timestep, data.settings, stream);
            timestepAndSettings = stream.str();
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::deserializeSimulationFromStrings(
    DeserializedSimulation& data,
    std::string const& content,
    std::string const& timestepAndSettings)
{
    try {
        {
            std::stringstream stdStream(content);
            zstr::istream stream(stdStream, std::ios::binary);
            if (!stream) {
                return false;
            }
            deserializeDataDescription(data.content, stream);
        }
        {
            std::stringstream stream(timestepAndSettings);
            deserializeTimestepAndSettings(data.timestep, data.settings, stream);
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

bool Serializer::deserializeContentFromFile(ClusteredDataDescription& content, std::string const& filename)
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

void Serializer::serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream)
{
    cereal::PortableBinaryOutputArchive archive(stream);
    archive(Const::ProgramVersion);
    archive(data);
}

void Serializer::serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings, std::ostream& stream)
{
    boost::property_tree::json_parser::write_json(stream, SettingsParser::encode(timestep, generalSettings));
}

bool Serializer::deserializeDataDescription(ClusteredDataDescription& data, std::string const& filename)
{
    zstr::ifstream stream(filename, std::ios::binary);
    if (!stream) {
        return false;
    }
    deserializeDataDescription(data, stream);
    return true;
}

namespace
{
    bool isVersionValid(std::string const& s)
    {
        std::vector<std::string> versionParts;
        boost::split(versionParts, s, boost::is_any_of("."));
        try {
            for (auto const& versionPart : versionParts) {
                int result = std::stoi(versionPart);
            }
        } catch (...) {
            return false;
        }
        return versionParts.size() == 3;
    }
    struct VersionParts
    {
        int major;
        int minor;
        int patch;
    };
    VersionParts getVersionParts(std::string const& s)
    {
        std::vector<std::string> versionParts;
        boost::split(versionParts, s, boost::is_any_of("."));
        return {std::stoi(versionParts.at(0)), std::stoi(versionParts.at(1)), std::stoi(versionParts.at(2))};
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
    auto versionParts = getVersionParts(version);
    if (versionParts.major == 4 && versionParts.minor == 0) {
        archive(data);
    } else {
        throw std::runtime_error("Version not supported.");
    }
}

void Serializer::deserializeTimestepAndSettings(uint64_t& timestep, Settings& settings, std::istream& stream)
{
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(stream, tree);
    std::tie(timestep, settings) = SettingsParser::decodeTimestepAndSettings(tree);
}

