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
#include "EngineInterface/CellComputationCompiler.h"
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
    inline void serialize(Archive& ar, CellFeatureDescription& data)
    {
        ar(data.type);
        for (auto& c : data.mutableData) {
            ar(c);
        }
        for (auto& c : data.staticData) {
            ar(c);
        }
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
           data.cellFunctionInvocations,
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
        ar(data.id, data.pos, data.vel, data.energy, data.metadata);
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

/************************************************************************/
/* Version 3.3 support code                                             */
/************************************************************************/
namespace
{
    int const MaxCellStaticBytes = 48;
    int const MaxCellMutableBytes = 16;

    int getNumberOfInstructions(std::string const& compiledCode)
    {
        auto result = compiledCode.size();
        auto bytesPerInstruction = CellComputationCompiler::getBytesPerInstruction();
        while (result >= bytesPerInstruction) {
            if (compiledCode[result - bytesPerInstruction] == 0 && compiledCode[result - bytesPerInstruction + 1] == 0
                && compiledCode[result - bytesPerInstruction + 2] == 0) {
                result -= bytesPerInstruction;
            } else {
                break;
            }
        }
        return result / bytesPerInstruction;
    }

    StaticData convertStaticData(std::string const& data, Enums::CellFunction const& cellFunction)
    {
        StaticData result;
        auto dataSize = data.size();
        if (cellFunction == Enums::CellFunction_Computation) {
            result[0] = static_cast<char>(getNumberOfInstructions(data));
            int i = 1;
            for (; i < sizeof(result) && i - 1 < dataSize; ++i) {
                result[i] = data[i - 1];
            }
            for (; i < sizeof(result); ++i) {
                result[i] = 0;
            }
        } else {
            int i = 0;
            for (; i < sizeof(result) && i < dataSize; ++i) {
                result[i] = data[i];
            }
            for (; i < sizeof(result); ++i) {
                result[i] = 0;
            }
        }
        return result;
    }

    MutableData convertMutableData(std::string const& data)
    {
        MutableData result;
        auto dataSize = data.size();
        int i = 0;
        for (; i < sizeof(result) && i < dataSize; ++i) {
            result[i] = data[i];
        }
        for (; i < sizeof(result); ++i) {
            result[i] = 0;
        }
        return result;
    }

    struct DEPRECATED_CellFeatureDescription_3_3
    {
        std::string mutableData;
        std::string staticData;
        Enums::CellFunction type;
    };

    struct DEPRECATED_CellDescription_3_3
    {
        uint64_t id = 0;

        RealVector2D pos;
        RealVector2D vel;
        double energy;
        int maxConnections;
        std::vector<ConnectionDescription> connections;
        bool tokenBlocked;
        int tokenBranchNumber;
        CellMetadata metadata;
        DEPRECATED_CellFeatureDescription_3_3 cellFeature;
        std::vector<TokenDescription> tokens;
        int cellFunctionInvocations;
        bool barrier;

        CellDescription convert() const
        {
            CellDescription result;
            result.id = id;
            result.pos = pos;
            result.vel = vel;
            result.energy = energy;
            result.maxConnections = maxConnections;
            result.connections = connections;
            result.tokenBlocked = tokenBlocked;
            result.tokenBranchNumber = tokenBranchNumber;
            result.metadata = metadata;
            result.cellFeature.setType(cellFeature.type);
            result.cellFeature.staticData = convertStaticData(cellFeature.staticData, result.cellFeature.getType());
            result.cellFeature.mutableData = convertMutableData(cellFeature.mutableData);
            if (result.cellFeature.getType() == Enums::CellFunction_NeuralNet) {
                result.cellFeature.setType(Enums::CellFunction_Computation);
                result.cellFeature.staticData[0] = 0;
            }
            result.tokens = tokens;
            result.cellFunctionInvocations = cellFunctionInvocations;
            result.barrier = barrier;
            result.age = 0;
            return result;
        }
    };

    struct DEPRECATED_ClusterDescription_3_3
    {
        uint64_t id = 0;

        std::vector<DEPRECATED_CellDescription_3_3> cells;
        ClusterDescription convert() const
        {
            ClusterDescription result;
            result.id = id;
            for (auto const& cell : cells) {
                result.cells.emplace_back(cell.convert());
            }
            return result;
        }
    };

    struct DEPRECATED_ClusteredDataDescription_3_3
    {
        std::vector<DEPRECATED_ClusterDescription_3_3> clusters;
        std::vector<ParticleDescription> particles;

        ClusteredDataDescription convert() const
        {
            ClusteredDataDescription result;
            for (auto const& cluster : clusters) {
                result.clusters.emplace_back(cluster.convert());
            }
            result.particles = particles;
            return result;
        }
    };
}

namespace cereal
{
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_CellFeatureDescription_3_3& data)
    {
        ar(data.type, data.mutableData, data.staticData);
    }

    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_CellDescription_3_3& data)
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
           data.cellFunctionInvocations,
           data.barrier);
    }
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_ClusterDescription_3_3& data)
    {
        ar(data.id, data.cells);
    }
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_ClusteredDataDescription_3_3& data)
    {
        ar(data.clusters, data.particles);
    }
}

/************************************************************************/
/* Version 3.2 support code                                             */
/************************************************************************/
namespace
{
    struct DEPRECATED_CellFeatureDescription_3_2
    {
        std::string mutableData;
        std::string staticData;
        Enums::CellFunction type;
    };

    struct DEPRECATED_CellDescription_3_2
    {
        uint64_t id = 0;

        RealVector2D pos;
        RealVector2D vel;
        double energy;
        int maxConnections;
        std::vector<ConnectionDescription> connections;
        bool tokenBlocked;
        int tokenBranchNumber;
        CellMetadata metadata;
        DEPRECATED_CellFeatureDescription_3_2 cellFeature;
        std::vector<TokenDescription> tokens;
        int cellFunctionInvocations;
        bool barrier;

        CellDescription convert() const
        {
            CellDescription result;
            result.id = id;
            result.pos = pos;
            result.vel = vel;
            result.energy = energy;
            result.maxConnections = maxConnections;
            result.connections = connections;
            result.tokenBlocked = tokenBlocked;
            result.tokenBranchNumber = tokenBranchNumber;
            result.metadata = metadata;
            result.cellFeature.setType(cellFeature.type);
            result.cellFeature.staticData = convertStaticData(cellFeature.staticData, result.cellFeature.getType());
            result.cellFeature.mutableData = convertMutableData(cellFeature.mutableData);
            if (result.cellFeature.getType() == Enums::CellFunction_NeuralNet) {
                result.cellFeature.setType(Enums::CellFunction_Computation);
                result.cellFeature.staticData[0] = 0;
            }
            result.tokens = tokens;
            result.cellFunctionInvocations = cellFunctionInvocations;
            result.barrier = barrier;
            result.age = 0;
            return result;
        }
    };

    struct DEPRECATED_ClusterDescription_3_2
    {
        uint64_t id = 0;

        std::vector<DEPRECATED_CellDescription_3_2> cells;
        ClusterDescription convert() const
        {
            ClusterDescription result;
            result.id = id;
            for (auto const& cell : cells) {
                result.cells.emplace_back(cell.convert());
            }
            return result;
        }
    };

    struct DEPRECATED_ClusteredDataDescription_3_2
    {
        std::vector<DEPRECATED_ClusterDescription_3_2> clusters;
        std::vector<ParticleDescription> particles;

        ClusteredDataDescription convert() const
        {
            ClusteredDataDescription result;
            for (auto const& cluster : clusters) {
                result.clusters.emplace_back(cluster.convert());
            }
            result.particles = particles;
            return result;
        }
    };
}

namespace cereal
{
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_CellFeatureDescription_3_2& data)
    {
        ar(data.type, data.mutableData, data.staticData);
    }

    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_CellDescription_3_2& data)
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
           data.cellFunctionInvocations,
           data.barrier);
    }
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_ClusterDescription_3_2& data)
    {
        ar(data.id, data.cells);
    }
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_ClusteredDataDescription_3_2& data)
    {
        ar(data.clusters, data.particles);
    }
}

/************************************************************************/
/* Unknown version support code                                         */
/************************************************************************/
namespace
{
    struct DEPRECATED_CellFeatureDescription
    {
        std::string mutableData;
        std::string staticData;
        Enums::CellFunction type;
    };

    struct DEPRECATED_CellDescription
    {
        uint64_t id = 0;

        RealVector2D pos;
        RealVector2D vel;
        double energy;
        int maxConnections;
        std::vector<ConnectionDescription> connections;
        bool tokenBlocked;
        int tokenBranchNumber;
        CellMetadata metadata;
        DEPRECATED_CellFeatureDescription cellFeature;
        std::vector<TokenDescription> tokens;
        int tokenUsages;

        CellDescription convert() const
        {
            CellDescription result;
            result.id = id;
            result.pos = pos;
            result.vel = vel;
            result.energy = energy;
            result.maxConnections = maxConnections;
            result.connections = connections;
            result.tokenBlocked = tokenBlocked;
            result.tokenBranchNumber = tokenBranchNumber;
            result.metadata = metadata;
            result.cellFeature.setType(cellFeature.type);
            result.cellFeature.staticData = convertStaticData(cellFeature.staticData, result.cellFeature.getType());
            result.cellFeature.mutableData = convertMutableData(cellFeature.mutableData);
            if (result.cellFeature.getType() == Enums::CellFunction_NeuralNet) {
                result.cellFeature.setType(Enums::CellFunction_Computation);
                result.cellFeature.staticData[0] = 0;
            }
            result.tokens = tokens;
            result.cellFunctionInvocations = tokenUsages;
            result.barrier = false;
            return result;
        }
    };

    struct DEPRECATED_ClusterDescription
    {
        uint64_t id = 0;

        std::vector<DEPRECATED_CellDescription> cells;
        ClusterDescription convert() const
        {
            ClusterDescription result;
            result.id = id;
            for (auto const& cell : cells) {
                result.cells.emplace_back(cell.convert());
            }
            return result;
        }
    };

    struct DEPRECATED_ClusteredDataDescription
    {
        std::vector<DEPRECATED_ClusterDescription> clusters;
        std::vector<ParticleDescription> particles;

        ClusteredDataDescription convert() const
        {
            ClusteredDataDescription result;
            for (auto const& cluster : clusters) {
                result.clusters.emplace_back(cluster.convert());
            }
            result.particles = particles;
            return result;
        }
    };
}

namespace cereal
{
    template <class Archive>
    inline void serialize(Archive& ar, DEPRECATED_CellFeatureDescription& data)
    {
        ar(data.type, data.mutableData, data.staticData);
    }

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

bool Serializer::serializeSimulationToStrings(
    std::string& content,
    std::string& timestepAndSettings,
    std::string& symbolMap,
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
        {
            std::stringstream stream;
            serializeSymbolMap(data.symbolMap, stream);
            symbolMap = stream.str();
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool Serializer::deserializeSimulationFromStrings(
    DeserializedSimulation& data,
    std::string const& content,
    std::string const& timestepAndSettings,
    std::string const& symbolMap)
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
        {
            std::stringstream stream(symbolMap);
            deserializeSymbolMap(data.symbolMap, stream);
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

bool Serializer::deserializeSymbolsFromFile(SymbolMap& symbolMap, std::string const& filename)
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
    boost::property_tree::json_parser::write_json(stream, SettingsParser::encode(timestep, generalSettings));
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
        try {
            for (auto const& versionPart : versionParts) {
                std::stoi(versionPart);
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
    if (versionParts.major == 3 && versionParts.minor == 3) {
        DEPRECATED_ClusteredDataDescription_3_3 oldData;
        archive(oldData);
        data = oldData.convert();
    } else if (versionParts.major == 3 && versionParts.minor <= 2) {
        DEPRECATED_ClusteredDataDescription_3_2 oldData;
        archive(oldData);
        data = oldData.convert();
    } else {
        archive(data);
    }
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
    std::tie(timestep, settings) = SettingsParser::decodeTimestepAndSettings(tree);
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
