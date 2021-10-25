#include "Serializer.h"

#include <sstream>
#include <regex>

#include <boost/property_tree/json_parser.hpp>

#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/range/adaptors.hpp>

#include "Base/ServiceLocator.h"

#include "Descriptions.h"
#include "ChangeDescriptions.h"
#include "SimulationParameters.h"
#include "Parser.h"

using namespace std;
using namespace boost;


namespace boost {
	namespace serialization {

        template<class Archive>
        inline void save(Archive& ar, CellFeatureDescription const& data, const unsigned int /*version*/)
        {
            ar << data.getType() << data.volatileData << data.constData;
        }
        template<class Archive>
        inline void load(Archive& ar, CellFeatureDescription& data, const unsigned int /*version*/)
        {
            Enums::CellFunction::Type type;
            ar >> type >> data.volatileData >> data.constData;
            data.setType(type);
        }
        template<class Archive>
        inline void serialize(Archive & ar, CellFeatureDescription& data, const unsigned int version)
        {
            boost::serialization::split_free(ar, data, version);
        }

/*
		template<class Archive>
		inline void serialize(Archive & ar, TokenDescription& data, const unsigned int / *version* /)
		{
			ar & data.energy & data.data;
		}
*/
        template <class Archive>
        inline void save(Archive& ar, TokenDescription const& data, const unsigned int /*version*/)
        {
            ar<< data.energy<< *data.data;
        }
        template <class Archive>
        inline void load(Archive& ar, TokenDescription& data, const unsigned int /*version*/)
        {
            std::string tokenData;
            ar >> data.energy >> tokenData;
            data.setData(tokenData);
        }
        template <class Archive>
        inline void serialize(Archive& ar, TokenDescription& data, const unsigned int version)
        {
            boost::serialization::split_free(ar, data, version);
        }
        template <class Archive>
		inline void serialize(Archive & ar, CellMetadata& data, const unsigned int /*version*/)
		{
			ar & data.computerSourcecode & data.name & data.description & data.color;
		}
        template <class Archive>
        inline void serialize(Archive& ar, ConnectionDescription& data, const unsigned int /*version*/)
        {
            ar& data.cellId& data.distance& data.angleFromPrevious;
        }
        template <class Archive>
        inline void serialize(Archive& ar, CellDescription& data, const unsigned int /*version */)
        {
            ar& data.id& data.pos& data.vel& data.energy& data.maxConnections& data.connections;
            ar& data.tokenBlocked& data.tokenBranchNumber& data.metadata& data.cellFeature;
            ar& data.tokens& data.tokenUsages;
        }

		template<class Archive>
		inline void serialize(Archive & ar, ClusterDescription& data, const unsigned int /*version*/)
		{
            ar& data.id& data.cells;
        }

        template <class Archive>
		inline void serialize(Archive & ar, ParticleMetadata& data, const unsigned int /*version*/)
		{
			ar & data.color;
		}
		template<class Archive>
		inline void serialize(Archive & ar, ParticleDescription& data, const unsigned int /*version*/)
		{
			ar & data.id & data.pos & data.vel & data.energy & data.metadata;
		}

        template <class Archive>
        inline void serialize(Archive& ar, DataDescription& data, const unsigned int /*version*/)
        {
            ar & data.clusters & data.particles;
        }

		template<class Archive>
		inline void serialize(Archive & ar, IntVector2D& data, const unsigned int /*version*/)
		{
			ar & data.x & data.y;
		}
        template <class Archive>
        inline void serialize(Archive& ar, RealVector2D& data, const unsigned int /*version*/)
        {
            ar& data.x& data.y;
        }
    }
}

bool _Serializer::loadSimulationDataFromFile(string const& filename, SerializedSimulation& data)
{
    try {
        std::regex fileEndingExpr("\\.\\w+$");
        if (!std::regex_search(filename, fileEndingExpr)) {
            return false;
        }
        auto settingsFilename = std::regex_replace(filename, fileEndingExpr, ".settings.json");
        auto symbolsFilename = std::regex_replace(filename, fileEndingExpr, ".symbols.json");

        if (!loadDataFromFile(filename, data.content)) {
            return false;
        }
        if (!loadDataFromFile(settingsFilename, data.timestepAndSettings)) {
            return false;
        }
        if (!loadDataFromFile(symbolsFilename, data.symbolMap)) {
            return false;
        }
    } catch (std::exception const& e) {
        throw std::exception(("An error occurred while loading the file " + filename + ": " + e.what()).c_str());
    }

    return true;
}

bool _Serializer::saveSimulationDataToFile(string const& filename, SerializedSimulation& data)
{
    try {
        std::regex fileEndingExpr("\\.\\w+$");
        if (!std::regex_search(filename, fileEndingExpr)) {
            return false;
        }
        auto settingsFilename = std::regex_replace(filename, fileEndingExpr, ".settings.json");
        auto symbolsFilename = std::regex_replace(filename, fileEndingExpr, ".symbols.json");

        if (!saveDataToFile(filename, data.content)) {
            return false;
        }
        if (!saveDataToFile(settingsFilename, data.timestepAndSettings)) {
            return false;
        }
        if (!saveDataToFile(symbolsFilename, data.symbolMap)) {
            return false;
        }
    } catch (std::exception const& e) {
        throw std::exception(
            ("An error occurred while saving the file " + filename + ": " + e.what()).c_str());
    }

    return true;
}

SerializedSimulation _Serializer::serializeSimulation(DeserializedSimulation const& data)
{
    try {
        return {
            serializeTimestepAndSettings(data.timestep, data.settings),
            serializeSymbolMap(data.symbolMap),
            serializeDataDescription(data.content)};
    } catch (std::exception const& e) {
        throw std::exception((std::string("An error occurred while serializing simulation data: ") + e.what()).c_str());
    }
}

DeserializedSimulation _Serializer::deserializeSimulation(SerializedSimulation const& data)
{
    try {
        auto [timestep, settings] = deserializeTimestepAndSettings(data.timestepAndSettings);
        return {timestep, settings, deserializeSymbolMap(data.symbolMap), deserializeDataDescription(data.content)};
    } catch (std::exception const& e) {
        throw std::exception((std::string("An error occurred while deserializing simulation data: ") + e.what()).c_str());
    }
}

string _Serializer::serializeSymbolMap(SymbolMap const symbols) const
{
    boost::property_tree::ptree tree;
    for (auto const& [key, value] : symbols) {
        tree.add(key, value);
    }

    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, tree);
    return ss.str();
}

SymbolMap _Serializer::deserializeSymbolMap(string const& data)
{
    std::stringstream ss;
    ss << data;
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(ss, tree);
    std::map<std::string, std::string> result;
    for (auto const& [key, value] : tree) {
        result.emplace(key.data(), value.data());
    }
    return result;
}

string _Serializer::serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings) const
{
    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, Parser::encode(timestep, generalSettings));
    return ss.str();
}

std::pair<uint64_t, Settings> _Serializer::deserializeTimestepAndSettings(
    std::string const& data) const
{
    std::stringstream ss;
    ss << data;
    boost::property_tree::ptree tree;
    boost::property_tree::read_json(ss, tree);
    return Parser::decodeTimestepAndSettings(tree);
}

string _Serializer::serializeDataDescription(DataDescription const& desc) const
{
    ostringstream stream;
    boost::archive::binary_oarchive archive(stream);

    archive << desc;
    return stream.str();
}

DataDescription _Serializer::deserializeDataDescription(string const& data)
{
    istringstream stream(data);
    boost::archive::binary_iarchive ia(stream);

    DataDescription result;
    ia >> result;
    return result;
}

bool _Serializer::loadDataFromFile(std::string const& filename, std::string& data)
{
    ifstream stream(filename, ios::binary);
    if (!stream) {
        return false;
    }
    data = std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    stream.close();
    return true;
}

bool _Serializer::saveDataToFile(std::string const& filename, std::string const& data)
{
    ofstream stream(filename, ios::binary);
    if (!stream) {
        return false;
    }
    stream << data;
    stream.close();
    return true;
}
