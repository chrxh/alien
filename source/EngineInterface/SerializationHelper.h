#pragma once

#include <iostream>
#include <fstream>

#include <QRegularExpression>

#include "Definitions.h"

class SerializationHelper
{
public:
	template<typename EntityType>
	static bool loadFromFile(string const& filename, std::function<EntityType(string const&)> deserializer, EntityType& entity);
	static bool saveToFile(string const& filename, std::function<string()> serializer);
	static bool saveToFile(string const& filename, std::function<SerializedSimulation()> serializer);

private:
    static bool saveToFileIntern(std::string const& filename, std::string const& data);
};

template<typename EntityType>
inline bool SerializationHelper::loadFromFile(string const & filename, std::function<EntityType(string const&)> deserializer, EntityType & entity)
{
    string data;
    try {
        std::ifstream stream(filename, std::ios_base::in | std::ios_base::binary);

        size_t size;

        stream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
        data.resize(size);
        stream.read(&data[0], size);
        stream.close();

        if (stream.fail()) {
            return false;
        }
    } catch (std::exception const& e) {
        return false;
	}

	entity = deserializer(data);
	return true;
}

inline bool SerializationHelper::saveToFile(string const& filename, std::function<string()> serializer)
{
    return saveToFileIntern(filename, serializer());
}

inline bool SerializationHelper::saveToFile(string const& filename, std::function<SerializedSimulation()> serializer)
{
    SerializedSimulation const& data = serializer();
    if (!saveToFileIntern(filename, data.content)) {
        return false;
    }
    auto settingsFilename =
        QString::fromStdString(filename).replace(QRegularExpression("\\.\\w+$"), ".settings.json");
    if (!saveToFileIntern(settingsFilename.toStdString(), data.generalSettings)) {
        return false;
    }
    auto parametersFilename =
        QString::fromStdString(filename).replace(QRegularExpression("\\.\\w+$"), ".parameters.json");
    if (!saveToFileIntern(parametersFilename.toStdString(), data.simulationParameters)) {
        return false;
    }
    auto symbolsFilename = QString::fromStdString(filename).replace(QRegularExpression("\\.\\w+$"), ".symbols.json");
    if (!saveToFileIntern(symbolsFilename.toStdString(), data.symbolMap)) {
        return false;
    }
    return true;
}

inline bool SerializationHelper::saveToFileIntern(std::string const& filename, std::string const& data)
{
    try {
        std::ofstream stream(filename, std::ios_base::out | std::ios_base::binary);
        size_t dataSize = data.size();
        stream.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
        stream.write(&data[0], data.size());
        stream.close();
        if (stream.fail()) {
            return false;
        }

    } catch (...) {
        return false;
    }
    return true; 
}
