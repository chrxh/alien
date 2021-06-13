#pragma once

#include <iostream>
#include <fstream>

#include <QFile>
#include <QRegularExpression>

#include "Definitions.h"

class SerializationHelper
{
public:
	template<typename EntityType>
	static bool loadFromFile(string const& filename, std::function<EntityType(string const&)> deserializer, EntityType& entity);
    static bool loadFromFile(
        string const& filename,
        std::function<SimulationController*(SerializedSimulation const&)> deserializer,
        SimulationController*& entity);
    static bool saveToFile(string const& filename, std::function<string()> serializer);
	static bool saveToFile(string const& filename, std::function<SerializedSimulation()> serializer);

private:
    static bool loadFromFileIntern(std::string const& filename, std::string& data);
    static bool saveToFileIntern(std::string const& filename, std::string const& data);
};

template<typename EntityType>
inline bool SerializationHelper::loadFromFile(string const & filename, std::function<EntityType(string const&)> deserializer, EntityType & entity)
{
    string data;
    if (!loadFromFileIntern(filename, data)) {
        return false;
    }
	entity = deserializer(data);
	return true;
}

inline bool SerializationHelper::loadFromFile(
    string const& filename,
    std::function<SimulationController*(SerializedSimulation const&)> deserializer,
    SimulationController*& entity)
{
    SerializedSimulation data;
    if (!loadFromFileIntern(filename, data.content)) {
        return false;
    }
    auto settingsFilename = QString::fromStdString(filename).replace(QRegularExpression("\\.\\w+$"), ".settings.json");
    if (!loadFromFileIntern(settingsFilename.toStdString(), data.generalSettings)) {
        return false;
    }
    auto parametersFilename =
        QString::fromStdString(filename).replace(QRegularExpression("\\.\\w+$"), ".parameters.json");
    if (!loadFromFileIntern(parametersFilename.toStdString(), data.simulationParameters)) {
        return false;
    }
    auto symbolsFilename = QString::fromStdString(filename).replace(QRegularExpression("\\.\\w+$"), ".symbols.json");
    if (!loadFromFileIntern(symbolsFilename.toStdString(), data.symbolMap)) {
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

inline bool SerializationHelper::loadFromFileIntern(std::string const& filename, std::string& data)
{
    try {
        QFile file(QString::fromStdString(filename));
        if (!file.open(QIODevice::ReadOnly)) {
            return false;
        }
        QByteArray blob = file.readAll();
        data =  blob.toStdString();
        return true;

    } catch (std::exception const& e) {
        return false;
    }
    return true;
}

inline bool SerializationHelper::saveToFileIntern(std::string const& filename, std::string const& data)
{
    try {
        QFile file(QString::fromStdString(filename));
        if (!file.open(QIODevice::WriteOnly)) {
            return false;
        }
        file.write(data.c_str(), data.length());
    } catch (...) {
        return false;
    }
    return true; 
}
