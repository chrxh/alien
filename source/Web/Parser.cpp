#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QString>

#include "Base/Exceptions.h"
#include "Parser.h"

vector<SimulationInfo> Parser::parseForSimulationInfos(QByteArray const & raw)
{
    vector<SimulationInfo> result;

    auto jsonDoc = QJsonDocument::fromJson(raw);
    if (!jsonDoc.isObject()) {
        throw ParseErrorException("Parser error.");
    }
    auto jsonObject = jsonDoc.object();

    if (1 != jsonObject.size()) {
        throw ParseErrorException("Parser error.");
    }
    auto dataValue = jsonObject.value(jsonObject.keys().front());
    if (!dataValue.isArray()) {
        throw ParseErrorException("Parser error.");
    }

    auto simulationInfoArray = dataValue.toArray();
    for (auto simIndex = 0; simIndex < simulationInfoArray.size(); ++simIndex) {
        auto simulationInfoValue = simulationInfoArray.at(simIndex);
        if (!simulationInfoValue.isObject()) {
            throw ParseErrorException("Parser error.");
        }
        auto simulationInfoObject = simulationInfoValue.toObject();

        SimulationInfo simulationInfo;
        simulationInfo.simulationId = std::to_string(simulationInfoObject.value("id").toInt());
        simulationInfo.isActive = simulationInfoObject.value("isActive").toBool();
        simulationInfo.simulationName = simulationInfoObject.value("simulationName").toString().toStdString();
        simulationInfo.userName = simulationInfoObject.value("userName").toString().toStdString();
        simulationInfo.timestep = simulationInfoObject.value("timestep").toInt();

        auto worldSizeValue = simulationInfoObject.value("worldSize");
        if (!worldSizeValue.isArray()) {
            throw ParseErrorException("Parser error.");
        }
        auto worldSizeArray = worldSizeValue.toArray();
        simulationInfo.worldSize = IntVector2D{ worldSizeArray.at(0).toInt(), worldSizeArray.at(1).toInt() };

        result.emplace_back(simulationInfo);
    }
    return result;
}

vector<Task> Parser::parseForUnprocessedTasks(QByteArray const & raw)
{
    vector<Task> result;

    auto jsonDoc = QJsonDocument::fromJson(raw);
    if (!jsonDoc.isObject()) {
        throw ParseErrorException("Parser error.");
    }
    auto jsonObject = jsonDoc.object();

    if (1 != jsonObject.size()) {
        throw ParseErrorException("Parser error.");
    }
    auto dataValue = jsonObject.value(jsonObject.keys().front());
    if (!dataValue.isArray()) {
        throw ParseErrorException("Parser error.");
    }

    auto taskArray = dataValue.toArray();
    for (auto taskIndex = 0; taskIndex < taskArray.size(); ++taskIndex) {
        auto taskValue = taskArray.at(taskIndex);
        if (!taskValue.isObject()) {
            throw ParseErrorException("Parser error.");
        }
        auto taskObject = taskValue.toObject();

        Task task;
        task.id = std::to_string(taskObject.value("id").toInt());

        auto posValue = taskObject.value("pos");
        if (!posValue.isArray()) {
            throw ParseErrorException("Parser error.");
        }
        auto posArray = posValue.toArray();
        task.pos = { posArray.at(0).toInt(), posArray.at(1).toInt() };

        auto sizeValue = taskObject.value("size");
        if (!sizeValue.isArray()) {
            throw ParseErrorException("Parser error.");
        }
        auto sizeArray = sizeValue.toArray();
        task.size = { sizeArray.at(0).toInt(), sizeArray.at(1).toInt() };
        result.emplace_back(task);
    }
    return result;
}
