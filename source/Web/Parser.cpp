#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include "Parser.h"

vector<SimulationInfo> Parser::parse(QByteArray const & raw)
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
        simulationInfo.simulationId = simulationInfoObject.value("id").toString().toStdString();
        simulationInfo.isActive= simulationInfoObject.value("isActive").toBool();
        simulationInfo.simulationName = simulationInfoObject.value("simulationName").toString().toStdString();
        simulationInfo.userName = simulationInfoObject.value("userName").toString().toStdString();
        simulationInfo.timestep = simulationInfoObject.value("timestep").toInt();

        result.emplace_back(simulationInfo);
    }
    return result;
}
