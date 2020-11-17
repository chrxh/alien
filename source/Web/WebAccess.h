#pragma once

#include <QBuffer>

#include "Definitions.h"

#include "SimulationInfo.h"
#include "Task.h"

class WEB_EXPORT WebAccess : public QObject
{
    Q_OBJECT
public:
    virtual ~WebAccess() = default;

    virtual void init() = 0;

    virtual void requestSimulationInfos() = 0;
    virtual void requestConnectToSimulation(string const& simulationId, string const& password) = 0;
    virtual void requestUnprocessedTasks(string const& simulationId, string const& token) = 0;
    virtual void sendProcessedTask(string const& simulationId, string const& token, string const& taskId, QBuffer* data) = 0;
    virtual void requestDisconnect(string const& simulationId, string const& token) = 0;

    Q_SIGNAL void simulationInfosReceived(vector<SimulationInfo> simulationInfos);
    Q_SIGNAL void connectToSimulationReceived(optional<string> token);
    Q_SIGNAL void unprocessedTasksReceived(vector<Task> tasks);
    Q_SIGNAL void sendProcessedTaskReceived();
    Q_SIGNAL void error(string message);

};