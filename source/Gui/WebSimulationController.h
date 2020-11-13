#pragma once

#include <mutex>

#include <QObject>

#include "ModelBasic/Definitions.h"
#include "Web/Definitions.h"
#include "Web/Task.h"

class WebSimulationController
    : public QObject
{
    Q_OBJECT
public:
    WebSimulationController(WebAccess* webAccess, QWidget* parent = nullptr);

    void init(SimulationAccess* access);

    bool onConnectToSimulation();
    bool onDisconnectToSimulation(string const& simulationId, string const& token);

    optional<string> getCurrentSimulationId() const;
    optional<string> getCurrentToken() const;

private:
    Q_SLOT void checkIfSimulationImageIsRequired() const;
    Q_SLOT void unprocessedTasksReceived(vector<UnprocessedTask> tasks);

    void processTasks();
    Q_SLOT void tasksProcessed();

    optional<string> _currentSimulationId;
    optional<string> _currentToken;

    map<string, UnprocessedTask> _taskById;
    optional<string> _processingTaskId;
    QImagePtr _targetImage;
    std::mutex _mutex;

    SimulationAccess* _access = nullptr;
    QWidget* _parent = nullptr;
    WebAccess* _webAccess = nullptr;
    QTimer* _timer = nullptr;
};