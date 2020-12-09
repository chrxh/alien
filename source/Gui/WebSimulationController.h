#pragma once

#include <mutex>
#include <QObject>

#include "Base/Definitions.h"
#include "ModelBasic/Definitions.h"
#include "Web/Definitions.h"
#include "Web/Task.h"

#include "Definitions.h"

class WebSimulationController
    : public QObject
{
    Q_OBJECT
public:
    WebSimulationController(WebAccess* webAccess, QWidget* parent = nullptr);

    void init(SimulationAccess* access, SimulationMonitor* monitor, SpaceProperties* space);

    bool onConnectToSimulation();
    bool onDisconnectToSimulation(string const& simulationId, string const& token);

    optional<string> getCurrentSimulationId() const;
    optional<string> getCurrentToken() const;

private:
    Q_SLOT void requestUnprocessedTasks() const;
    Q_SLOT void unprocessedTasksReceived(vector<Task> tasks);

    void processJobs();
    Q_SLOT void sendStatistics();

    optional<string> _currentSimulationId;
    optional<string> _currentToken;

    Worker _worker;

    QByteArray _encodedImageData;
    QBuffer* _buffer = nullptr;

    SimulationAccess* _simAccess = nullptr;
    SimulationMonitor* _monitor = nullptr;
    SpaceProperties* _space = nullptr;
    QWidget* _parent = nullptr;
    WebAccess* _webAccess = nullptr;
    QTimer* _pollingTimer = nullptr;
    QTimer* _processJobsTimer = nullptr;
    QTimer* _updateStatisticsTimer = nullptr;

    list<QMetaObject::Connection> _connections;
};