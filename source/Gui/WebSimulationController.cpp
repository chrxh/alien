#include "WebSimulationController.h"

#include <iostream>
#include <sstream>

#include <QInputDialog>
#include <QEventLoop>
#include <QMessageBox>
#include <QTimer>

#include "Base/Job.h"
#include "Base/Worker.h"
#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"
#include "EngineInterface/SimulationMonitor.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SpaceProperties.h"
#include "Web/WebAccess.h"

#include "SendLiveImageJob.h"
#include "SendLastImageJob.h"
#include "SendStatisticsJob.h"
#include "SimulationConfig.h"
#include "WebSimulationSelectionController.h"
#include "Settings.h"

namespace
{
    auto const POLLING_INTERVAL = 300;
    auto const PROCESS_JOBS_INTERVAL = 50;
    auto const UPDATE_STATISTICS_INTERVAL = 1000;
}

WebSimulationController::WebSimulationController(WebAccess * webAccess, QWidget* parent /*= nullptr*/)
    : QObject(parent)
    , _webAccess(webAccess)
    , _parent(parent)
    , _pollingTimer(new QTimer(this))
    , _processJobsTimer(new QTimer(this))
    , _updateStatisticsTimer(new QTimer(this))
{
    connect(_pollingTimer, &QTimer::timeout, this, &WebSimulationController::requestUnprocessedTasks);
    connect(_processJobsTimer, &QTimer::timeout, this, &WebSimulationController::processJobs);
    connect(_updateStatisticsTimer, &QTimer::timeout, this, &WebSimulationController::sendStatistics);
    connect(_webAccess, &WebAccess::unprocessedTasksReceived, this, &WebSimulationController::unprocessedTasksReceived);
    connect(_webAccess, &WebAccess::error, [&](auto const& message) {
        QMessageBox msgBox(QMessageBox::Critical, "Error", QString::fromStdString(message));
        msgBox.exec();
    });
}

void WebSimulationController::init(
    SimulationAccess * access, 
    SimulationMonitor* monitor, 
    SimulationConfig const& config)
{
    _config = config;

    for (auto const& connection : _connections) {
        disconnect(connection);
    }

    SET_CHILD(_simAccess, access);
    SET_CHILD(_monitor, monitor);

    _worker = boost::make_shared<_Worker>();
    _processJobsTimer->stop();
    _processJobsTimer->start(PROCESS_JOBS_INTERVAL);
}

bool WebSimulationController::onConnectToSimulation()
{
    auto const dialog = new WebSimulationSelectionController(_webAccess, _parent);
    if (!dialog->execute()) {
        delete dialog;
        return false;
    }
    auto const simulationInfo = dialog->getSelectedSimulation();
    auto const title = "Connecting to " + QString::fromStdString(simulationInfo.simulationName);
    auto const label = "Enter password for user " + QString::fromStdString(simulationInfo.userName);
    auto const password = QInputDialog::getText(_parent, title, label, QLineEdit::Password);
    delete dialog;

    if (password.isEmpty()) {
        return false;
    }

    QEventLoop loop;
    bool error = false;
    std::vector<QMetaObject::Connection> connections;
    connections.emplace_back(connect(_webAccess, &WebAccess::connectToSimulationReceived, [&, this](auto const& token) {
        _currentSimulationId = simulationInfo.simulationId;
        _currentToken = token;
        loop.quit();
    }));
    connections.emplace_back(connect(_webAccess, &WebAccess::error, [&](auto const& message) {
        error = true;
        loop.quit();
    }));
    _webAccess->requestConnectToSimulation(simulationInfo.simulationId, password.toStdString());
    loop.exec();
    for (auto const& connection : connections) {
        disconnect(connection);
    }
    if (error) {
        return false;
    }

    if (_currentToken) {
        QMessageBox msgBox(QMessageBox::Information, "Connection successful",
            QString(Const::InfoConnectedTo).arg(QString::fromStdString(simulationInfo.simulationName)));
        msgBox.exec();
        _pollingTimer->start(POLLING_INTERVAL);
        _updateStatisticsTimer->start(UPDATE_STATISTICS_INTERVAL);

        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Important, "Web: connected");

        return true;
    }
    else {
        QMessageBox msgBox(QMessageBox::Critical, "Error", Const::ErrorInvalidPassword);
        msgBox.exec();
        return false;
    }
}

bool WebSimulationController::onDisconnectToSimulation(string const& simulationId, string const & token)
{
    _pollingTimer->stop();
    _updateStatisticsTimer->stop();

    auto newJob = new SendLastImageJob(
        *_currentSimulationId, 
        *_currentToken, 
        IntVector2D{ 0, 0 }, 
        _config->universeSize, 
        _simAccess, 
        _webAccess, 
        this);
    _worker->add(newJob);

    return true;
}

optional<string> WebSimulationController::getCurrentSimulationId() const
{
    return _currentSimulationId;
}

optional<string> WebSimulationController::getCurrentToken() const
{
    return _currentToken;
}

void WebSimulationController::requestUnprocessedTasks() const
{
    _webAccess->requestUnprocessedTasks(*_currentSimulationId, *_currentToken);
}

void WebSimulationController::unprocessedTasksReceived(vector<Task> tasks)
{
    if (tasks.empty() || !_currentSimulationId) {
        return;
    }

    auto numNewJobs = 0;
    for (auto const& task : tasks) {
        if (!_worker->contains(task.id)) {
            auto worldSize = _config->universeSize;
            if (task.pos.x >= worldSize.x || task.pos.y >= worldSize.y) {
                return;
            }

            auto taskSize = task.size;
            if (taskSize.x + task.pos.x >= worldSize.x) {
                taskSize.x = worldSize.x - task.pos.x;
            }
            if (taskSize.y + task.pos.y >= worldSize.y) {
                taskSize.y = worldSize.y - task.pos.y;
            }
            auto newJob = new SendLiveImageJob(
                *_currentSimulationId, *_currentToken, task.id, task.pos, taskSize, _simAccess, _webAccess, this);
            _worker->add(newJob);

            ++numNewJobs;
        }
    }
    if (numNewJobs > 0) {
        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();

        std::stringstream stream;
        stream << "Web: " << numNewJobs << " new task(s) received";
        loggingService->logMessage(Priority::Important, stream.str().c_str());
    }
}

void WebSimulationController::processJobs()
{
    _worker->process();
}

void WebSimulationController::sendStatistics()
{
    if (!_currentSimulationId) {
        return;
    }

    auto const newJob = new SendStatisticsJob(
        *_currentSimulationId, *_currentToken, _monitor, _webAccess, _config, this);
    _worker->add(newJob);
}

