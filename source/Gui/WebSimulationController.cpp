#include "WebSimulationController.h"

#include <iostream>

#include <QInputDialog>
#include <QEventLoop>
#include <QMessageBox>
#include <QTimer>
#include <QBuffer>

#include "ModelBasic/SimulationMonitor.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SpaceProperties.h"
#include "Web/WebAccess.h"

#include "WebSimulationSelectionController.h"

namespace
{
    auto const POLLING_INTERVAL = 300;
    auto const UPDATE_STATISTICS_INTERVAL = 1000;
}

WebSimulationController::WebSimulationController(WebAccess * webAccess, QWidget* parent /*= nullptr*/)
    : QObject(parent)
    , _webAccess(webAccess)
    , _parent(parent)
    , _pollingTimer(new QTimer(this))
    , _updateStatisticsTimer(new QTimer(this))
{
    connect(_pollingTimer, &QTimer::timeout, this, &WebSimulationController::requestUnprocessedTasks);
    connect(_updateStatisticsTimer, &QTimer::timeout, this, &WebSimulationController::sendStatistics);
    connect(_webAccess, &WebAccess::unprocessedTasksReceived, this, &WebSimulationController::unprocessedTasksReceived);
    connect(_webAccess, &WebAccess::error, [&](auto const& message) {
        QMessageBox msgBox(QMessageBox::Critical, "Error", QString::fromStdString(message));
        msgBox.exec();
    });
    connect(_webAccess, &WebAccess::sendProcessedTaskReceived, this, &WebSimulationController::imageSent);
    connect(_webAccess, &WebAccess::sendLastImageReceived, this, &WebSimulationController::imageSent);
}

void WebSimulationController::init(SimulationAccess * access, SimulationMonitor* monitor, SpaceProperties* space)
{
    _requestImageType = RequestImageType::LiveUpdate;
    _space = space;

    for (auto const& connection : _connections) {
        disconnect(connection);
    }

    SET_CHILD(_simAccess, access);
    _connections.emplace_back(connect(_simAccess, &SimulationAccess::imageReady, this, &WebSimulationController::imageReceived));

    SET_CHILD(_monitor, monitor);
    _connections.emplace_back(connect(_monitor, &SimulationMonitor::dataReadyToRetrieve, this, &WebSimulationController::statisticsReadyToRetrieve));

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
        QMessageBox msgBox(QMessageBox::Information, "Connection successful", "You are connected to "
            + QString::fromStdString(simulationInfo.simulationName) + ".");
        msgBox.exec();
        _pollingTimer->start(POLLING_INTERVAL);
        _updateStatisticsTimer->start(UPDATE_STATISTICS_INTERVAL);
        std::cerr << "[Web] connected" << std::endl;
        return true;
    }
    else {
        QMessageBox msgBox(QMessageBox::Critical, "Error", "The password you entered is incorrect.");
        msgBox.exec();
        return false;
    }
}

bool WebSimulationController::onDisconnectToSimulation(string const& simulationId, string const & token)
{
    _pollingTimer->stop();
    _updateStatisticsTimer->stop();

    requestLastImage();
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
    if (tasks.empty()) {
        return;
    }
    if (!_currentSimulationId) {
        _processingTaskId = boost::none;
        _taskById.clear();
        return;
    }

    auto numPrevTask = _taskById.size();
    for (auto const& task : tasks) {
        _taskById.insert_or_assign(task.id, task);
    }
    if (tasks.size() - numPrevTask > 0) {
        std::cerr << "[Web] " << tasks.size() - numPrevTask << " new task(s) received" << std::endl;
        processTasks();
    }
}

void WebSimulationController::processTasks()
{
    if (_processingTaskId || _taskById.empty()) {
        return;
    }
    auto const task = _taskById.begin()->second;
    _processingTaskId = task.id;

    _image = boost::make_shared<QImage>(task.size.x, task.size.y, QImage::Format_RGB32);
    auto const rect = IntRect{ task.pos, IntVector2D{task.pos.x + task.size.x, task.pos.y + task.size.y } };
    std::cerr 
        << "[Web] processing task " 
        << task.id 
        << ": request image with size " 
        << task.size.x 
        << " x " 
        << task.size.y 
        << std::endl;

    _simAccess->requireImage(rect, _image, _mutex);
}

void WebSimulationController::imageReceived()
{
    if (!_currentSimulationId) {
        _processingTaskId = boost::none;
        _taskById.clear();
        return;
    }

    delete _buffer;
    _buffer = new QBuffer(&_encodedImageData);
    _buffer->open(QIODevice::ReadWrite);
    _image->save(_buffer, "PNG");
    _buffer->seek(0);

    if (RequestImageType::LastImage == _requestImageType) {
        _webAccess->sendLastImage(*_currentSimulationId, *_currentToken, _buffer);
    }
    else {
        _webAccess->sendProcessedTask(*_currentSimulationId, *_currentToken, *_processingTaskId, _buffer);
    }
}

void WebSimulationController::imageSent()
{
    if (!_currentSimulationId) {
        _processingTaskId = boost::none;
        _taskById.clear();
        return;
    }

    if (RequestImageType::LastImage == _requestImageType) {
        _webAccess->requestDisconnect(*_currentSimulationId, *_currentToken);
        std::cerr << "[Web] disconnected" << std::endl;
    }
    else {
        auto const taskId = *_processingTaskId;

        std::cerr << "[Web] task " << taskId << " processed" << std::endl;

        _taskById.erase(taskId);
        _processingTaskId = boost::none;
        processTasks();
    }
}

void WebSimulationController::sendStatistics()
{
    _monitor->requireData();
}

void WebSimulationController::statisticsReadyToRetrieve()
{
    if (!_currentSimulationId || !_currentToken) {
        return;
    }

    auto monitorData = _monitor->retrieveData();

    _webAccess->sendStatistics(*_currentSimulationId, *_currentToken, {
        { "timestep", std::to_string(monitorData.timeStep) },
        { "numCells", std::to_string(monitorData.numCells) },
        { "numParticles", std::to_string(monitorData.numParticles) },
        { "numClusters", std::to_string(monitorData.numClusters) },
        { "numActiveClusters", std::to_string(monitorData.numClustersWithTokens) },
        { "numTokens", std::to_string(monitorData.numTokens) },
    });
}

void WebSimulationController::requestLastImage()
{
    auto size = _space->getSize();
    _image = boost::make_shared<QImage>(size.x, size.y, QImage::Format_RGB32);
    auto const rect = IntRect{ {0,0}, {size.x - 1, size.y - 1} };
    std::cerr
        << "[Web] request last image"
        << std::endl;
    _requestImageType = RequestImageType::LastImage;
    _simAccess->requireImage(rect, _image, _mutex);
}

