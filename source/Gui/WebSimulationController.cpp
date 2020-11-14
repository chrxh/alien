#include "WebSimulationController.h"

#include <iostream>

#include <QInputDialog>
#include <QEventLoop>
#include <QMessageBox>
#include <QTimer>
#include <QBuffer>

#include "ModelBasic/SimulationAccess.h"
#include "Web/WebAccess.h"

#include "WebSimulationSelectionController.h"

WebSimulationController::WebSimulationController(WebAccess * webAccess, QWidget* parent /*= nullptr*/)
    : QObject(parent)
    , _webAccess(webAccess)
    , _parent(parent)
    , _timer(new QTimer(this))
{
    connect(_timer, &QTimer::timeout, this, &WebSimulationController::checkIfSimulationImageIsRequired);
    connect(_webAccess, &WebAccess::unprocessedTasksReceived, this, &WebSimulationController::unprocessedTasksReceived);
    connect(_webAccess, &WebAccess::error, [&](auto const& message) {
        QMessageBox msgBox(QMessageBox::Critical, "Error", QString::fromStdString(message));
        msgBox.exec();
    });
}

void WebSimulationController::init(SimulationAccess * access)
{
    SET_CHILD(_simAccess, access);
    connect(_simAccess, &SimulationAccess::imageReady, this, &WebSimulationController::tasksProcessed);
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
        _timer->start(1000);
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
    _timer->stop();
    _webAccess->requestDisconnect(simulationId, token);
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

void WebSimulationController::checkIfSimulationImageIsRequired() const
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
    std::cerr << "[Web] " << tasks.size() - numPrevTask << " new task(s) received" << std::endl;
    processTasks();
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
    _simAccess->requireImage(rect, _image, _mutex);
}

#include <QDebug>
void WebSimulationController::tasksProcessed()
{
    if (!_currentSimulationId) {
        _processingTaskId = boost::none;
        _taskById.clear();
        return;
    }

    std::cerr << "[Web] task processed" << std::endl;

    auto const taskId = _taskById.at(*_processingTaskId);

    _encodedImageData.clear();

    delete _buffer;
    _buffer = new QBuffer(&_encodedImageData);
    _buffer->open(QIODevice::ReadWrite);

    _image->save(_buffer, "PNG");
    _buffer->seek(0);

/*
    auto const imageSize = _targetImage->size();
    auto const numBytes = imageSize.width() * imageSize.height() * 4;
    QByteArray imageData(reinterpret_cast<const char*>(_targetImage->bits()), numBytes);
*/
    _webAccess->sendProcessedTask(*_currentSimulationId, *_currentToken, _buffer);

    _taskById.erase(*_processingTaskId);
    _processingTaskId = boost::none;
    processTasks();
}

