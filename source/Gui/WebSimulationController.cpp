#include "WebSimulationController.h"

#include <QInputDialog>
#include <QEventLoop>
#include <QMessageBox>
#include <QTimer>

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

}

void WebSimulationController::init(SimulationAccess * access)
{
    SET_CHILD(_access, access);
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
    connect(_webAccess, &WebAccess::connectToSimulationReceived, [&, this](auto const& token) {
        _currentSimulationId = simulationInfo.simulationId;
        _currentToken = token;
        loop.quit();
    });
    connect(_webAccess, &WebAccess::error, [&](auto const& message) {
        QMessageBox msgBox(QMessageBox::Critical, "Error", QString::fromStdString(message));
        msgBox.exec();
        error = true;
        loop.quit();
    });
    _webAccess->requestConnectToSimulation(simulationInfo.simulationId, password.toStdString());
    loop.exec();
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

