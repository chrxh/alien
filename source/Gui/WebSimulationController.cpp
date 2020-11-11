#include "WebSimulationController.h"

#include <QInputDialog>
#include <QEventLoop>
#include <QMessageBox>

#include "Web/WebController.h"

#include "WebSimulationSelectionController.h"

WebSimulationController::WebSimulationController(WebController * webController, QWidget* parent /*= nullptr*/)
    : QObject(parent)
    , _webController(webController)
    , _parent(parent)
{
}

bool WebSimulationController::onConnectToSimulation()
{
    auto const dialog = new WebSimulationSelectionController(_webController, _parent);
    if (!dialog->execute()) {
        delete dialog;
        return false;
    }
    auto const simulationInfo = dialog->getSelectedSimulation();
    auto const title = "Connecting to " + QString::fromStdString(simulationInfo.simulationName);
    auto const label = "Enter password for " + QString::fromStdString(simulationInfo.userName);
    auto const password = QInputDialog::getText(_parent, title, label, QLineEdit::Password);
    delete dialog;

    if (password.isEmpty()) {
        return false;
    }

    QEventLoop loop;
    bool error = false;
    connect(_webController, &WebController::connectToSimulationReceived, [this, &loop](auto const& token) {
        _token = token;
        loop.quit();
    });
    connect(_webController, &WebController::error, [&](auto const& message) {
        QMessageBox msgBox(QMessageBox::Critical, "Error", QString::fromStdString(message));
        msgBox.exec();
        error = true;
        loop.quit();
    });
    _webController->requestConnectToSimulation(simulationInfo.simulationId, password.toStdString());
    loop.exec();
    if (error) {
        return false;
    }

    if (_token) {
        QMessageBox msgBox(QMessageBox::Information, "Connection successful", "You are connected to "
            + QString::fromStdString(simulationInfo.simulationName) + ".");
        msgBox.exec();
        return true;
    }
    else {
        QMessageBox msgBox(QMessageBox::Critical, "Error", "The password you entered is incorrect.");
        msgBox.exec();
        return false;
    }
}

bool WebSimulationController::onDisconnectToSimulation(string const & token)
{
    return false;
}

optional<string> WebSimulationController::getConnectionToken() const
{
    return _token;
}

