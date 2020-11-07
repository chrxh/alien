#include <QMessageBox>
#include <QString>

#include "Web/WebController.h"

#include "WebSimulationSelectionView.h"
#include "WebSimulationTableModel.h"

#include "WebSimulationSelectionController.h"

WebSimulationSelectionController::WebSimulationSelectionController(WebController* webController, QWidget * parent)
    : QObject(parent), _webController(webController)
{
    _model = new WebSimulationTableModel(parent);
    _view = new WebSimulationSelectionView(this, _model, parent);
    connect(webController, &WebController::simulationInfosReceived, this, &WebSimulationSelectionController::simulationInfosReceived);
    connect(webController, &WebController::error, this, &WebSimulationSelectionController::error);

    webController->requestSimulationInfos();
}

SimulationInfo WebSimulationSelectionController::getSelectedSimulation() const
{
    auto const index = _view->getIndexOfSelectedSimulation();
    return _model->getSimulationInfo(index);
}

int WebSimulationSelectionController::execute()
{
    return _view->exec();
}

void WebSimulationSelectionController::refresh()
{
    _webController->requestSimulationInfos();
}

void WebSimulationSelectionController::simulationInfosReceived(vector<SimulationInfo> simulationInfos)
{
    _model->setSimulationInfos(simulationInfos);
}

void WebSimulationSelectionController::error(string message)
{
    QMessageBox msgBox(QMessageBox::Critical, "Error", QString::fromStdString(message));
    msgBox.exec();
}
