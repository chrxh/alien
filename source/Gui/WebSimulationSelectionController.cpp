#include <QMessageBox>
#include <QString>

#include "Web/WebAccess.h"

#include "WebSimulationSelectionView.h"
#include "WebSimulationTableModel.h"

#include "WebSimulationSelectionController.h"

WebSimulationSelectionController::WebSimulationSelectionController(WebAccess* webAccess, QWidget * parent)
    : QObject(parent), _webAccess(webAccess)
{
    _model = new WebSimulationTableModel(parent);
    _view = new WebSimulationSelectionView(this, _model, parent);
    connect(webAccess, &WebAccess::simulationInfosReceived, this, &WebSimulationSelectionController::simulationInfosReceived);

    refresh();
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
    _view->setWindowTitle("Select a simulation to connect (requesting data from server)");
    _webAccess->requestSimulationInfos();
}

void WebSimulationSelectionController::simulationInfosReceived(vector<SimulationInfo> simulationInfos)
{
    _view->setWindowTitle("Select a simulation to connect");
    _model->setSimulationInfos(simulationInfos);
}
