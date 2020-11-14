#pragma once

#include "Web/Definitions.h"
#include "Web/SimulationInfo.h"

#include "Definitions.h"

class WebSimulationSelectionController
    : public QObject
{
    Q_OBJECT
public:
    WebSimulationSelectionController(WebAccess* webController, QWidget *parent = nullptr);

    SimulationInfo getSelectedSimulation() const;

    int execute();

    void refresh();

private:
    Q_SLOT void simulationInfosReceived(vector<SimulationInfo> simulationInfos);

private:
    WebSimulationTableModel* _model = nullptr;
    WebSimulationSelectionView* _view = nullptr;
    WebAccess* _webAccess = nullptr;
};