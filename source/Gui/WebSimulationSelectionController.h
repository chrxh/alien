#pragma once

#include "Web/Definitions.h"
#include "Web/SimulationInfo.h"

#include "Definitions.h"

class WebSimulationSelectionController
    : public QObject
{
    Q_OBJECT
public:
    WebSimulationSelectionController(WebController* webController, QWidget *parent = nullptr);

    int execute();

private:
    Q_SLOT void simulationInfosReceived(vector<SimulationInfo> simulationInfos);
    Q_SLOT void error(string message);

private:
    WebSimulationTableModel* _model = nullptr;
    WebSimulationSelectionView* _view = nullptr;
    WebController* _webController = nullptr;
};