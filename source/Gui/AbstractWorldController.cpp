#include "AbstractWorldController.h"

#include <QGraphicsView>
#include <QScrollBar>

AbstractWorldController::AbstractWorldController(
    SimulationViewWidget* simulationViewWidget,
    QObject* parent /*= nullptr*/)
    : QObject(parent)
    , _simulationViewWidget(simulationViewWidget)
{

}

