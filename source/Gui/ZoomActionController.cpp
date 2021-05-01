#include "ZoomActionController.h"

#include <QAction>

#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"

#include "ActionHolder.h"
#include "SimulationViewWidget.h"
#include "Settings.h"

ZoomActionController::ZoomActionController(QObject* parent)
    : QObject(parent)
{
    connect(&_continuousZoomTimer, &QTimer::timeout, this, &ZoomActionController::onContinuousZoom);
}

void ZoomActionController::init(ActionHolder* actions, SimulationViewWidget* simulationViewWidget)
{
    _actions = actions;
    _simulationViewWidget = simulationViewWidget;
}

void ZoomActionController::onZoomInClicked()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Unimportant, "zoom in");

    auto zoomFactor = _simulationViewWidget->getZoomFactor();
    _simulationViewWidget->setZoomFactor(zoomFactor * 2);

    loggingService->logMessage(Priority::Unimportant, "zoom in finished");

    if (!_actions->actionEditor->isChecked()) {
        if (_simulationViewWidget->getZoomFactor()
            > Const::ZoomLevelForAutomaticEditorSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
            setItemView();
        } else {
            setPixelOrVectorView();
        }
    }
    if (!_actions->actionRunSimulation->isChecked()) {
        _simulationViewWidget->refresh();
    }
    Q_EMIT updateActionsState();
}

void ZoomActionController::onZoomOutClicked()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Unimportant, "zoom out");

    auto zoomFactor = _simulationViewWidget->getZoomFactor();
    _simulationViewWidget->setZoomFactor(zoomFactor / 2);

    loggingService->logMessage(Priority::Unimportant, "zoom out finished");

    if (_actions->actionEditor->isChecked()) {
        if (_simulationViewWidget->getZoomFactor()
            > Const::ZoomLevelForAutomaticEditorSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
        } else {
//            _actions->actionEditor->toggle();
            setPixelOrVectorView();
        }
    } else {
        setPixelOrVectorView();
    }
    if (!_actions->actionRunSimulation->isChecked()) {
        _simulationViewWidget->refresh();
    }
    Q_EMIT updateActionsState();
}

void ZoomActionController::onContinuousZoomIn(QVector2D const& worldPos)
{
    _continuousZoomTimer.start(std::chrono::milliseconds(20));
    _continuousZoomMode = ContinuousZoomMode::In;
    _continuousZoomWorldPos = worldPos;
}

void ZoomActionController::onContinuousZoomOut(QVector2D const& worldPos)
{
    _continuousZoomTimer.start(std::chrono::milliseconds(20));
    _continuousZoomMode = ContinuousZoomMode::Out;
    _continuousZoomWorldPos = worldPos;
}

void ZoomActionController::onEndContinuousZoom()
{
    _continuousZoomTimer.stop();
    _continuousZoomMode = boost::none;
    _continuousZoomWorldPos = boost::none;
}

void ZoomActionController::onContinuousZoom()
{
    if (ContinuousZoomMode::In == _continuousZoomMode) {
        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Unimportant, "continuous zoom in");

        auto zoomFactor = _simulationViewWidget->getZoomFactor();
        _simulationViewWidget->setZoomFactor(zoomFactor * 1.05, *_continuousZoomWorldPos);

        loggingService->logMessage(Priority::Unimportant, "continuous zoom in finished");

        if (!_actions->actionEditor->isChecked()) {
            if (_simulationViewWidget->getZoomFactor()
                > Const::ZoomLevelForAutomaticEditorSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
                setItemView();
            } else {
                setPixelOrVectorView();
            }
        }
    }
    if (ContinuousZoomMode::Out == _continuousZoomMode) {
        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Unimportant, "continuous zoom out");

        auto zoomFactor = _simulationViewWidget->getZoomFactor();
        _simulationViewWidget->setZoomFactor(zoomFactor / 1.05, *_continuousZoomWorldPos);

        loggingService->logMessage(Priority::Unimportant, "continuous zoom out finished");

        if (_actions->actionEditor->isChecked()) {
            if (_simulationViewWidget->getZoomFactor()
                > Const::ZoomLevelForAutomaticEditorSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
            } else {
                setPixelOrVectorView();
            }
        } else {
            setPixelOrVectorView();
        }
    }
    if (!isSimulationRunning()) {
        _simulationViewWidget->refresh();
    }
    Q_EMIT updateActionsState();
}

bool ZoomActionController::isSimulationRunning() const
{
    return _actions->actionRunSimulation->isChecked();
}

void ZoomActionController::setItemView()
{
    _actions->actionEditor->setChecked(true);
}


void ZoomActionController::setPixelOrVectorView()
{
    _actions->actionEditor->setChecked(false);
    Q_EMIT _actions->actionEditor->toggled(false);  //for switching from vector to pixel view
}

