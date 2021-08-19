#include "ZoomActionController.h"

#include <QAction>

#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"

#include "EngineInterface/ZoomLevels.h"

#include "ActionHolder.h"
#include "SimulationViewController.h"
#include "Settings.h"

ZoomActionController::ZoomActionController(QObject* parent)
    : QObject(parent)
{
    connect(&_continuousZoomTimer, &QTimer::timeout, this, &ZoomActionController::onContinuousZoom);
}

void ZoomActionController::init(ActionHolder* actions, SimulationViewController* simulationViewController)
{
    _actions = actions;
    _simulationViewController = simulationViewController;
}

void ZoomActionController::onZoomInClicked()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Unimportant, "zoom in");

    auto zoomFactor = _simulationViewController->getZoomFactor();
    _simulationViewController->setZoomFactor(zoomFactor * 2);

    loggingService->logMessage(Priority::Unimportant, "zoom in finished");

    if (!_actions->actionItemView->isChecked()) {
        if (_simulationViewController->getZoomFactor()
            > Const::ZoomLevelForAutomaticEditorSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
            setItemView();
        } else {
            setPixelOrVectorView();
        }
    }
    if (!_actions->actionRunSimulation->isChecked()) {
        _simulationViewController->refresh();
    }
    Q_EMIT updateActionsState();
}

void ZoomActionController::onZoomOutClicked()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Unimportant, "zoom out");

    auto zoomFactor = _simulationViewController->getZoomFactor();
    _simulationViewController->setZoomFactor(zoomFactor / 2);

    loggingService->logMessage(Priority::Unimportant, "zoom out finished");

    if (_actions->actionItemView->isChecked()) {
        if (_simulationViewController->getZoomFactor()
            > Const::ZoomLevelForAutomaticEditorSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
        } else {
            setPixelOrVectorView();
        }
    } else {
        setPixelOrVectorView();
    }
    if (!_actions->actionRunSimulation->isChecked()) {
        _simulationViewController->refresh();
    }
    Q_EMIT updateActionsState();
}

void ZoomActionController::onContinuousZoomIn(IntVector2D const& viewPos)
{
    if (!_continuousZoomTimer.isActive()) {
        _continuousZoomTimer.start(Const::ContinuousZoomInterval);
    }
    _continuousZoomMode = ContinuousZoomMode::In;
    _continuousZoomWorldPos = viewPos;
}

void ZoomActionController::onContinuousZoomOut(IntVector2D const& viewPos)
{
    if (!_continuousZoomTimer.isActive()) {
        _continuousZoomTimer.start(Const::ContinuousZoomInterval);
    }
    _continuousZoomMode = ContinuousZoomMode::Out;
    _continuousZoomWorldPos = viewPos;
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

        auto zoomFactor = _simulationViewController->getZoomFactor();
        _simulationViewController->setZoomFactor(zoomFactor * 1.05, *_continuousZoomWorldPos);

        loggingService->logMessage(Priority::Unimportant, "continuous zoom in finished");

        if (!_actions->actionItemView->isChecked()) {
            if (_simulationViewController->getZoomFactor()
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

        auto zoomFactor = _simulationViewController->getZoomFactor();
        _simulationViewController->setZoomFactor(zoomFactor / 1.05, *_continuousZoomWorldPos);

        loggingService->logMessage(Priority::Unimportant, "continuous zoom out finished");

        if (_actions->actionItemView->isChecked()) {
            if (_simulationViewController->getZoomFactor()
                > Const::ZoomLevelForAutomaticEditorSwitch - FLOATINGPOINT_MEDIUM_PRECISION) {
            } else {
                setPixelOrVectorView();
            }
        } else {
            setPixelOrVectorView();
        }
    }
    if (!isSimulationRunning()) {
        _simulationViewController->refresh();
    }
    Q_EMIT updateActionsState();
}

bool ZoomActionController::isSimulationRunning() const
{
    return _actions->actionRunSimulation->isChecked();
}

void ZoomActionController::setItemView()
{
//    _actions->actionItemView->setChecked(true);
}


void ZoomActionController::setPixelOrVectorView()
{
    _actions->actionOpenGLView->setChecked(true);
//    Q_EMIT _actions->actionItemView->toggled(false);  //for switching from vector to pixel view
}

