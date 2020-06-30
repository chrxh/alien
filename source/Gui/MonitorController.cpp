#include <QTimer>
#include <QWidget>

#include "ModelBasic/SimulationMonitor.h"

#include "MonitorView.h"
#include "MonitorController.h"
#include "MainController.h"

namespace
{
	const int millisec = 200;
}

MonitorController::MonitorController(QWidget* parent)
	: QObject(parent)
{
	_view = new MonitorView(parent);
	_view->setVisible(false);

	_updateTimer = new QTimer(this);

	connect(_view, &MonitorView::closed, this, &MonitorController::closed);
	connect(_updateTimer, &QTimer::timeout, this, &MonitorController::timerTimeout);
}

void MonitorController::init(MainController* mainController)
{
	_model = boost::make_shared<MonitorData>();
	_mainController = mainController;
	_view->init(_model);
}

void MonitorController::onShow(bool show)
{
	_view->setVisible(show);
	if (show) {
		_updateTimer->start(millisec);
	}
	else {
		_updateTimer->stop();
	}
}

void MonitorController::timerTimeout()
{
	for (auto const& connection : _monitorConnections) {
		disconnect(connection);
	}
	SimulationMonitor* simMonitor = _mainController->getSimulationMonitor();
	_monitorConnections.push_back(connect(simMonitor, &SimulationMonitor::dataReadyToRetrieve, this, &MonitorController::dataReadyToRetrieve, Qt::QueuedConnection));
	simMonitor->requireData();
}

void MonitorController::dataReadyToRetrieve()
{
	SimulationMonitor* simMonitor = _mainController->getSimulationMonitor();
	MonitorData const& data = simMonitor->retrieveData();
    *_model = data;
	_view->update();
}
