#include <QTimer>
#include <QWidget>

#include "Model/Api/SimulationMonitor.h"

#include "MonitorView.h"
#include "MonitorModel.h"
#include "MonitorController.h"

namespace
{
	const int millisec = 200;
}

MonitorController::MonitorController(QWidget* parent)
	: QObject(parent)
{
	_view = new MonitorView(parent);
	_view->setVisible(false);
	connect(_view, &MonitorView::closed, this, &MonitorController::closed);

	_updateTimer = new QTimer(this);
}

void MonitorController::init(SimulationMonitor* simMonitor)
{
	_model = boost::make_shared<_MonitorModel>();
	_view->init(_model);
	_simMonitor = simMonitor;

	for (auto const& connection : _connections) {
		disconnect(connection);
	}
	_connections.push_back(connect(simMonitor, &SimulationMonitor::dataReadyToRetrieve, this, &MonitorController::dataReadyToRetrieve, Qt::QueuedConnection));
	_connections.push_back(connect(_updateTimer, &QTimer::timeout, _simMonitor, &SimulationMonitor::requireData));
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

void MonitorController::dataReadyToRetrieve()
{
	MonitorData const& data = _simMonitor->retrieveData();
	_model->numClusters= data.numClusters;
	_model->numCells = data.numCells;
	_model->numParticles = data.numParticles;
	_model->numTokens = data.numTokens;
	_model->totalInternalEnergy = data.totalInternalEnergy;
	_model->totalLinearKineticEnergy = data.totalLinearKineticEnergy;
	_model->totalRotationalKineticEnergy = data.totalRotationalKineticEnergy;
	_view->update();
}
