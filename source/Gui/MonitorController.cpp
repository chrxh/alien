#include <QTimer>
#include <QWidget>

#include "EngineInterface/SimulationMonitor.h"

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

QWidget* MonitorController::getWidget() const
{
    return _view;
}

void MonitorController::pauseTimer()
{
    _updateTimer->stop();
}

void MonitorController::continueTimer()
{
    _updateTimer->start(millisec);
}

void MonitorController::startWritingToFile(std::string const& filename)
{
    stopWritingToFile();

    FileInfo info;
    info.file.open(filename, std::ios_base::out);
    info.file << "time step, number of cells, number of particles, number of tokens, total internal energy"
              << std::endl;

    _fileInfo = std::move(info);
}

void MonitorController::stopWritingToFile()
{
    if (_fileInfo) {
        _fileInfo->file.close(); 
    }
    _fileInfo = boost::none;
}

void MonitorController::timerTimeout()
{
	for (auto const& connection : _monitorConnections) {
		disconnect(connection);
	}
    _monitorConnections.clear();

    if (SimulationMonitor* simMonitor = _mainController->getSimulationMonitor()) {
        _monitorConnections.push_back(connect(
            simMonitor,
            &SimulationMonitor::dataReadyToRetrieve,
            this,
            &MonitorController::dataReadyToRetrieve,
            Qt::QueuedConnection));
        simMonitor->requireData();
    }
}

void MonitorController::dataReadyToRetrieve()
{
    SimulationMonitor* simMonitor = _mainController->getSimulationMonitor();
	MonitorData const& data = simMonitor->retrieveData();
    *_model = data;
	_view->update();

    if (_fileInfo) {
        writeDataToFile();
    }
}

void MonitorController::writeDataToFile()
{
    if (_model->timeStep != _fileInfo->lastTimestep) {
        _fileInfo->file << _model->timeStep << ", " << _model->numCells << ", " << _model->numParticles << ", " << _model->numTokens << ", "
                        << _model->totalInternalEnergy << std::endl;

        _fileInfo->lastTimestep = _model->timeStep;
    }
}
