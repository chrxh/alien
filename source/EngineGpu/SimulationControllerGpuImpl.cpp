#include <QTimer>
#include <QTime>

#include "CudaController.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationControllerGpuImpl.h"

namespace
{
	const int updateFrameInMilliSec = 30.0;
}

SimulationControllerGpuImpl::SimulationControllerGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationControllerGpu(parent)
	, _oneSecondTimer(new QTimer(this))
	, _frameTimer(new QTimer(this))
{
	connect(_oneSecondTimer, &QTimer::timeout, this, &SimulationControllerGpuImpl::oneSecondTimerTimeout);
	connect(_frameTimer, &QTimer::timeout, this, &SimulationControllerGpuImpl::frameTimerTimeout);

	_oneSecondTimer->start(1000);

    setEnableCalculateFrames(true);
}

void SimulationControllerGpuImpl::init(SimulationContext * context)
{
	SET_CHILD(_context, static_cast<SimulationContextGpuImpl*>(context));
	connect(_context->getCudaController(), &CudaController::timestepCalculated, [this]() {
		Q_EMIT nextTimestepCalculated();
		++_timestepsPerSecond;
		if (_mode == RunningMode::OpenEnded) {
            if (QTime::currentTime().msecsTo(_timeSinceLastStart) > updateFrameInMilliSec * _displayedFramesSinceLastStart) {
				++_displayedFramesSinceLastStart;
			}
		}

		if (_mode != RunningMode::OpenEnded) {
			Q_EMIT nextFrameCalculated();
			_mode = RunningMode::DoNothing;
		}

	});
}

bool SimulationControllerGpuImpl::getRun()
{
    return RunningMode::OpenEnded == _mode;
}

void SimulationControllerGpuImpl::setRun(bool run)
{
	_displayedFramesSinceLastStart = 0;
	if (run) {
		_mode = RunningMode::OpenEnded;
        _timeSinceLastStart = QTime::currentTime();
	}
	else {
		_mode = RunningMode::DoNothing;
	}
	_context->getCudaController()->calculate(_mode);
}

void SimulationControllerGpuImpl::calculateSingleTimestep()
{
	_mode = RunningMode::CalcSingleTimestep;
    _timeSinceLastStart = QTime::currentTime();
    _context->getCudaController()->calculate(_mode);
}

SimulationContext * SimulationControllerGpuImpl::getContext() const
{
	return _context;
}

void SimulationControllerGpuImpl::setRestrictTimestepsPerSecond(optional<int> tps)
{
	_context->getCudaController()->restrictTimestepsPerSecond(tps);
}

void SimulationControllerGpuImpl::setEnableCalculateFrames(bool enabled)
{
    if (enabled) {
        _frameTimer->start(updateFrameInMilliSec);
    }
    else {
        _frameTimer->stop();
    }
}

void SimulationControllerGpuImpl::oneSecondTimerTimeout()
{
	_timestepsPerSecond = 0;
}

void SimulationControllerGpuImpl::frameTimerTimeout()
{
	if (_mode != RunningMode::DoNothing) {
		Q_EMIT nextFrameCalculated();
	}
}
