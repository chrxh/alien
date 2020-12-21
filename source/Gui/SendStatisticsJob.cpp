#include "SendStatisticsJob.h"

#include <iostream>
#include <QBuffer>
#include <QImage>

#include "ModelBasic/SimulationMonitor.h"

#include "Web/WebAccess.h"

#include "SimulationConfig.h"

SendStatisticsJob::SendStatisticsJob(
    string const& currentSimulationId,
    string const& currentToken,
    SimulationMonitor* simMonitor,
    WebAccess* webAccess,
    SimulationConfig const& config,
    QObject* parent)
    : Job("SendStatisticsJob", parent)
    , _currentSimulationId(currentSimulationId)
    , _currentToken(currentToken)
    , _simMonitor(simMonitor)
    , _webAccess(webAccess)
    , _config(config)
{
    connect(_simMonitor, &SimulationMonitor::dataReadyToRetrieve, this, &SendStatisticsJob::statisticsFromGpuReceived);
}

void SendStatisticsJob::process()
{
    if (!_isReady) {
        return;
    }

    switch (_state)
    {
    case State::Init:
        requestStatistics();
        break;
    case State::StatisticsFromGpuRequested:
        sendStatisticsToServer();
        break;
    default:
        break;
    }
}

bool SendStatisticsJob::isFinished() const
{
    return State::Finished == _state;
}

bool SendStatisticsJob::isBlocking() const
{
    return true;
}

void SendStatisticsJob::requestStatistics()
{
    _simMonitor->requireData();

    _state = State::StatisticsFromGpuRequested;
    _isReady = false;
}

void SendStatisticsJob::sendStatisticsToServer()
{
    auto monitorData = _simMonitor->retrieveData();

    _webAccess->sendStatistics(_currentSimulationId, _currentToken, {
        { "timestep", std::to_string(monitorData.timeStep) },
        { "numCells", std::to_string(monitorData.numCells) },
        { "numParticles", std::to_string(monitorData.numParticles) },
        { "numClusters", std::to_string(monitorData.numClusters) },
        { "numActiveClusters", std::to_string(monitorData.numClustersWithTokens) },
        { "numTokens", std::to_string(monitorData.numTokens) },
        { "sizeX", std::to_string(_config->universeSize.x) },
        { "sizeY", std::to_string(_config->universeSize.y) },
        { "numBlocks", std::to_string(_config->cudaConstants.NUM_BLOCKS) },
        { "numThreadsPerBlock", std::to_string(_config->cudaConstants.NUM_THREADS_PER_BLOCK) },
    });

    _state = State::Finished;
    _isReady = false;
}

void SendStatisticsJob::statisticsFromGpuReceived()
{
    if (State::StatisticsFromGpuRequested != _state) {
        return;
    }
    _isReady = true;
}
