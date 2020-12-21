#pragma once

#include "Base/Job.h"

#include "Web/Definitions.h"

#include "Definitions.h"

class SendStatisticsJob
    : public Job
{
    Q_OBJECT
public:
    SendStatisticsJob(
        string const& currentSimulationId,
        string const& currentToken,
        SimulationMonitor* simMonitor,
        WebAccess* webAccess,
        SimulationConfig const& config,
        QObject* parent);

    void process() override;
    bool isFinished() const override;
    bool isBlocking() const override;

private:
    void requestStatistics();
    void sendStatisticsToServer();

    Q_SLOT void statisticsFromGpuReceived();

    enum class State
    {
        Init,
        StatisticsFromGpuRequested,
        Finished
    };

    bool _isReady = true;
    State _state = State::Init;

    string _currentSimulationId;
    string _currentToken;

    SimulationMonitor* _simMonitor = nullptr;
    WebAccess* _webAccess = nullptr;
    SimulationConfig _config;
};
