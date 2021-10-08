#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineImpl/Definitions.h"

#include "Definitions.h"

class _StatisticsWindow
{
public:
    _StatisticsWindow(SimulationController const& simController);

    void reset();
    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    void processLiveStatistics();
    void processLongtermStatistics();
    void processLivePlot(int row, std::vector<float> const& valueHistory);

    void updateData();

    SimulationController _simController;

    bool _on = true;
    bool _live = true;

    struct LiveStatistics
    {
        float timepoint = 0.0f; //in seconds
        float history = 10.0f;  //in seconds
        std::vector<float> timepointsHistory;
        std::vector<float> numCellsHistory;
        std::vector<float> numParticlesHistory;
        std::vector<float> numTokensHistory;
        std::vector<float> numCreatedCellsHistory;
        std::vector<float> numSuccessfulAttacksHistory;
        std::vector<float> numFailedAttacksHistory;
        std::vector<float> numMuscleActivitiesHistory;

        void truncate();
        void add(OverallStatistics const& statistics);
    };
    LiveStatistics _liveStatistics;

    struct LongtermStatistics
    {
        std::vector<float> timestepHistory;
        std::vector<float> numCellsHistory;
        std::vector<float> numParticlesHistory;
        std::vector<float> numTokensHistory;
        std::vector<float> numCreatedCellsHistory;
        std::vector<float> numSuccessfullAttacksHistory;
        std::vector<float> numFailedAttacksHistory;
        std::vector<float> numMuscleActivitiesHistory;

        void add(OverallStatistics const& statistics);
    };
    LongtermStatistics _longtermStatistics;
};
