#include "CollectedStatisticsData.h"

#include <cmath>
#include <imgui.h>

#include "EngineInterface/StatisticsData.h"

DataPoint DataPoint::operator+(DataPoint const& other) const
{
    DataPoint result;
    result.time = time + other.time;
    result.numCells = numCells + other.numCells;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.numCellsByColor[i] = numCellsByColor[i] + other.numCellsByColor[i];
    }
    result.numConnections = numConnections + other.numConnections;
    result.numParticles = numParticles + other.numParticles;
    result.numCreatedCells = numCreatedCells + other.numCreatedCells;
    result.numSuccessfulAttacks = numSuccessfulAttacks + other.numSuccessfulAttacks;
    result.numFailedAttacks = numFailedAttacks + other.numFailedAttacks;
    result.numMuscleActivities = numMuscleActivities + other.numMuscleActivities;
    return result;
}

DataPoint DataPoint::operator/(double divisor) const
{
    DataPoint result;
    result.time = time / divisor;
    result.numCells = numCells / divisor;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.numCellsByColor[i] = numCellsByColor[i] / divisor;
    }
    result.numConnections = numConnections / divisor;
    result.numParticles = numParticles / divisor;
    result.numCreatedCells = numCreatedCells / divisor;
    result.numSuccessfulAttacks = numSuccessfulAttacks / divisor;
    result.numFailedAttacks = numFailedAttacks / divisor;
    result.numMuscleActivities = numMuscleActivities / divisor;
    return result;
}

void TimelineLiveStatistics::truncate()
{
    if (!dataPoints.empty() && dataPoints.back().time - dataPoints.front().time > (MaxLiveHistory + 1.0)) {
        dataPoints.erase(dataPoints.begin());
    }
}

namespace
{
    //time on DataPoint will not be set
    DataPoint convertToDataPoint(
        TimelineStatistics const& data,
        uint64_t timestep,
        std::optional<TimelineStatistics> const& lastData,
        std::optional<uint64_t> lastTimestep)
    {
        DataPoint result;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.numCellsByColor[i] = toDouble(data.timestep.numCellsByColor[i]);
            result.numCells += toDouble(data.timestep.numCellsByColor[i]);
        }
        result.numConnections = toDouble(data.timestep.numConnections);
        result.numParticles = toDouble(data.timestep.numParticles);
        if (lastData) {
            auto deltaTimesteps = toDouble(timestep) - toDouble(*lastTimestep);
            result.numCreatedCells = toDouble(data.accumulated.numCreatedCells - lastData->accumulated.numCreatedCells) / deltaTimesteps;
            result.numSuccessfulAttacks = toDouble(data.accumulated.numSuccessfulAttacks - lastData->accumulated.numSuccessfulAttacks) / deltaTimesteps;
            result.numFailedAttacks = toDouble(data.accumulated.numFailedAttacks - lastData->accumulated.numFailedAttacks) / deltaTimesteps;
            result.numMuscleActivities = toDouble(data.accumulated.numMuscleActivities - lastData->accumulated.numMuscleActivities) / deltaTimesteps;
        } else {
            result.numCreatedCells = toDouble(data.accumulated.numCreatedCells);
            result.numSuccessfulAttacks = toDouble(data.accumulated.numSuccessfulAttacks);
            result.numFailedAttacks = toDouble(data.accumulated.numFailedAttacks);
            result.numMuscleActivities = toDouble(data.accumulated.numMuscleActivities);
        }
        return result;
    }
}

void TimelineLiveStatistics::add(TimelineStatistics const& data, uint64_t timestep)
{
    truncate();

    timepoint += ImGui::GetIO().DeltaTime;

    auto newDataPoint = convertToDataPoint(data, timestep, lastData, lastTimestep);
    newDataPoint.time = timepoint;
    dataPoints.emplace_back(newDataPoint);
    lastData = data;
    lastTimestep = timestep;
}

void TimelineLongtermStatistics::add(TimelineStatistics const& data, uint64_t timestep)
{
    if (!lastData || toDouble(timestep) - dataPoints.back().time > longtermTimestepDelta) {
        auto newDataPoint = convertToDataPoint(data, timestep, lastData, lastTimestep);
        newDataPoint.time = toDouble(timestep);
        dataPoints.emplace_back(newDataPoint);
        lastData = data;
        lastTimestep = timestep;

        if (dataPoints.size() > 1000) {
            std::vector<DataPoint> newDataPoints;
            newDataPoints.reserve(dataPoints.size() / 2);
            for (size_t i = 0; i < (dataPoints.size() - 1) / 2; ++i) {
                DataPoint newDataPoint = (dataPoints.at(i * 2) + dataPoints.at(i * 2 + 1)) / 2.0;
                newDataPoint.time = dataPoints.at(i * 2).time;
                newDataPoints.emplace_back(newDataPoint);
            }
            newDataPoints.emplace_back(dataPoints.back());
            dataPoints.swap(newDataPoints);

            longtermTimestepDelta *= 2;
        }
    }
}
