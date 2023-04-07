#include "CollectedStatisticsData.h"

#include <cmath>
#include <imgui.h>

#include "EngineInterface/StatisticsData.h"

DataPoint DataPoint::operator+(DataPoint const& other) const
{
    DataPoint result;
    result.time = time + other.time;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.numCells[i] = numCells[i] + other.numCells[i];
        result.numConnections[i] = numConnections[i] + other.numConnections[i];
        result.numParticles[i] = numParticles[i] + other.numParticles[i];
        result.numCreatedCells[i] = numCreatedCells[i] + other.numCreatedCells[i];
        result.numAttacks[i] = numAttacks[i] + other.numAttacks[i];
        result.numMuscleActivities[i] = numMuscleActivities[i] + other.numMuscleActivities[i];
        result.numDefenderActivities[i] = numDefenderActivities[i] + other.numDefenderActivities[i];
        result.numTransmitterActivities[i] = numTransmitterActivities[i] + other.numTransmitterActivities[i];
        result.numInjectionActivities[i] = numInjectionActivities[i] + other.numInjectionActivities[i];
        result.numCompletedInjections[i] = numCompletedInjections[i] + other.numCompletedInjections[i];
        result.numNervePulses[i] = numNervePulses[i] + other.numNervePulses[i];
        result.numNeuronActivities[i] = numNeuronActivities[i] + other.numNeuronActivities[i];
    }
    return result;
}

DataPoint DataPoint::operator/(double divisor) const
{
    DataPoint result;
    result.time = time / divisor;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.numCells[i] = numCells[i] / divisor;
        result.numConnections[i] = numConnections[i] / divisor;
        result.numParticles[i] = numParticles[i] / divisor;
        result.numCreatedCells[i] = numCreatedCells[i] / divisor;
        result.numAttacks[i] = numAttacks[i] / divisor;
        result.numMuscleActivities[i] = numMuscleActivities[i] / divisor;
        result.numDefenderActivities[i] = numDefenderActivities[i] / divisor;
        result.numTransmitterActivities[i] = numTransmitterActivities[i] / divisor;
        result.numInjectionActivities[i] = numInjectionActivities[i] / divisor;
        result.numCompletedInjections[i] = numCompletedInjections[i] / divisor;
        result.numNervePulses[i] = numNervePulses[i] / divisor;
        result.numNeuronActivities[i] = numNeuronActivities[i] / divisor;
    }
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
            result.numCells[i] = toDouble(data.timestep.numCells[i]);
            result.numConnections[i] = toDouble(data.timestep.numConnections[i]);
            result.numParticles[i] = toDouble(data.timestep.numParticles[i]);
        }
        if (lastData) {
            auto deltaTimesteps = toDouble(timestep) - toDouble(*lastTimestep);
            if (deltaTimesteps < NEAR_ZERO) {
                deltaTimesteps = 1.0;
             }
            for (int i = 0; i < MAX_COLORS; ++i) {
                result.numCreatedCells[i] = toDouble(data.accumulated.numCreatedCells[i] - lastData->accumulated.numCreatedCells[i]) / deltaTimesteps;
                result.numAttacks[i] = toDouble(data.accumulated.numAttacks[i] - lastData->accumulated.numAttacks[i]) / deltaTimesteps;
                result.numMuscleActivities[i] =
                    toDouble(data.accumulated.numMuscleActivities[i] - lastData->accumulated.numMuscleActivities[i]) / deltaTimesteps;
                result.numDefenderActivities[i] =
                    toDouble(data.accumulated.numDefenderActivities[i] - lastData->accumulated.numDefenderActivities[i]) / deltaTimesteps;
                result.numTransmitterActivities[i] =
                    toDouble(data.accumulated.numTransmitterActivities[i] - lastData->accumulated.numTransmitterActivities[i]) / deltaTimesteps;
                result.numInjectionActivities[i] = toDouble(data.accumulated.numInjectionActivities[i] - lastData->accumulated.numInjectionActivities[i]) / deltaTimesteps;
                result.numCompletedInjections[i] =
                    toDouble(data.accumulated.numCompletedInjections[i] - lastData->accumulated.numCompletedInjections[i]) / deltaTimesteps;
                result.numNervePulses[i] = toDouble(data.accumulated.numNervePulses[i] - lastData->accumulated.numNervePulses[i]) / deltaTimesteps;
                result.numNeuronActivities[i] =
                    toDouble(data.accumulated.numNeuronActivities[i] - lastData->accumulated.numNeuronActivities[i]) / deltaTimesteps;
            }
        } else {
            for (int i = 0; i < MAX_COLORS; ++i) {
                result.numCreatedCells[i] = toDouble(data.accumulated.numCreatedCells[i]);
                result.numAttacks[i] = toDouble(data.accumulated.numAttacks[i]);
                result.numMuscleActivities[i] = toDouble(data.accumulated.numMuscleActivities[i]);
                result.numDefenderActivities[i] = toDouble(data.accumulated.numDefenderActivities[i]);
                result.numTransmitterActivities[i] = toDouble(data.accumulated.numTransmitterActivities[i]);
                result.numInjectionActivities[i] = toDouble(data.accumulated.numInjectionActivities[i]);
                result.numCompletedInjections[i] = toDouble(data.accumulated.numCompletedInjections[i]);
                result.numNervePulses[i] = toDouble(data.accumulated.numNervePulses[i]);
                result.numNeuronActivities[i] = toDouble(data.accumulated.numNeuronActivities[i]);
            }
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
