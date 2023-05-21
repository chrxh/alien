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
        result.numSelfReplicators[i] = numSelfReplicators[i] + other.numSelfReplicators[i];
        result.numViruses[i] = numViruses[i] + other.numViruses[i];
        result.numConnections[i] = numConnections[i] + other.numConnections[i];
        result.numParticles[i] = numParticles[i] + other.numParticles[i];
        result.averageGenomeSize[i] = averageGenomeSize[i] + other.averageGenomeSize[i];
        result.totalEnergy[i] = totalEnergy[i] + other.totalEnergy[i];
        result.numCreatedCells[i] = numCreatedCells[i] + other.numCreatedCells[i];
        result.numAttacks[i] = numAttacks[i] + other.numAttacks[i];
        result.numMuscleActivities[i] = numMuscleActivities[i] + other.numMuscleActivities[i];
        result.numDefenderActivities[i] = numDefenderActivities[i] + other.numDefenderActivities[i];
        result.numTransmitterActivities[i] = numTransmitterActivities[i] + other.numTransmitterActivities[i];
        result.numInjectionActivities[i] = numInjectionActivities[i] + other.numInjectionActivities[i];
        result.numCompletedInjections[i] = numCompletedInjections[i] + other.numCompletedInjections[i];
        result.numNervePulses[i] = numNervePulses[i] + other.numNervePulses[i];
        result.numNeuronActivities[i] = numNeuronActivities[i] + other.numNeuronActivities[i];
        result.numSensorActivities[i] = numSensorActivities[i] + other.numSensorActivities[i];
        result.numSensorMatches[i] = numSensorMatches[i] + other.numSensorMatches[i];
    }
    result.sumAverageGenomeSize = sumAverageGenomeSize + other.sumAverageGenomeSize;
    return result;
}

DataPoint DataPoint::operator/(double divisor) const
{
    DataPoint result;
    result.time = time / divisor;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.numCells[i] = numCells[i] / divisor;
        result.numSelfReplicators[i] = numSelfReplicators[i] / divisor;
        result.numViruses[i] = numViruses[i] / divisor;
        result.numConnections[i] = numConnections[i] / divisor;
        result.numParticles[i] = numParticles[i] / divisor;
        result.averageGenomeSize[i] = averageGenomeSize[i] / divisor;
        result.totalEnergy[i] = totalEnergy[i] / divisor;
        result.numCreatedCells[i] = numCreatedCells[i] / divisor;
        result.numAttacks[i] = numAttacks[i] / divisor;
        result.numMuscleActivities[i] = numMuscleActivities[i] / divisor;
        result.numDefenderActivities[i] = numDefenderActivities[i] / divisor;
        result.numTransmitterActivities[i] = numTransmitterActivities[i] / divisor;
        result.numInjectionActivities[i] = numInjectionActivities[i] / divisor;
        result.numCompletedInjections[i] = numCompletedInjections[i] / divisor;
        result.numNervePulses[i] = numNervePulses[i] / divisor;
        result.numNeuronActivities[i] = numNeuronActivities[i] / divisor;
        result.numSensorActivities[i] = numSensorActivities[i] / divisor;
        result.numSensorMatches[i] = numSensorMatches[i] / divisor;
    }
    result.sumAverageGenomeSize = sumAverageGenomeSize / divisor;
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
        auto sumGenomeBytes = 0.0;
        auto sumNumGenomes = 0.0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.numCells[i] = toDouble(data.timestep.numCells[i]);
            result.numSelfReplicators[i] = toDouble(data.timestep.numSelfReplicators[i]);
            result.numViruses[i] = toDouble(data.timestep.numViruses[i]);
            result.numConnections[i] = toDouble(data.timestep.numConnections[i]);
            result.numParticles[i] = toDouble(data.timestep.numParticles[i]);
            result.averageGenomeSize[i] = toDouble(data.timestep.numGenomeBytes[i]);
            auto numGenomes = result.numSelfReplicators[i];
            sumGenomeBytes += result.averageGenomeSize[i];
            sumNumGenomes += numGenomes;
            if (numGenomes > 0) {
                result.averageGenomeSize[i] /= numGenomes;
            }
            result.totalEnergy[i] = toDouble(data.timestep.totalEnergy[i]);
        }
        result.sumAverageGenomeSize = sumNumGenomes > 0 ? sumGenomeBytes / sumNumGenomes : sumGenomeBytes;

        auto deltaTimesteps = lastTimestep ? toDouble(timestep) - toDouble(*lastTimestep) : 1.0;
        if (deltaTimesteps < NEAR_ZERO) {
            deltaTimesteps = 1.0;
        }

        auto lastDataValue = lastData.value_or(data);
        for (int i = 0; i < MAX_COLORS; ++i) {
            auto numCells = std::max(result.numCells[i], 1.0);
            auto calcNumProcesses = [&](uint64_t value, uint64_t lastValue) {
                if (lastValue > value) {
                    return 0.0;
                }
                return toDouble(value - lastValue) / deltaTimesteps / numCells;
            };

            result.numCreatedCells[i] = calcNumProcesses(data.accumulated.numCreatedCells[i], lastDataValue.accumulated.numCreatedCells[i]);
            result.numAttacks[i] = calcNumProcesses(data.accumulated.numAttacks[i], lastDataValue.accumulated.numAttacks[i]);
            result.numMuscleActivities[i] = calcNumProcesses(data.accumulated.numMuscleActivities[i], lastDataValue.accumulated.numMuscleActivities[i]);
            result.numDefenderActivities[i] = calcNumProcesses(data.accumulated.numDefenderActivities[i], lastDataValue.accumulated.numDefenderActivities[i]);
            result.numTransmitterActivities[i] =
                calcNumProcesses(data.accumulated.numTransmitterActivities[i], lastDataValue.accumulated.numTransmitterActivities[i]);
            result.numInjectionActivities[i] =
                calcNumProcesses(data.accumulated.numInjectionActivities[i], lastDataValue.accumulated.numInjectionActivities[i]);
            result.numCompletedInjections[i] =
                calcNumProcesses(data.accumulated.numCompletedInjections[i], lastDataValue.accumulated.numCompletedInjections[i]);
            result.numNervePulses[i] = calcNumProcesses(data.accumulated.numNervePulses[i], lastDataValue.accumulated.numNervePulses[i]);
            result.numNeuronActivities[i] = calcNumProcesses(data.accumulated.numNeuronActivities[i], lastDataValue.accumulated.numNeuronActivities[i]);
            result.numSensorActivities[i] = calcNumProcesses(data.accumulated.numSensorActivities[i], lastDataValue.accumulated.numSensorActivities[i]);
            result.numSensorMatches[i] = calcNumProcesses(data.accumulated.numSensorMatches[i], lastDataValue.accumulated.numSensorMatches[i]);
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
