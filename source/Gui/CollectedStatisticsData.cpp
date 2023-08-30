#include "CollectedStatisticsData.h"

#include <cmath>
#include <imgui.h>

#include "EngineInterface/StatisticsData.h"

DataPoint DataPoint::operator+(DataPoint const& other) const
{
    DataPoint result;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.values[i] = values[i] + other.values[i];
    }
    result.summedValues = summedValues + other.summedValues;
    return result;
}

DataPoint DataPoint::operator/(double divisor) const
{
    DataPoint result;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.values[i] = values[i] / divisor;
    }
    result.summedValues = summedValues / divisor;
    return result;
}

DataPointCollection DataPointCollection::operator+(DataPointCollection const& other) const
{
    DataPointCollection result;
    result.time = time + other.time;
    result.numCells = numCells + other.numCells;
    result.numSelfReplicators = numSelfReplicators + other.numSelfReplicators;
    result.numViruses = numViruses + other.numViruses;
    result.numConnections = numConnections + other.numConnections;
    result.numParticles = numParticles + other.numParticles;
    result.averageGenomeNodes = averageGenomeNodes + other.averageGenomeNodes;
    result.totalEnergy = totalEnergy + other.totalEnergy;
    result.numCreatedCells = numCreatedCells + other.numCreatedCells;
    result.numAttacks = numAttacks + other.numAttacks;
    result.numMuscleActivities = numMuscleActivities + other.numMuscleActivities;
    result.numDefenderActivities = numDefenderActivities + other.numDefenderActivities;
    result.numTransmitterActivities = numTransmitterActivities + other.numTransmitterActivities;
    result.numInjectionActivities = numInjectionActivities + other.numInjectionActivities;
    result.numCompletedInjections = numCompletedInjections + other.numCompletedInjections;
    result.numNervePulses = numNervePulses + other.numNervePulses;
    result.numNeuronActivities = numNeuronActivities + other.numNeuronActivities;
    result.numSensorActivities = numSensorActivities + other.numSensorActivities;
    result.numSensorMatches = numSensorMatches + other.numSensorMatches;
    return result;
}

DataPointCollection DataPointCollection::operator/(double divisor) const
{
    DataPointCollection result;
    result.time = time / divisor;
    result.numCells = numCells / divisor;
    result.numSelfReplicators = numSelfReplicators / divisor;
    result.numViruses = numViruses / divisor;
    result.numConnections = numConnections / divisor;
    result.numParticles = numParticles / divisor;
    result.averageGenomeNodes = averageGenomeNodes / divisor;
    result.totalEnergy = totalEnergy / divisor;
    result.numCreatedCells = numCreatedCells / divisor;
    result.numAttacks = numAttacks / divisor;
    result.numMuscleActivities = numMuscleActivities / divisor;
    result.numDefenderActivities = numDefenderActivities / divisor;
    result.numTransmitterActivities = numTransmitterActivities / divisor;
    result.numInjectionActivities = numInjectionActivities / divisor;
    result.numCompletedInjections = numCompletedInjections / divisor;
    result.numNervePulses = numNervePulses / divisor;
    result.numNeuronActivities = numNeuronActivities / divisor;
    result.numSensorActivities = numSensorActivities / divisor;
    result.numSensorMatches = numSensorMatches / divisor;
    return result;
}

void TimelineLiveStatistics::truncate()
{
    if (!dataPointCollectionHistory.empty() && dataPointCollectionHistory.back().time - dataPointCollectionHistory.front().time > (MaxLiveHistory + 1.0)) {
        dataPointCollectionHistory.erase(dataPointCollectionHistory.begin());
    }
}

namespace
{
    template<typename T>
    DataPoint getDataPointForTimestepProperty(ColorVector<T> const& values)
    {
        DataPoint result;
        result.summedValues = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.values[i] = toDouble(values[i]);
            result.summedValues += result.values[i];
        }
        return result;
    }

    DataPoint getDataPointForAverageGenomeNodes(ColorVector<uint64_t> const& numGenomeNodes, ColorVector<int> const& numSelfReplicators)
    {
        DataPoint result;
        auto sumNumGenomeNodes = 0.0;
        auto sumNumSelfReplicators = 0.0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.values[i] = toDouble(numGenomeNodes[i]);
            sumNumGenomeNodes += result.values[i];
            sumNumSelfReplicators += numSelfReplicators[i];
            if (numSelfReplicators[i] > 0) {
                result.values[i] /= numSelfReplicators[i];
            }
        }
        result.summedValues = sumNumSelfReplicators > 0 ? sumNumGenomeNodes / sumNumSelfReplicators : sumNumGenomeNodes;
        return result;
    }

    DataPoint getDataPointForProcessProperty(
        ColorVector<uint64_t> const& values,
        ColorVector<uint64_t> const& lastValues,
        ColorVector<int> const& numCells,
        double deltaTimesteps)
    {
        DataPoint result;
        result.summedValues = 0;
        auto sumNumCells = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (lastValues[i] > values[i] || numCells[i] == 0) {
                result.values[i] = 0;
            } else {
                result.values[i] = toDouble(values[i] - lastValues[i]) / deltaTimesteps / toDouble(numCells[i]);
                result.summedValues += toDouble(values[i] - lastValues[i]) / deltaTimesteps;
                sumNumCells += numCells[i];
            }
        }
        result.summedValues /= toDouble(sumNumCells);
        return result;
    }

    //time on DataPointCollection will not be set
    DataPointCollection convertToDataPointCollection(
        TimelineStatistics const& data,
        uint64_t timestep,
        std::optional<TimelineStatistics> const& lastData,
        std::optional<uint64_t> lastTimestep)
    {
        DataPointCollection result;

        result.numCells = getDataPointForTimestepProperty(data.timestep.numCells);
        result.numSelfReplicators = getDataPointForTimestepProperty(data.timestep.numSelfReplicators);
        result.numViruses = getDataPointForTimestepProperty(data.timestep.numViruses);
        result.numConnections = getDataPointForTimestepProperty(data.timestep.numConnections);
        result.numParticles = getDataPointForTimestepProperty(data.timestep.numParticles);
        result.averageGenomeNodes = getDataPointForAverageGenomeNodes(data.timestep.numGenomeNodes, data.timestep.numSelfReplicators);
        result.totalEnergy = getDataPointForTimestepProperty(data.timestep.totalEnergy);

        auto deltaTimesteps = lastTimestep ? toDouble(timestep) - toDouble(*lastTimestep) : 1.0;
        if (deltaTimesteps < NEAR_ZERO) {
            deltaTimesteps = 1.0;
        }

        auto lastDataValue = lastData.value_or(data);
        result.numCreatedCells =
            getDataPointForProcessProperty(data.accumulated.numCreatedCells, lastDataValue.accumulated.numCreatedCells, data.timestep.numCells, deltaTimesteps);
        result.numAttacks =
            getDataPointForProcessProperty(data.accumulated.numAttacks, lastDataValue.accumulated.numAttacks, data.timestep.numCells, deltaTimesteps);
        result.numMuscleActivities = getDataPointForProcessProperty(
            data.accumulated.numMuscleActivities, lastDataValue.accumulated.numMuscleActivities, data.timestep.numCells, deltaTimesteps);
        result.numDefenderActivities = getDataPointForProcessProperty(
            data.accumulated.numDefenderActivities, lastDataValue.accumulated.numDefenderActivities, data.timestep.numCells, deltaTimesteps);
        result.numTransmitterActivities = getDataPointForProcessProperty(
            data.accumulated.numTransmitterActivities, lastDataValue.accumulated.numTransmitterActivities, data.timestep.numCells, deltaTimesteps);
        result.numInjectionActivities = getDataPointForProcessProperty(
            data.accumulated.numInjectionActivities, lastDataValue.accumulated.numInjectionActivities, data.timestep.numCells, deltaTimesteps);
        result.numCompletedInjections = getDataPointForProcessProperty(
            data.accumulated.numCompletedInjections, lastDataValue.accumulated.numCompletedInjections, data.timestep.numCells, deltaTimesteps);
        result.numNervePulses = getDataPointForProcessProperty(data.accumulated.numNervePulses, lastDataValue.accumulated.numNervePulses, data.timestep.numCells, deltaTimesteps);
        result.numNeuronActivities = getDataPointForProcessProperty(
            data.accumulated.numNeuronActivities, lastDataValue.accumulated.numNeuronActivities, data.timestep.numCells, deltaTimesteps);
        result.numSensorActivities = getDataPointForProcessProperty(
            data.accumulated.numSensorActivities, lastDataValue.accumulated.numSensorActivities, data.timestep.numCells, deltaTimesteps);
        result.numSensorMatches = getDataPointForProcessProperty(
            data.accumulated.numSensorMatches, lastDataValue.accumulated.numSensorMatches, data.timestep.numCells, deltaTimesteps);

        return result;
    }
}

void TimelineLiveStatistics::add(TimelineStatistics const& data, uint64_t timestep)
{
    truncate();

    timepoint += ImGui::GetIO().DeltaTime;

    auto newDataPoint = convertToDataPointCollection(data, timestep, lastData, lastTimestep);
    newDataPoint.time = timepoint;
    dataPointCollectionHistory.emplace_back(newDataPoint);
    lastData = data;
    lastTimestep = timestep;
}

void TimelineLongtermStatistics::add(TimelineStatistics const& data, uint64_t timestep)
{
    if (!lastData || toDouble(timestep) - dataPointCollectionHistory.back().time > longtermTimestepDelta) {
        auto newDataPoint = convertToDataPointCollection(data, timestep, lastData, lastTimestep);
        newDataPoint.time = toDouble(timestep);
        dataPointCollectionHistory.emplace_back(newDataPoint);
        lastData = data;
        lastTimestep = timestep;

        if (dataPointCollectionHistory.size() > 1000) {
            std::vector<DataPointCollection> newDataPoints;
            newDataPoints.reserve(dataPointCollectionHistory.size() / 2);
            for (size_t i = 0; i < (dataPointCollectionHistory.size() - 1) / 2; ++i) {
                DataPointCollection newDataPoint = (dataPointCollectionHistory.at(i * 2) + dataPointCollectionHistory.at(i * 2 + 1)) / 2.0;
                newDataPoint.time = dataPointCollectionHistory.at(i * 2).time;
                newDataPoints.emplace_back(newDataPoint);
            }
            newDataPoints.emplace_back(dataPointCollectionHistory.back());
            dataPointCollectionHistory.swap(newDataPoints);

            longtermTimestepDelta *= 2;
        }
    }
}
