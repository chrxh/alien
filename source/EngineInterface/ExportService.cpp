#include "ExportService.h"

#include <fstream>

#include "EngineConstants.h"

bool ExportService::exportCollectedStatistics(std::vector<DataPointCollection> const& dataPoints, std::string const& filename)
{
    std::ofstream file;
    file.open(filename, std::ios_base::out);
    if (!file) {
        return false;
    }

    file << "Time step";
    auto writeLabelAllColors = [&file](auto const& name) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << name << " (color " << i << ")";
        }
    };
    writeLabelAllColors("Cells");
    writeLabelAllColors("Self-replicators");
    writeLabelAllColors("Viruses");
    writeLabelAllColors("Cell connections");
    writeLabelAllColors("Energy particles");
    writeLabelAllColors("Total energy");
    writeLabelAllColors("Genome size");
    writeLabelAllColors("Created cells");
    writeLabelAllColors("Attacks");
    writeLabelAllColors("Muscle activities");
    writeLabelAllColors("Transmitter activities");
    writeLabelAllColors("Defender activities");
    writeLabelAllColors("Injection activities");
    writeLabelAllColors("Completed injections");
    writeLabelAllColors("Nerve pulses");
    writeLabelAllColors("Neuron activities");
    writeLabelAllColors("Sensor activities");
    writeLabelAllColors("Sensor matches");
    writeLabelAllColors("Reconnector creations");
    writeLabelAllColors("Reconnector deletions");
    writeLabelAllColors("Detonations");

    file << std::endl;

    auto writeIntValueAllColors = [&file](DataPoint const& dataPoint) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << static_cast<uint64_t>(dataPoint.values[i]);
        }
    };
    auto writeDoubleValueAllColors = [&file](DataPoint const& dataPoint) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << dataPoint.values[i];
        }
    };
    for (auto const& dataPointCollection : dataPoints) {
        file << static_cast<uint64_t>(dataPointCollection.time);
        writeIntValueAllColors(dataPointCollection.numCells);
        writeIntValueAllColors(dataPointCollection.numSelfReplicators);
        writeIntValueAllColors(dataPointCollection.numViruses);
        writeIntValueAllColors(dataPointCollection.numConnections);
        writeIntValueAllColors(dataPointCollection.numParticles);
        writeDoubleValueAllColors(dataPointCollection.totalEnergy);
        writeDoubleValueAllColors(dataPointCollection.averageGenomeCells);
        writeDoubleValueAllColors(dataPointCollection.numCreatedCells);
        writeDoubleValueAllColors(dataPointCollection.numAttacks);
        writeDoubleValueAllColors(dataPointCollection.numMuscleActivities);
        writeDoubleValueAllColors(dataPointCollection.numTransmitterActivities);
        writeDoubleValueAllColors(dataPointCollection.numDefenderActivities);
        writeDoubleValueAllColors(dataPointCollection.numInjectionActivities);
        writeDoubleValueAllColors(dataPointCollection.numCompletedInjections);
        writeDoubleValueAllColors(dataPointCollection.numNervePulses);
        writeDoubleValueAllColors(dataPointCollection.numNeuronActivities);
        writeDoubleValueAllColors(dataPointCollection.numSensorActivities);
        writeDoubleValueAllColors(dataPointCollection.numSensorMatches);
        writeDoubleValueAllColors(dataPointCollection.numReconnectorCreated);
        writeDoubleValueAllColors(dataPointCollection.numReconnectorRemoved);
        writeDoubleValueAllColors(dataPointCollection.numDetonations);
        file << std::endl;
    }
    file.close();
    return true;
}

bool ExportService::exportStatistics(uint64_t timestep, StatisticsData const& statisticsData, std::string const& filename)
{
    std::ofstream file;
    file.open(filename, std::ios_base::out);
    if (!file) {
        return false;
    }

    auto writeLabelAllColors = [&file](auto const& name) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << name << " (color " << i << ")";
        }
    };
    file << "Time step";
    writeLabelAllColors("Cells");
    writeLabelAllColors("Self-replicators");
    writeLabelAllColors("Viruses");
    writeLabelAllColors("Cell connections");
    writeLabelAllColors("Energy particles");
    writeLabelAllColors("Total energy");
    writeLabelAllColors("Total genome cells");
    writeLabelAllColors("Created cells");
    writeLabelAllColors("Attacks");
    writeLabelAllColors("Muscle activities");
    writeLabelAllColors("Transmitter activities");
    writeLabelAllColors("Defender activities");
    writeLabelAllColors("Injection activities");
    writeLabelAllColors("Completed injections");
    writeLabelAllColors("Nerve pulses");
    writeLabelAllColors("Neuron activities");
    writeLabelAllColors("Sensor activities");
    writeLabelAllColors("Sensor matches");
    writeLabelAllColors("Reconnector creations");
    writeLabelAllColors("Reconnector deletions");
    writeLabelAllColors("Detonations");
    file << std::endl;

    auto writeIntValueAllColors = [&file](ColorVector<int> const& values) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << static_cast<uint64_t>(values[i]);
        }
    };
    auto writeInt64ValueAllColors = [&file](ColorVector<uint64_t> const& values) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << values[i];
        }
    };
    auto writeFloatValueAllColors = [&file](ColorVector<float> const& values) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << values[i];
        }
    };
    file << timestep;
    writeIntValueAllColors(statisticsData.timeline.timestep.numCells);
    writeIntValueAllColors(statisticsData.timeline.timestep.numSelfReplicators);
    writeIntValueAllColors(statisticsData.timeline.timestep.numViruses);
    writeIntValueAllColors(statisticsData.timeline.timestep.numConnections);
    writeIntValueAllColors(statisticsData.timeline.timestep.numParticles);
    writeFloatValueAllColors(statisticsData.timeline.timestep.totalEnergy);
    writeInt64ValueAllColors(statisticsData.timeline.timestep.numGenomeCells);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numCreatedCells);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numAttacks);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numMuscleActivities);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numTransmitterActivities);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numDefenderActivities);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numInjectionActivities);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numCompletedInjections);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numNervePulses);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numNeuronActivities);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numSensorActivities);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numSensorMatches);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numReconnectorCreated);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numReconnectorRemoved);
    writeInt64ValueAllColors(statisticsData.timeline.accumulated.numDetonations);

    file << std::endl;
    file.close();
    return true;
}
