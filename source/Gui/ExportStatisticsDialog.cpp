#include "ExportStatisticsDialog.h"

#include <fstream>
#include <ImFileDialog.h>

#include "Base/Definitions.h"
#include "Base/StringHelper.h"

#include "GlobalSettings.h"
#include "MessageDialog.h"

_ExportStatisticsDialog::_ExportStatisticsDialog()
{
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("dialogs.export statistics.starting path", path.string());
}

_ExportStatisticsDialog::~_ExportStatisticsDialog()
{
    GlobalSettings::getInstance().setStringState("dialogs.export statistics.starting path", _startingPath);
}

void _ExportStatisticsDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("ExportStatisticsDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        onSaveStatistics(firstFilename.string());

    }
    ifd::FileDialog::Instance().Close();
}

void _ExportStatisticsDialog::show(TimelineLongtermStatistics const& longtermStatistics)
{
    _statistics = longtermStatistics;
    ifd::FileDialog::Instance().Save("ExportStatisticsDialog", "Export statistics", "Comma-separated values (*.csv){.csv},.*", _startingPath);
}

void _ExportStatisticsDialog::onSaveStatistics(std::string const& filename)
{
    std::ofstream file;
    file.open(filename, std::ios_base::out);
    if (!file) {
        MessageDialog::getInstance().show("Export statistics", "The statistics could not be saved to the specified file.");
        return;
    }

    file << "time step";
    auto writeLabelAllColors = [&file](auto const& name) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << name << " (color " << i << ")";
        }
    };
    writeLabelAllColors("Cells");
    writeLabelAllColors("Cell connections");
    writeLabelAllColors("Energy particles");
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
    file << std::endl;

    auto writeIntValueAllColors = [&file](auto const& colorVector) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << static_cast<uint64_t>(colorVector[i]);
        }
    };
    auto writeDoubleValueAllColors = [&file](auto const& colorVector) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            file << ", " << StringHelper::format(toFloat(colorVector[i]), 8);
        }
    };
    for (auto const& dataPoint : _statistics.dataPoints) {
        file << static_cast<uint64_t>(dataPoint.time);
        writeIntValueAllColors(dataPoint.numCells);
        writeIntValueAllColors(dataPoint.numConnections);
        writeIntValueAllColors(dataPoint.numParticles);
        writeDoubleValueAllColors(dataPoint.numCreatedCells);
        writeDoubleValueAllColors(dataPoint.numAttacks);
        writeDoubleValueAllColors(dataPoint.numMuscleActivities);
        writeDoubleValueAllColors(dataPoint.numDefenderActivities);
        writeDoubleValueAllColors(dataPoint.numTransmitterActivities);
        writeDoubleValueAllColors(dataPoint.numInjectionActivities);
        writeDoubleValueAllColors(dataPoint.numCompletedInjections);
        writeDoubleValueAllColors(dataPoint.numNervePulses);
        writeDoubleValueAllColors(dataPoint.numNeuronActivities);
        writeDoubleValueAllColors(dataPoint.numSensorActivities);
        writeDoubleValueAllColors(dataPoint.numSensorMatches);
        file << std::endl;
    }
    file.close();
}
