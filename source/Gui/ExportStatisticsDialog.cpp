#include "ExportStatisticsDialog.h"

#include <fstream>
#include <ImFileDialog.h>

#include "Base/Definitions.h"

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

void _ExportStatisticsDialog::show(LongtermStatistics const& longtermStatistics)
{
    _statistics = longtermStatistics;
    ifd::FileDialog::Instance().Save("ExportStatisticsDialog", "Export statistics", "Comma-separated values (*.csv){.csv},.*", _startingPath);
}

void _ExportStatisticsDialog::onSaveStatistics(std::string const& filename)
{
    for (auto const& data : _statistics.datas) {
        CHECK(_statistics.timestepHistory.size() == _statistics.datas[0].size());
    }

    std::ofstream file;
    file.open(filename, std::ios_base::out);
    if (!file) {
        MessageDialog::getInstance().show("Export statistics", "The statistics could not be saved to the specified file.");
        return;
    }

    file << "time step, cells, particles, tokens, created cells, successful attacks, failed attacks, muscle activities" << std::endl;
    for (int i = 0; i < _statistics.timestepHistory.size(); ++i) {
        file << static_cast<uint64_t>(_statistics.timestepHistory.at(i)) << ", " << static_cast<uint64_t>(_statistics.datas[0].at(i)) << ", "
             << static_cast<uint64_t>(_statistics.datas[1].at(i)) << ", " << static_cast<uint64_t>(_statistics.datas[2].at(i)) << ", "
             << static_cast<uint64_t>(_statistics.datas[3].at(i)) << ", " << static_cast<uint64_t>(_statistics.datas[4].at(i)) << ", "
             << static_cast<uint64_t>(_statistics.datas[5].at(i)) << ", " << static_cast<uint64_t>(_statistics.datas[6].at(i)) << std::endl;
    }
    file.close();
}
