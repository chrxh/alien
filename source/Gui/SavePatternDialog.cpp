#include <imgui.h>

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Serializer.h"
#include "ImFileDialog.h"
#include "SavePatternDialog.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"

_SavePatternDialog::_SavePatternDialog(SimulationController const& simController)
    : _simController(simController)
{
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("dialogs.save pattern.starting path", path.string());
}

_SavePatternDialog::~_SavePatternDialog()
{
    GlobalSettings::getInstance().setStringState("dialogs.save pattern.starting path", _startingPath);
}

void _SavePatternDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("SavePatternDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        auto content = _simController->getSelectedClusteredSimulationData(_includeClusters);

        if (!Serializer::serializeContentToFile(firstFilename.string(), content)) {
            MessageDialog::getInstance().show("Save pattern", "The selected pattern could not be saved to the specified file.");
        }
    }
    ifd::FileDialog::Instance().Close();
}

void _SavePatternDialog::show(bool includeClusters)
{
    _includeClusters = includeClusters;
    ifd::FileDialog::Instance().Save("SavePatternDialog", "Save pattern", "Pattern file (*.sim){.sim},.*", _startingPath);
}
