#include <imgui.h>

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Serializer.h"
#include "ImFileDialog.h"
#include "SavePatternDialog.h"

_SavePatternDialog::_SavePatternDialog(SimulationController const& simController)
    : _simController(simController)
{}

void _SavePatternDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("SavePatternDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        std::vector<std::filesystem::path> const& res = ifd::FileDialog::Instance().GetResults();
        auto firstFilename = res.front();

        auto content = _simController->getSelectedClusteredSimulationData(_includeClusters);

        Serializer serializer = std::make_shared<_Serializer>();
        serializer->serializeContentToFile(firstFilename.string(), content);
    }
    ifd::FileDialog::Instance().Close();
}

void _SavePatternDialog::show(bool includeClusters)
{
    _includeClusters = includeClusters;
    ifd::FileDialog::Instance().Save("SavePatternDialog", "Save pattern", "Pattern file (*.sim){.sim},.*");
}
