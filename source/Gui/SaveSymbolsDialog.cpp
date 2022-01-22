#include "SaveSymbolsDialog.h"

#include <imgui.h>

#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"
#include "ImFileDialog.h"

_SaveSymbolsDialog::_SaveSymbolsDialog(SimulationController const& simController)
    : _simController(simController)
{}

void _SaveSymbolsDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("SaveSymbolsDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        std::vector<std::filesystem::path> const& res = ifd::FileDialog::Instance().GetResults();
        auto firstFilename = res.front();

        Serializer serializer = std::make_shared<_Serializer>();
        serializer->serializeSymbolsToFile(firstFilename.string(), _simController->getSymbolMap());
    }
    ifd::FileDialog::Instance().Close();
}

void _SaveSymbolsDialog::show()
{
    ifd::FileDialog::Instance().Save("SaveSymbolsDialog", "Save symbols", "Symbols file (*.json){.json},.*");
}
