#include "OpenSymbolsDialog.h"

#include <imgui.h>

#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"
#include "ImFileDialog.h"

_OpenSymbolsDialog::_OpenSymbolsDialog(SimulationController const& simController)
    : _simController(simController)
{}

void _OpenSymbolsDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("OpenSymbolsDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        const std::vector<std::filesystem::path>& res = ifd::FileDialog::Instance().GetResults();
        auto firstFilename = res.front();

        Serializer serializer = std::make_shared<_Serializer>();

        SymbolMap symbolMap;
        if (serializer->deserializeSymbolsFromFile(firstFilename.string(), symbolMap)) {
            _simController->setSymbolMap(symbolMap);
        }
    }
    ifd::FileDialog::Instance().Close();
}

void _OpenSymbolsDialog::show()
{
    ifd::FileDialog::Instance().Open("OpenSymbolsDialog", "Open symbols", "Symbols file (*.json){.json},.*", false);
}
