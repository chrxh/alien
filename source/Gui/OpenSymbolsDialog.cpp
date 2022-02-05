#include "OpenSymbolsDialog.h"

#include <imgui.h>

#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"
#include "ImFileDialog.h"
#include "GlobalSettings.h"
#include "MessageDialog.h"

_OpenSymbolsDialog::_OpenSymbolsDialog(SimulationController const& simController)
    : _simController(simController)
{
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("dialogs.open symbols.starting path", path.string());
}

_OpenSymbolsDialog::~_OpenSymbolsDialog()
{
    GlobalSettings::getInstance().setStringState("dialogs.open symbols.starting path", _startingPath);
}

void _OpenSymbolsDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("OpenSymbolsDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        SymbolMap symbolMap;
        if (Serializer::deserializeSymbolsFromFile(firstFilename.string(), symbolMap)) {
            _simController->setSymbolMap(symbolMap);
        } else {
            MessageDialog::getInstance().show("Open symbols", "The selected file could not be opened.");
        }
    }
    ifd::FileDialog::Instance().Close();
}

void _OpenSymbolsDialog::show()
{
    ifd::FileDialog::Instance().Open("OpenSymbolsDialog", "Open symbols", "Symbols file (*.json){.json},.*", false, _startingPath);
}
