#include "EditSimulationDialog.h"

#include "MessageDialog.h"
#include "BrowserWindow.h"

_EditSimulationDialog::_EditSimulationDialog(BrowserWindow const& browserWindow)
    : _AlienDialog("")
    , _browserWindow(browserWindow)
{}

_EditSimulationDialog::~_EditSimulationDialog()
{
}

void _EditSimulationDialog::open(NetworkResourceTreeTO const& treeTO)
{
    changeTitle("Change name or description");
    _AlienDialog::open();
}

void _EditSimulationDialog::processIntern()
{
}
