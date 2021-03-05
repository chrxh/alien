#include "ColorizeDialogController.h"

#include <QInputDialog>

#include "Settings.h"

boost::optional<int> ColorizeDialogController::executeDialogAndReturnColorCode()
{
    bool ok;

    auto savedColorCode = GuiSettings::getSettingsValue(Const::ColorizeColorCodeKey, Const::ColorizeColorCodeDefault);

    auto colorCode = QInputDialog::getInt(
        nullptr, "Colorize selection", "Enter a color code (a value between 0 and 6):", savedColorCode, 0, 6, 1, &ok);

    if (!ok) {
        return boost::none;
    }

    GuiSettings::setSettingsValue(Const::ColorizeColorCodeKey, colorCode);

    return colorCode;
}
