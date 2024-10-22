#include "GenericFileDialog.h"

#include <ImFileDialog.h>

#include "MainLoopEntityController.h"
#include "WindowController.h"

void GenericFileDialog::showOpenFileDialog(
    std::string const& title,
    std::string const& filter,
    std::string startingPath,
    std::function<void(std::filesystem::path const&)> const& actionFunc)
{
    _actionFunc = actionFunc;
    ifd::FileDialog::Instance().Open("GenericFileDialog", title, filter, false, startingPath);
}

void GenericFileDialog::showSaveFileDialog(
    std::string const& title,
    std::string const& filter,
    std::string startingPath,
    std::function<void(std::filesystem::path const&)> const& actionFunc)
{
    _actionFunc = actionFunc;
    ifd::FileDialog::Instance().Save("GenericFileDialog", title, filter, startingPath);
}

void GenericFileDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("GenericFileDialog", WindowController::get().getContentScaleFactor())) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        _actionFunc(ifd::FileDialog::Instance().GetResult());
    }
    ifd::FileDialog::Instance().Close();
}
