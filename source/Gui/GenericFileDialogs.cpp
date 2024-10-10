#include "GenericFileDialogs.h"

#include <imgui.h>
#include <ImFileDialog.h>

#include "AlienImGui.h"
#include "WindowController.h"

void GenericFileDialogs::showOpenFileDialog(
    std::string const& title,
    std::string const& filter,
    std::string startingPath,
    std::function<void(std::filesystem::path const&)> const& actionFunc)
{
    _actionFunc = actionFunc;
    ifd::FileDialog::Instance().Open("GenericFileDialog", title, filter, false, startingPath);
}

void GenericFileDialogs::showSaveFileDialog(
    std::string const& title,
    std::string const& filter,
    std::string startingPath,
    std::function<void(std::filesystem::path const&)> const& actionFunc)
{
    _actionFunc = actionFunc;
    ifd::FileDialog::Instance().Save("GenericFileDialog", title, filter, startingPath);
}

void GenericFileDialogs::process()
{
    if (!ifd::FileDialog::Instance().IsDone("GenericFileDialog", WindowController::getContentScaleFactor())) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        _actionFunc(ifd::FileDialog::Instance().GetResult());
    }
    ifd::FileDialog::Instance().Close();
}
