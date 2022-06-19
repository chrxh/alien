#include "GenericOpenFileDialog.h"

#include <imgui.h>
#include <ImFileDialog.h>

#include "AlienImGui.h"

GenericOpenFileDialog& GenericOpenFileDialog::getInstance()
{
    static GenericOpenFileDialog instance;
    return instance;
}

void GenericOpenFileDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("GenericOpenDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        _actionFunc(ifd::FileDialog::Instance().GetResult());
    }
    ifd::FileDialog::Instance().Close();
}

void GenericOpenFileDialog::show(
    std::string const& title,
    std::string const& filter,
    std::string startingPath,
    std::function<void(std::filesystem::path const&)> const& actionFunc)
{
    _actionFunc = actionFunc;
    ifd::FileDialog::Instance().Open("GenericOpenDialog", title, filter, false, startingPath);
}
