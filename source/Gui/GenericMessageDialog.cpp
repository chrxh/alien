#include "GenericMessageDialog.h"

#include <algorithm>

#include <boost/algorithm/string.hpp>

#include <imgui.h>

#include <Base/LoggingService.h>
#include <Base/MarkdownParser.h>
#include <Base/WebLinkHelper.h>

#include "AlienGui.h"
#include "StyleService.h"

void GenericMessageDialog::processIntern()
{
    switch (_dialogType) {
    case DialogType::Information:
        processInformation();
        break;
    case DialogType::YesNo:
        processYesNo();
        break;
    }
}

void GenericMessageDialog::information(std::string const& title, std::string const& message)
{
    _title = title;
    _message = message;
    _markdownMessage.reset();
    _dialogType = DialogType::Information;
    log(Priority::Important, "message dialog showing: '" + message + "'");

    AlienDialog::open();
    changeTitle(title);
}

void GenericMessageDialog::information(std::string const& title, std::vector<PersisterErrorInfo> const& errors)
{
    std::vector<std::string> errorMessages;
    for (auto const& error : errors) {
        errorMessages.emplace_back(error.message);
    }
    GenericMessageDialog::get().information(title, boost::join(errorMessages, "\n\n"));
}

void GenericMessageDialog::markdownInformation(std::string const& title, std::string const& markdownMessage)
{
    information(title, markdownMessage);
    _markdownMessage = MarkdownParser::parse(markdownMessage);
}

void GenericMessageDialog::yesNo(std::string const& title, std::string const& message, std::function<void()> const& yesFunction)
{
    _title = title;
    _message = message;
    _markdownMessage.reset();
    _dialogType = DialogType::YesNo;
    _execFunction = yesFunction;

    AlienDialog::open();
    changeTitle(title);
}

GenericMessageDialog::GenericMessageDialog()
    : AlienDialog("Message")
{}

void GenericMessageDialog::processInformation()
{
    processMessageText();
    AlienGui::Separator();

    if (AlienGui::Button("OK")) {
        close();
    }
}

void GenericMessageDialog::processYesNo()
{
    processMessageText();
    AlienGui::Separator();

    if (AlienGui::Button("Yes")) {
        close();
        _execFunction();
    }
    ImGui::SameLine();
    if (AlienGui::Button("No")) {
        close();
    }
}

void GenericMessageDialog::processMessageText()
{
    auto messageHeight = std::max(scale(20.0f), ImGui::GetContentRegionAvail().y - scale(50.0f));
    ImGui::BeginChild("MessageText", {0, messageHeight});
    if (_markdownMessage.has_value()) {
        if (auto clickedLink = _markdownRenderer.render(*_markdownMessage, {})) {
            WebLinkHelper::openInBrowser(*clickedLink);
        }
    } else {
        ImGui::TextWrapped("%s", _message.c_str());
    }
    ImGui::EndChild();
}
