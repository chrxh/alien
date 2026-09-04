#include "GettingStartedWindow.h"

#include <imgui.h>

#include "AlienGui.h"
#include "StyleRepository.h"

void GettingStartedWindow::initIntern()
{
    _showAfterStartup = _on;
}


GettingStartedWindow::GettingStartedWindow()
    : AlienWindow("Getting started", "windows.getting started", true, false, {520.0f, 176.0f}, {738.0f, 566.0f})
{}

void GettingStartedWindow::shutdownIntern()
{
    _on = _showAfterStartup;
}

void GettingStartedWindow::processIntern()
{
    drawTitle();

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)), false)) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + ImGui::GetContentRegionAvail().x);

        drawHeading1("Introduction");

        drawParagraph("ALIEN is an artificial life and physics simulation tool based on a CUDA-powered 2D particle engine for soft bodies and fluids.");
        drawParagraph(
            "Each particle can be equipped with higher-level functions including sensors, muscles, neurons, constructors, etc. that allow to "
            "mimic certain functionalities of biological cells or of robotic components. Multi-cellular organisms are simulated as networks of "
            "particles that exchange energy and information over their bonds. The engine encompasses a genetic system capable of encoding the "
            "blueprints of organisms in genomes which are stored in individual cells. The simulator is capable to simulate entire ecosystems inhabited "
            "by different populations where every object is composed of interacting particles with specific functions (regardless of whether it models a "
            "plant, herbivore, carnivore, virus, environmental structure, etc.).");

        drawHeading1("Documentation");

        drawParagraph("A new version of ALIEN is currently under development. Since the program has changed considerably in the meantime, the former getting "
                      "started guide has been removed. An updated documentation will follow.");

        drawHeading1("Examples");

        drawParagraph("Example simulations can be found in the browser window, from where they can be downloaded and opened directly.");

        ImGui::Dummy(ImVec2(0.0f, scale(20.0f)));

        ImGui::PopTextWrapPos();
    }
    ImGui::EndChild();

    AlienGui::Separator();
    AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Show after startup"), _showAfterStartup);
}

void GettingStartedWindow::drawTitle()
{
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::HeadlineColor);

    ImGui::PushFont(StyleRepository::get().getMediumFont());
    ImGui::Text("What is ");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienGui::MoveTickLeft();
    ImGui::PushFont(StyleRepository::get().getMediumBoldFont());
    ImGui::Text("A");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienGui::MoveTickLeft();
    AlienGui::MoveTickLeft();
    ImGui::PushFont(StyleRepository::get().getMediumFont());
    ImGui::Text("rtificial ");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienGui::MoveTickLeft();
    ImGui::PushFont(StyleRepository::get().getMediumBoldFont());
    ImGui::Text("LI");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienGui::MoveTickLeft();
    AlienGui::MoveTickLeft();
    ImGui::PushFont(StyleRepository::get().getMediumFont());
    ImGui::Text("fe ");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienGui::MoveTickLeft();
    ImGui::PushFont(StyleRepository::get().getMediumBoldFont());
    ImGui::Text("EN");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienGui::MoveTickLeft();
    AlienGui::MoveTickLeft();
    ImGui::PushFont(StyleRepository::get().getMediumFont());
    ImGui::Text("vironment ?");
    ImGui::PopFont();

    ImGui::PopStyleColor();
    AlienGui::Separator();
}

void GettingStartedWindow::drawHeading1(std::string const& text)
{
    AlienGui::Separator();
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::HeadlineColor);
    AlienGui::Text(AlienGui::TextParameters().text(text).style(AlienGui::TextStyle::Bold));
    ImGui::PopStyleColor();
    AlienGui::Separator();
}

void GettingStartedWindow::drawParagraph(std::string const& text)
{
    AlienGui::Text(text);
}
