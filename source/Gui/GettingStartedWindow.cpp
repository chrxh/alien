#include "GettingStartedWindow.h"

#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include "GlobalSettings.h"
#include "StyleRepository.h"
#include "AlienImGui.h"

#ifdef _WIN32
#include <windows.h>
#endif

_GettingStartedWindow::_GettingStartedWindow()
    : _AlienWindow("Getting started", "windows.getting started", true)
{
    _showAfterStartup = _on;
}


_GettingStartedWindow::~_GettingStartedWindow()
{
    _on = _showAfterStartup;
}

void _GettingStartedWindow::processIntern()
{
    ImGui::PushFont(StyleRepository::getInstance().getMediumFont());
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::HeadlineColor);
    ImGui::Text("What is (A)rtificial (LI)fe (EN)vironment?");
    ImGui::PopStyleColor();
    ImGui::PopFont();
    AlienImGui::Separator();

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - 50), false)) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + ImGui::GetContentRegionAvail().x);

        headline("Introduction");

        ImGui::Text("ALIEN is an artificial life and physics simulation tool based on a 2D CUDA-powered particle engine for soft bodies and fluids.");
        ImGui::Text("Each particle can be equipped with higher-level functions including sensors, muscles, neurons, constructors, etc. that allow to "
                    "mimic certain functionalities of biological cells or of robotic components. Multi-cellular organisms are simulated as networks of "
                    "particles that exchange energy and information over their bonds. The engine encompasses a genetic system capable of encoding the "
                    "blueprints of organisms in genomes which are stored in individual cells. The simulator is capable to simulate entire ecosystems inhabited "
                    "by different populations where every object is composed of interacting particles with specific functions (regardless of whether it models a "
                    "plant, herbivore, carnivore, virus, environmental structure, etc.).");

        headline("First steps");

        ImGui::Text("The easiest way to get to know the ALIEN simulator is to download and run an existing simulation file. You can then try out different "
                    "function and modify the simulation according to your wishes.");
        ImGui::Text("Various examples can be found in the in-game simulation browser demonstrating capabilities of the "
                    "engine ranging from pure physics examples, self-deploying structures, self-replicators to evolving ecosystems. If not already open, please "
                    "invoke Network " ICON_FA_ARROW_RIGHT " Browser in the menu bar. "
                    "Simulations can be conveniently downloaded and uploaded from/to the connected server (alien-project.org by default). "
                    "In order to upload own simulations to the server or rate other simulations, you need to register a new user, which can be accomplished in "
                    "the login dialog.");
            
        ImGui::Text("For the beginning, however, you can use the evolution example already loaded. Initially, it is advisable to become acquainted with the "
                    "windows for temporal and spatial controls. The handling should be intuitive and requires no deeper knowledge.");
        itemText("In the temporal control window, a simulation can be started or paused. The execution speed "
                    "may be regulated if necessary. In addition, it is possible to calculate and revert single time steps as "
                    "well as to make snapshots of a simulation to which one can return at any time without having "
                    "to reload the simulation from a file.");
        itemText("The spatial control window combines zoom information and settings on the one hand, and "
                    "scaling functions on the other hand. A quite useful feature in the dialog for "
                    "scaling/resizing is the option 'Scale content'. If activated, periodic spatial copies of "
                    "the original world can be made.");
        ImGui::Text("There are basically two modes of how the user can operate in the view where the simulation is "
                    "shown: a navigation mode and an edit mode. You can switch between these two modes by invoking "
                    "the edit button at the bottom left of the screen or in the menu via Editor " ICON_FA_ARROW_RIGHT " Activate.");
        itemText("The navigation mode is enabled by default and allows you to zoom in (holding the left mouse "
                    "button) and out (holding the right mouse button) continuously. By holding the middle mouse "
                    "button and moving the mouse, you can pan the visualized section of the world.");
        itemText("In the edit mode, it is possible to apply forces to bodies in a running simulation by holding and moving the right mouse button."
                    "With the left mouse button you can drag and drop objects. Please try this out. It can make a lot of fun! The editing mode also allows you "
                    "to activate various editing windows (Pattern editor, Creator, Multiplier, etc.) whose possibilities can be explored over time.");

        ImGui::Text("To be able to experiment with existing simulation files, it is important to know and change the "
                    "simulation parameters. This can be accomplished in the window 'Simulation parameters'. For example, "
                    "the radiation intensity can be increased or the friction can be adjusted. Explanations to the "
                    "individual parameters can be found in the tooltip next to them.");

        ImGui::Text("ALIEN offers the possibility for users to customize the basic entities through a color system with 7 different colors. More precisely, each "
                    "cell is assigned a specific color, allowing the application of different simulation parameter values based on the cell's color. This "
                    "enables the creation of specific conditions for populations coexisting in a shared world. For example, "
                    "plant-like organisms may have a higher absorption rate for radiation particles, so they can get their energy from that.");

        AlienImGui::Separator();

        ImGui::Text(
            "IMPORTANT: On older graphics cards or when using a high resolution (e.g. 4K), it is recommended to reduce the rendered frames per second, "
            "as this significantly increases the simulation speed (time steps per second). This adjustment can be made in the display settings.");

        headline("Basic notion");

        ImGui::Text("Generally, in an ALIEN simulation, all objects as well as thermal radiation are modeled by different types of particles. The following "
                    "terms are frequently used:");

        ImGui::Spacing();
        AlienImGui::BoldText("World");
        ImGui::Text("An ALIEN world is two-dimensional rectangular domain with periodic boundary conditions. The space is modeled as a continuum.");

        ImGui::Spacing();
        AlienImGui::BoldText("Cell");
        ImGui::Text(
            "Cells are the basic building blocks that make up everything. They can be connected to each others, possibly attached to the background "
            "(to model barriers), possess special functions and transport neural activities. Additionally, cells have various physical properties, including");
        itemText("Position in space");
        itemText("Velocity");
        itemText("Internal energy (corresponding to its temperature)");
        itemText("Upper limit of connections");

        //ImGui::Text("There is a lot to explore. ALIEN features an extensive graph and particle editor in order to build custom worlds with desired "
        //            "environmental structures and machines. A documentation with tutorial-like introductions to various topics can be found at");

        //ImGui::Dummy(ImVec2(0.0f, 20.0f));

        //ImGui::PushFont(StyleRepository::getInstance().getMonospaceMediumFont());
        //auto windowWidth = ImGui::GetWindowSize().x;
        //auto weblink = "https://alien-project.gitbook.io/docs";
        //auto textWidth = ImGui::CalcTextSize(weblink).x;
        //ImGui::SetCursorPosX((windowWidth - textWidth) * 0.5f);
        //if(AlienImGui::Button(weblink)) {
        //    openWeblink(weblink);
        //}
        //ImGui::PopFont();

        ImGui::Dummy(ImVec2(0.0f, 20.0f));

        ImGui::PopTextWrapPos();
    }
    ImGui::EndChild();

    AlienImGui::Separator();
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Show after startup"), _showAfterStartup);
}

void _GettingStartedWindow::headline(std::string const& text)
{
    AlienImGui::Separator();
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::HeadlineColor);
    AlienImGui::BoldText(text);
    ImGui::PopStyleColor();
    AlienImGui::Separator();
}

void _GettingStartedWindow::itemText(std::string const& text)
{
    ImGui::Text(ICON_FA_CARET_RIGHT);
    ImGui::SameLine();
    AlienImGui::Text(text);
}

void _GettingStartedWindow::openWeblink(std::string const& link)
{
#ifdef _WIN32
    ShellExecute(NULL, "open", link.c_str(), NULL, NULL, SW_SHOWNORMAL);
#endif
}
 