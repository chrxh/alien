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
    drawTitle();

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - 50), false)) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + ImGui::GetContentRegionAvail().x);

        drawHeadline("Introduction");

        ImGui::Text("ALIEN is an artificial life and physics simulation tool based on a 2D CUDA-powered particle engine for soft bodies and fluids.");
        ImGui::Text(
            "Each particle can be equipped with higher-level functions including sensors, muscles, neurons, constructors, etc. that allow to "
            "mimic certain functionalities of biological cells or of robotic components. Multi-cellular organisms are simulated as networks of "
            "particles that exchange energy and information over their bonds. The engine encompasses a genetic system capable of encoding the "
            "blueprints of organisms in genomes which are stored in individual cells. The simulator is capable to simulate entire ecosystems inhabited "
            "by different populations where every object is composed of interacting particles with specific functions (regardless of whether it models a "
            "plant, herbivore, carnivore, virus, environmental structure, etc.).");

        drawHeadline("First steps");

        ImGui::Text("The easiest way to get to know the ALIEN simulator is to download and run an existing simulation file. You can then try out different "
                    "function and modify the simulation according to your wishes.");
        ImGui::Text(
            "Various examples can be found in the in-game simulation browser demonstrating capabilities of the "
            "engine ranging from pure physics examples, self-deploying structures, self-replicators to evolving ecosystems. If not already open, please "
            "invoke Network " ICON_FA_LONG_ARROW_ALT_RIGHT " Browser in the menu bar. "
            "Simulations can be conveniently downloaded and uploaded from/to the connected server (alien-project.org by default). "
            "In order to upload own simulations to the server or rate other simulations, you need to register a new user, which can be accomplished in "
            "the login dialog.");
            
        ImGui::Text("For the beginning, however, you can use the evolution example already loaded. Initially, it is advisable to become acquainted with the "
                    "windows for temporal and spatial controls. The handling should be intuitive and requires no deeper knowledge.");
        drawItemText("In the temporal control window, a simulation can be started or paused. The execution speed "
                     "may be regulated if necessary. In addition, it is possible to calculate and revert single time steps as "
                     "well as to make snapshots of a simulation to which one can return at any time without having "
                     "to reload the simulation from a file.");
        drawItemText("The spatial control window combines zoom information and settings on the one hand, and "
                     "scaling functions on the other hand. A quite useful feature in the dialog for "
                     "scaling/resizing is the option 'Scale content'. If activated, periodic spatial copies of "
                     "the original world can be made.");
        ImGui::Text("There are basically two modes of how the user can operate in the view where the simulation is "
                    "shown: a navigation mode and an edit mode. You can switch between these two modes by invoking "
                    "the edit button at the bottom left of the screen or in the menu via Editor " ICON_FA_LONG_ARROW_ALT_RIGHT " Activate.");
        drawItemText("The navigation mode is enabled by default and allows you to zoom in (holding the left mouse "
                     "button) and out (holding the right mouse button) continuously. By holding the middle mouse "
                     "button and moving the mouse, you can pan the visualized section of the world.");
        drawItemText(
            "In the edit mode, it is possible to push bodies around in a running simulation by holding and moving the right mouse button. "
            "With the left mouse button you can drag and drop objects. Please try this out. It can make a lot of fun! The editing mode also allows you "
            "to activate lot of editing windows (Pattern editor, Creator, Multiplier, Genome editor, etc.) whose possibilities can be explored over time. "
            "Practically all properties of each single particle can be manipulated. In addition, there are mass editing functions available.");

        ImGui::Text("To be able to experiment with existing simulations, it is important to know and change the "
                    "simulation parameters. This can be accomplished in the window 'Simulation parameters'. For example, "
                    "the radiation intensity can be increased or the friction can be adjusted. Explanations to the "
                    "individual parameters can be found in the tooltip next to them.");

        ImGui::Text(
            "ALIEN offers the possibility for users to customize the basic entities through a color system with 7 different colors. More precisely, each "
            "cell is assigned a specific color, allowing the application of different simulation parameter values based on the cell's color. This "
            "enables the creation of specific conditions for populations coexisting in a shared world. For example, "
            "plant-like organisms may have a higher absorption rate for radiation particles, so they can get their energy from that.");

        ImGui::Spacing();

        AlienImGui::BoldText("Important");
        ImGui::Text(
            "On older graphics cards or when using a high resolution (e.g. 4K), it is recommended to reduce the rendered frames per second, "
            "as this significantly increases the simulation speed (time steps per second). This adjustment can be made in the display settings.");

        drawHeadline("Basic notion");

        ImGui::Text("Generally, in an ALIEN simulation, all objects as well as thermal radiation are modeled by different types of particles moving through an "
                    "empty space. The following terms are frequently used:");

        ImGui::Spacing();
        AlienImGui::BoldText("World");
        ImGui::Text("An ALIEN world is two-dimensional rectangular domain with periodic boundary conditions. The space is modeled as a continuum.");

        ImGui::Spacing();
        AlienImGui::BoldText("Cell");
        ImGui::Text(
            "Cells are the basic building blocks that make up everything. They can be connected to each others, possibly attached to the background "
            "(to model barriers), possess special functions and transport activity values. Additionally, cells have various physical properties, including");
        drawItemText("Position in space");
        drawItemText("Velocity");
        drawItemText("Internal energy (may be interpreted as its temperature)");
        drawItemText("Upper limit of connections");

        ImGui::Spacing();
        AlienImGui::BoldText("Cell connection");
        ImGui::Text(
            "A cell connection is a bond between two cells. It stores the reference distance and on each side a reference angle to a possibly further cell "
            "connection. The reference distance and angles are calculated when the connection is established. As soon as the actual distance deviates from "
            "the reference distance, a pulling/pushing force is applied at both ends. Furthermore, tangential forces are applied at both ends in the "
            "case of an angle mismatch.");

        ImGui::Spacing();
        AlienImGui::BoldText("Cell activity");
        ImGui::Text("Cells can contain an activity state comprising of 8 real values, primarily utilized for controlling cell functions. The activities are "
                    "refreshed periodically, specifically when the cell functions are executed. To be more precise, each cell function is executed at regular "
                    "time intervals (every 6 time steps). The 'execution order number' specifies the exact time within each interval.");
        ImGui::Text("The process for updating the cell activity is as follows: Firstly, the activities of all connected cells that serve as input are summed "
                    "up. The resulted sum is then employed as input for the cell function, which may potentially alter the activity values. Subsequently, the "
                    "outcome is utilized to determine the new cell activity.");

        ImGui::Spacing();
        AlienImGui::BoldText("Cell function");
        ImGui::Text("It is possible to assign a special function to a cell, which will be executed at regular time intervals. The following functions are "
                    "implemented:");
        drawItemText("Neuron: It equips the cell with a small network of 8 neurons. It processes the activity values fetched from the input.");
        drawItemText(
            "Transmitter: It distributes energy to other constructors, transmitters or surrounding cells. In particular, it can be used to power active "
            "constructors. No activity is required for triggering.");
        drawItemText("Constructor: A constructor can build a cell cluster based on a built-in genome. The construction is done cell by cell and requires "
                     "energy. A constructor can either be controlled via activities or become active automatically (default).");
        drawItemText("Injector: It can infect other constructor cells to inject its own built-in genome.");
        drawItemText("Nerve: On the one hand, it transfers activity values from connected input cells and on the other hand, it can optionally generate "
                     "activity pulses at specific intervals.");
        drawItemText("Attacker: If activated, it attacks (not connected) surrounding cells.");
        drawItemText("Defender: It reduces the attack strength when another cell in the vicinity performs an attack.");
        drawItemText("Muscle: When a muscle cell is activated, it can produce either a movement, a bending or a change in length of the cell connection.");
        drawItemText("Sensor: If activated, it performs a long-range scan for the concentration of cells with a certain color.");

        ImGui::Spacing();
        AlienImGui::BoldText("Cell color");
        ImGui::Text("In addition to cell functions, a color can be used to perform additional user-defined customization of cells. For this purpose, most "
                    "simulation parameters can be adjusted separately for each color, if desired. As a result, cells of different colors may have individual "
                    "properties.");

        ImGui::Spacing();
        AlienImGui::BoldText("Cell cluster");
        ImGui::Text("A cell cluster (or cluster for short) is a connected graph consisting of cells and cell connections. Two cells in a cluster are therefore "
                    "connected to each other directly or via other cells. A cluster physically represents a particular body.");

        ImGui::Spacing();
        AlienImGui::BoldText("Energy particle");
        ImGui::Text(
            "An energy particle is a particle which has only an energy value, position and velocity. Unlike cells, they cannot form clusters or perform any "
            "additional functions. Energy particles are produced by cells as radiation or during decay and can, in turn, also be absorbed.");

        ImGui::Spacing();
        AlienImGui::BoldText("Pattern");
        ImGui::Text("A pattern is a set of cell clusters and energy particles.");

        drawHeadline("Examples");
        ImGui::Text(
            "ALIEN comes with a lot of simulation files that can be found in the browser window. They are good for experimenting with certain aspects of the "
                    "program. We pick some examples to give a short overview:");
        AlienImGui::BoldText("Fluids/Pump with Soft-Bodies");
        ImGui::Text("This is a pure physics simulation consisting of different colored fluids, walls and soft bodies. One can control the behavior with "
                    "different simulation parameters like 'Smoothing length', 'Pressure', 'Viscosity', etc.");

        
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

void _GettingStartedWindow::drawTitle()
{
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::HeadlineColor);

    ImGui::PushFont(StyleRepository::getInstance().getMediumFont());
    ImGui::Text("What is ");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienImGui::NegativeSpacing();
    ImGui::PushFont(StyleRepository::getInstance().getMediumBoldFont());
    ImGui::Text("A");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienImGui::NegativeSpacing();
    AlienImGui::NegativeSpacing();
    ImGui::PushFont(StyleRepository::getInstance().getMediumFont());
    ImGui::Text("rtificial ");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienImGui::NegativeSpacing();
    ImGui::PushFont(StyleRepository::getInstance().getMediumBoldFont());
    ImGui::Text("LI");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienImGui::NegativeSpacing();
    AlienImGui::NegativeSpacing();
    ImGui::PushFont(StyleRepository::getInstance().getMediumFont());
    ImGui::Text("fe ");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienImGui::NegativeSpacing();
    ImGui::PushFont(StyleRepository::getInstance().getMediumBoldFont());
    ImGui::Text("EN");
    ImGui::PopFont();

    ImGui::SameLine();
    AlienImGui::NegativeSpacing();
    AlienImGui::NegativeSpacing();
    ImGui::PushFont(StyleRepository::getInstance().getMediumFont());
    ImGui::Text("vironment ?");
    ImGui::PopFont();

    ImGui::PopStyleColor();
    AlienImGui::Separator();
}

void _GettingStartedWindow::drawHeadline(std::string const& text)
{
    AlienImGui::Separator();
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::HeadlineColor);
    AlienImGui::BoldText(text);
    ImGui::PopStyleColor();
    AlienImGui::Separator();
}

void _GettingStartedWindow::drawItemText(std::string const& text)
{
    ImGui::Text(ICON_FA_CHEVRON_RIGHT);
    ImGui::SameLine();
    AlienImGui::Text(text);
}

void _GettingStartedWindow::openWeblink(std::string const& link)
{
#ifdef _WIN32
    ShellExecute(NULL, "open", link.c_str(), NULL, NULL, SW_SHOWNORMAL);
#endif
}
 