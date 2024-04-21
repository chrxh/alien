#include "GettingStartedWindow.h"

#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

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

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)), false)) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + ImGui::GetContentRegionAvail().x);

        /**
         * INTRO
         */
        drawHeading1("Introduction");

        ImGui::Text("ALIEN is an artificial life and physics simulation tool based on a CUDA-powered 2D particle engine for soft bodies and fluids.");
        ImGui::Text(
            "Each particle can be equipped with higher-level functions including sensors, muscles, neurons, constructors, etc. that allow to "
            "mimic certain functionalities of biological cells or of robotic components. Multi-cellular organisms are simulated as networks of "
            "particles that exchange energy and information over their bonds. The engine encompasses a genetic system capable of encoding the "
            "blueprints of organisms in genomes which are stored in individual cells. The simulator is capable to simulate entire ecosystems inhabited "
            "by different populations where every object is composed of interacting particles with specific functions (regardless of whether it models a "
            "plant, herbivore, carnivore, virus, environmental structure, etc.).");

        /**
         * FIRST STEPS
         */
        drawHeading1("First steps");

        ImGui::Text("The easiest way to get to know the ALIEN simulator is to download and run an existing simulation file. You can then try out different "
                    "function and modify the simulation according to your wishes.");
        ImGui::Text(
            "Various examples can be found in the in-game simulation browser demonstrating capabilities of the "
            "engine ranging from pure physics examples, self-deploying structures, self-replicators to evolving ecosystems. If not already open, please "
            "invoke Network " ICON_FA_LONG_ARROW_ALT_RIGHT " Browser in the menu bar. "
            "Simulations can be conveniently downloaded and uploaded from/to the connected server (alien-project.org by default). "
            "In order to upload own simulations to the server or rate other simulations, you need to register a new user, which can be accomplished in "
            "the login dialog.");
            
        ImGui::Text("For the beginning, however, you can use the example already loaded. Initially, it is advisable to become acquainted with the "
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
        drawItemText(
            "The navigation mode is enabled by default and allows you to zoom in (holding the left mouse "
            "button) and out (holding the right mouse button) continuously. Alternatively, you can also use the mouse wheel. By holding the middle mouse "
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

        drawHeading2("Important");
        ImGui::Text(
            "On older graphics cards or when using a high resolution (e.g. 4K), it is recommended to reduce the rendered frames per second, "
            "as this significantly increases the simulation speed (time steps per second). This adjustment can be made in the display settings.");

        /**
         * EXAMPLES
         */
        drawHeading1("Examples");
        drawParagraph(
            "ALIEN comes with a lot of simulation files that can be found in the browser window. They are good for experimenting with certain aspects of the "
            "program. We pick some examples to give a short overview:");

        drawHeading2("Fluids, walls and soft bodies");
        drawParagraph("There are several pure physics simulations demonstrating the engines' capability. They are suitable for testing the influence of "
                      "simulation parameters such as 'Smoothing length', 'Pressure', 'Viscosity', etc.");
        drawItemText("Fluids/Pump with Soft-Bodies");
        drawItemText("Demos/Perpetual Motion Machine");
        drawItemText("Demos/Stormy Night");

        drawHeading2("Evolution simulations");
        drawParagraph("By attaching higher-level functions to particle networks, complex multi-cellular organisms can be modeled. They can evolve over time as "
                      "they are subject to mutations. The following examples consist of homogeneous as well as changing worlds populated by self-reproducing agents. Different "
                      "selection pressures control evolution.");
        drawItemText("Primordial World/Selected Results");
        drawItemText("v4.8-Evolution/Gradient/Selected Results");
        drawItemText("Evolution Sandbox/Example");
        drawItemText("Complex Evolution Testbed/Example");
        drawItemText("Diversity/Example");
        drawItemText("Color Niches/Example");

        drawHeading2("Plant-herbivore ecosystems");
        drawParagraph("By customizing the cells according to their color, it is possible to specify different types of organisms. There are many examples that "
                      "feature two classes: plants and herbivores. Plants are able to consume radiation particles, while herbivores can consume plants. This "
                      "simple relationship already provides interesting dynamics, as the following examples show.");
        drawItemText("Twin Worlds/Example");
        drawItemText("Bugs and Flowers/Example");
        drawItemText("Self-replicating Fluid/Initial Setting");

        drawHeading2("Swarming");
        drawParagraph("There are powerful sensors available as cell functions for detecting concentrations of specific colors in the surroundings. "
                      "Organisms equipped with these sensors can perceive their environment, nourish their neural networks, and respond accordingly.");
        drawItemText("Swarms/Space Invaders");
        drawItemText("Evolving Swarms/Example");

        /**
         * BASIC NOTION
         */
        drawHeading1("Basic notion");

        ImGui::Text("Generally, in an ALIEN simulation, all objects as well as thermal radiation are modeled by different types of particles moving through an "
                    "empty space. The following terms are frequently used:");

        ImGui::Spacing();
        drawHeading2("World");
        ImGui::Text("An ALIEN world is two-dimensional rectangular domain with periodic boundary conditions. The space is modeled as a continuum.");

        ImGui::Spacing();
        drawHeading2("Cell");
        ImGui::Text(
            "Cells are the basic building blocks that make up everything. They can be connected to each others, possibly attached to the background "
            "(to model barriers), possess special functions and transport activity values. Additionally, cells have various physical properties, including");
        drawItemText("Position in space");
        drawItemText("Velocity");
        drawItemText("Internal energy (may be interpreted as its temperature)");
        drawItemText("Upper limit of connections");
        drawItemText("Living state");

        ImGui::Spacing();
        drawHeading2("Cell connection");
        ImGui::Text(
            "A cell connection is a bond between two cells. It stores the reference distance and on each side a reference angle to a possibly further cell "
            "connection. The reference distance and angles are calculated when the connection is established. As soon as the actual distance deviates from "
            "the reference distance, a pulling/pushing force is applied at both ends. Furthermore, tangential forces are applied at both ends in the "
            "case of an angle mismatch.");

        ImGui::Spacing();
        drawHeading2("Cell function");
        ImGui::Text("It is possible to assign a special function to a cell, which will be executed at regular time intervals. The following functions are "
                    "implemented:");
        drawItemText("Neuron: It equips the cell with a small network of 8 neurons. It processes input gained from the activity states of connected cells and provides an "
                     "output to other connected cells.");
        drawItemText(
            "Transmitter: It distributes energy to other constructors, transmitters or surrounding cells. In particular, it can be used to power active "
            "constructors. No activity is required for triggering.");
        drawItemText("Constructor: A constructor can build a cell network based on a built-in genome. The construction is done cell by cell and requires "
                     "energy. A constructor can either be controlled via activities or become active automatically (default).");
        drawItemText("Injector: It can infect other constructor cells to inject its own built-in genome.");
        drawItemText("Nerve: On the one hand, it transfers activity values from connected input cells and on the other hand, it can optionally generate "
                     "activity pulses at specific intervals.");
        drawItemText("Attacker: It attacks surrounding cells from other cell networks by stealing energy from them.");
        drawItemText("Defender: It reduces the attack strength when another cell in the vicinity performs an attack.");
        drawItemText("Muscle: When a muscle cell is activated, it can produce either a movement, a bending or a change in length of the cell connection.");
        drawItemText("Sensor: If activated, it performs a long-range scan for the concentration of cells with a certain color.");
        drawItemText("Reconnector: Has the ability to dynamically create or destroy connections to other cells with a specified color.");
        drawItemText(
            "Detonator: A cell which can explode by an activity signal. It generates a large amount of kinetic energy for the objects in its surroundings.");

        ImGui::Spacing();
        drawHeading2("Activity states");
        drawParagraph(
            "Cells contain activity states comprising of 8 values, primarily utilized for controlling cell functions and sometimes referred to as channel #0 "
            "to channel #7. The states are refreshed periodically, specifically when the cell functions are executed. To be more precise, each cell function "
            "is executed at regular time intervals (every 6 time steps). The 'execution order number' specifies the exact time offset within those intervals.");
        drawParagraph("The process for updating the activity states is as follows: Firstly, the states of all connected cells that serve as input (i.e. "
                    "the connected cells which matches with the input execution number) are summed up. The resulted sum is then employed as input for "
                    "the cell function, which may potentially alter the values. Subsequently, the outcome is used to update the activity states.");

        ImGui::Spacing();
        drawHeading2("Cell color");
        drawParagraph("In addition to cell functions, a color can be used to perform additional user-defined customization of cells. For this purpose, most "
                    "simulation parameters can be adjusted separately for each color, if desired. As a result, cells of different colors may have individual "
                    "properties.");

        ImGui::Spacing();
        drawHeading2("Cell network");
        drawParagraph("A cell network is a connected graph consisting of cells and cell connections. Two cells in a network are therefore "
                    "connected to each other directly or via other cells. A cell network physically represents a particular body.");

        ImGui::Spacing();
        drawHeading2("Genome");
        drawParagraph("The blueprints for entire cell networks can be stored in genomes. These genomes are translated into real cell networks by constructor "
              "cells and, if necessary, copied to their offspring. Furthermore, injector cells are able to inject their own genome into other "
              "constructor cells, which allows to model virus behaviors.");
        drawParagraph("A genome in ALIEN may describe several cell networks, which are hierarchically structured und possibly connected when constructed. More "
                      "precisely, it means that there is a top-level blueprint describing a certain network. If there are further constructor cells "
                      "within this network, they can also contain further genomes, which in turn describe other cell networks and so on.");

        ImGui::Spacing();
        drawHeading2("Energy particle");
        drawParagraph(
            "An energy particle is a particle which has only an energy value, position and velocity. Unlike cells, they cannot form networks or perform any "
            "additional functions. Energy particles are produced by cells as radiation or during decay and can, in turn, also be absorbed.");

        ImGui::Spacing();
        drawHeading2("Pattern");
        drawParagraph("A pattern is a set of cell networks and energy particles.");

        /**
         * SIMULATION PARAMETERS
         */
        drawHeading1("Simulation parameters");
        drawParagraph(
            "All parameters relevant to the simulation can be adjusted here. By default, the parameters are set uniformly for the entire world. However, it is "
            "also possible to allow certain parameters to vary locally in special zones. To do this, you can create a new tab in the simulation parameter window by clicking on "
            "the '+' button. It adds a spatially (fuzzy) delimited area where the global parameters can be overwritten. This zone is also visible by a "
            "different color.");
        drawParagraph("Regardless of this, many parameters can also be set depending on the cell color. For this purpose click the '+' button beside the "
                      "parameter. This customization is useful when you want to define different classes of species.");
        drawParagraph("Sometimes it is difficult to set precise values in the slider. In this case, you can select the slider and then press the TAB key. This "
                      "allows you to enter the exact value in an input field and confirm it by pressing ENTER.");
        drawParagraph("In general, the following types of parameters can be set.");
        drawHeading2("Rendering");
        drawParagraph(
            "In addition to the background color, you can determine the coloring of the cells here. Each cell is assigned a specific color, which can be used "
            "for customization and which is also used by default for rendering. However, in evolution simulations, it can be very useful to color mutants "
            "differently. This allows for better visual evaluation of diversities, mutation rates, and successful mutants, etc. For this purpose, you can "
            "switch the cell coloring to the mutation id.");
        drawHeading2("Physics");
        drawParagraph("Basic physical properties can be modified in these settings. This includes adjusting the radiation intensity, various thresholds, and "
                     "the motion algorithm. Changes can have significant effects on performance and, in the worst case, may lead to program crashes.");
        drawHeading2("Radiation sources");
        drawParagraph("Optionally, you can define radiation sources by opening the corresponding editor. Typically, all cells lose energy over time by "
                      "emitting particles. These energy particles travel through space and can be absorbed by other cells under certain conditions. When no "
                      "radiation source is defined, energy particles are emitted at the cell's position, resulting in a more or less uniform distribution of "
                      "energy particles throughout space over time. For certain simulations, especially in modeling plant species, it is beneficial to specify "
                      "explicit sources where energy particles should be generated. This can be achieved in the 'Radiation sources' window. Even when a source "
                      "is defined, cells continue to lose the same amount of energy as before. The difference is that particles are now spawned at the "
                      "specified source. The energy conservation principle remains intact.");
        drawHeading2("Cell specific parameters");
        drawParagraph("These parameter types are particularly important when simulating (self-replicating) agents composed of cell networks, going beyond pure "
                      "physical simulations. Many of the different cell functions depend on specific parameters, which can be adjusted here. Particularly "
                      "important are the parameters for mutation rates and the attack functions.With the latter, the food chain between cells of different "
                      "colors can be configured. For example, in the 'Food chain color matrix' one could specify that cells with a certain color can only "
                      "consume cells with a certain other color but not themselves.");
        drawParagraph("The mutation rates influence the probability of modifying a genome for the underlying cells. When adjusting these rates, it should "
                      "be noted that different types of mutations also have different impacts. For instance, a 'Duplication' mutation affects the genome much "
                      "more invasively than a 'Neural net' mutation, which only adjusts weights and biases. Furthermore, it should be considered that for "
                      "evolutionary simulations, where individuals require a long time for self-replication, high mutation rates should be "
                      "avoided. The correct values are best determined through experimentation.");

        /**
         * EDITORS
         */
        drawHeading1("Editors");
        drawParagraph(
            "If you want to design your own worlds, sceneries or organisms, there are many different editors available, which partially require deeper "
            "knowledge. To open the editors, you have to switch to the edit mode (e.g. a click on the icon at the bottom left). A short overview of the "
            "possibilities are given below.");
        drawHeading2("Drag and drop");
        drawParagraph(
            "The easiest way is to select and move objects with the mouse. You can simply drag and drop cell networks in the simulation view. This "
            "also works during running simulations. When the simulation is paused, you can select a rectangular area to be highlighted by holding "
            "down the right mouse button. The selection is visually highlighted and can be moved via drag and drop. By holding down the SHIFT button during"
            "a mouse action, only the selected cells and not the associated cell networks are shifted. This can lead to the destruction or formation of cell "
            "connections which are not selected.");
        drawHeading2("Creator");
        drawParagraph("If you want to create individual cells, cell networks, or energy particles, you can open the 'Creator' window. In this window you also "
                      "find a mode for creating cell structures by freehand drawing on the simulation area. The created cells are equipped with default values "
                      "and can be modified later if desired.");
        drawHeading2("Pattern editor");
        drawParagraph("Physical properties of already selected cells and energy particles, such as center velocities, positions, colors, etc., can be "
                      "conveniently changed in the 'Pattern editor'. In addition, selections can be saved, loaded, copied and pasted.");
        drawHeading2("Genome editor");
        drawParagraph("In the 'Genome editor', genomes describing cell networks can be created and modified. Each tab shows the principal part of the genome. "
                      "It is a sequence of cells that are supposed to be constructed in that order. If one of these cells is a constructor cell, it contains "
                      "an additional genome that can be edited in a separate tab, if desired.");
        drawParagraph("To translate a genome into an actual structure, there are two options:");
        drawItemText("You can generate a spore using the corresponding toolbar button. A spore is a single constructor cell containing the specific genome and "
                     "possessing enough energy to create the main structure described within.");
        drawItemText(
            "Another option is to inject the genome into an existing organism. To do this, you must select the organism and click on 'Inspect principal "
            "genome' in the editor menu. A window will open where you see the existing genome of that creature. Then you can inject your own genome by "
            "invoking the 'Inject from editor' button.");
        drawHeading2("Object inspection");
        drawParagraph("Nearly every property of each particle can be viewed and edited. For this purpose, special editing windows can be attached to a "
                      "particle. To do this, you have to select one or more particles (not too many) and invoke 'Inspect objects' from the editor menu. Each "
                      "selected particle is now connected to a window. It is even possible to view these editing windows during a running simulation. In this "
                      "way, you can monitor in real-time how properties of individual particles change over time.");
        drawHeading2("Mass operations");
        drawParagraph("Various mass editing functions are available. On the one hand, the pattern editor allows to change different physical properties of an "
                      "entire selection. On the other hand, through the 'Tools' menu, the 'Mass operations' dialog can be opened. Here, colors of cells and "
                      "genomes, energy values, and other properties can be globally modified. The new values will be randomly chosen from a specified range. "
                      "Cells within a cell networks will be assigned the same value.");
        drawParagraph("In addition, simulations can be scaled up by increasing the size of the world and filling the resulting empty space with copies of the "
                      "original world. This functionality is available via the resize dialog in the 'Spatial control' window, if you set a new size there and "
                      "activate 'Scale content'.");
        /**
         * FREQUENTLY ASK QUESTIONS
         */
        drawHeading1("Frequently asked questions");

        drawHeading2("How does a simple self-replicating organism work?");
        drawParagraph("In general, an organism in ALIEN consists of a network of cells where the cells work together by communicating with each other through "
                      "activity signals.");
        drawParagraph(
            "A simple creature first needs a constructor cell that contains its genome and is responsible for self-replication. The constructor cell "
            "is automatically triggered and generates (as soon as enough energy is available) the cell network of the offspring built cell by cell as described in "
            "its genome. The genome for the offspring is also copied and placed into the constructor cell of the offspring.");
        drawParagraph(
            "Self-replication requires energy, which must be obtained in some way. On the one hand, energy can be acquired by the absorption of "
            "energy particles flying around. This can be the main source of energy for plant-like species. On the other hand, there is the possibility to utilize "
            "attacker cells. They can attack cells from other organisms by stealing energy from them. If an attacker cell is part of the creature, it must be "
            "explicitly triggered by an activity signal. This signal may come, for example, from another cell equipped with a neural network. The energy "
            "obtained by an attacker cell is distributed to nearby constructor or transmitter cells.");
        drawParagraph("To perform movements, an organism requires muscle cells. These are also controlled by activity signals. Muscle cells can work in "
                      "various modes: they can bend, contract/expand, or generate an impulse.");
        drawParagraph("For the perception of the environment, sensor cells are available. When such a cell is triggered by activity signals, it provide "
                      "information about the relative position of cell concentrations with respect to a specific color, which can be further processed by e.g. "
                      "cell with neural network.");

        drawHeading2("Why does the radiation source generates no energy particles?");
        drawParagraph("The principle of energy conservation holds true in a simulation, which means that energy particles cannot spontaneously come into "
                      "existence out of nothingness. This principle plays a vital role in ensuring the stability of long-term simulations. If a radiation "
                      "source is defined, it will emit a particle only when a cell somewhere loses energy. Conversely, in the absence of a radiation source, "
                      "any emitted energy particle would originate directly in the spatial vicinity of the corresponding cell.");
        drawParagraph("The existence of matter in the form of cells is a prerequisite for the radiation source to emit particles. Furthermore, the simulation "
                      "parameters should be adjusted in a way that guarantees a gradual loss of energy from cells over time.");

        drawHeading2("How can neural networks be incorporated?");
        drawParagraph("Neural networks are available as cell functions. If a cell is assigned the function 'Neuron', it possesses a small neural network "
                      "consisting of 8 neurons. However, these networks can be interconnected to form arbitrarily large networks, as each cell receives input "
                      "from the output of certain connected cells.");
        drawParagraph("Furthermore, these networks can be fed by sensors and, in turn, control muscle and attacker cells.");

        drawHeading2("For how long should I run a simulation to see evolutionary changes?");
        drawParagraph(
            "This depends on many factors: On the size of the simulated world, on the mutation rates, on various selection pressures that can be influenced "
            "by the simulation parameters and on the self-replication duration. Usually one should wait for several dozen generations, which may correspond to hundreds of thousands or million time steps."
            "In small worlds with smaller organisms and high mutation rates, evolutionary changes can sometimes be observed every minute depending on the hardware. With more complex "
            "simulations, you should rather expect a few hours.");

        drawHeading2("How can I create a cell activity in the first place?");
        drawParagraph("To activate a cell function, an input from a connected cell in the form of non-zero activity states is required. The simplest methods "
                      "to generate an non-zero activity state are as follows:");
        drawItemText("The most direct approach involves using a nerve cell that generates an activity pulse at regular time intervals. The advantage here is that "
            "you can precisely configure the length of the time intervals.");
        drawItemText("Activities can also be generated within a neuron cell using bias values.");
        drawParagraph("Additionally, other cells such as constructor cells provide an activity output as soon as they are triggered (automatically).");

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

        ImGui::Dummy(ImVec2(0.0f, scale(20.0f)));

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

void _GettingStartedWindow::drawHeading1(std::string const& text)
{
    AlienImGui::Separator();
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::HeadlineColor);
    AlienImGui::BoldText(text);
    ImGui::PopStyleColor();
    AlienImGui::Separator();
}

void _GettingStartedWindow::drawHeading2(std::string const& text)
{
    ImGui::Spacing();
    AlienImGui::BoldText(text);
}

void _GettingStartedWindow::drawItemText(std::string const& text)
{
    ImGui::Text(ICON_FA_CHEVRON_RIGHT);
    ImGui::SameLine();
    AlienImGui::Text(text);
}

void _GettingStartedWindow::drawParagraph(std::string const& text)
{
    AlienImGui::Text(text);
}
 