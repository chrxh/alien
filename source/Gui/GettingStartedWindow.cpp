#include "GettingStartedWindow.h"

#include "imgui.h"

#include "GlobalSettings.h"
#include "StyleRepository.h"

_GettingStartedWindow::_GettingStartedWindow(StyleRepository const& styleRepository)
    : _styleRepository(styleRepository)
{
    _showAfterStartup = GlobalSettings::getInstance().getBoolState("windows.getting started.always active", true);
    _on = _showAfterStartup;
}

_GettingStartedWindow::~_GettingStartedWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.getting started.always active", _showAfterStartup);
}

void _GettingStartedWindow::process()
{
    if (!_on) {
        return;
    }
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin("Getting started", &_on, ImGuiWindowFlags_None)) {
        ImGui::End();
    }
}

bool _GettingStartedWindow::isOn() const
{
    return _on;
}

void _GettingStartedWindow::setOn(bool value)
{
    _on = value;
}

/*
What is Articial Life Environment(ALiEn)
    ? ALiEn is an artificial life simulation program based on a specialized physics and rendering engine with extensive
      world building features.It is a high performance simulation tool in the sense that it not only uses OpenGL
      for rendering,
    but also employs the CUDA api to run simulations on thousands of GPU threads.Each simulated
            body(named as cell clusters) has a graph
        - like structure of connected building blocks(named as cells) that can either be programmed
    or equipped with functions to act in the world(accelerators, sensors, weapons, constructors, etc.)
           .Such internal processes are triggered by circulating tokens.The bodies can be thought of as small machines
    or agents operating in a common environment.

        First steps The easiest way to get to know the alien simulator is to load
        and run an existing simulation file.You find various demos in examples / simulations/ * demonstrating capabilities of the engine ranging from physics examples, self-deploying structures, replicators to small ecosystems. To this end, go to Simulation -> Load and select a file. However, for starters, you can use the simple evolution example already loaded. 
Once a simulation file is loaded, click Simulation -> Run on the menu or toolbar to start it. Let us give you a brief overview of some important features. For detailed explanations and tutorials please visit Help -> Documentation from the menu. A running simulation is a good opportunity to learn some basic navigation, action and monitoring functions: 
Navigation and action mode: By clicking on View -> Navigation mode and View -> Action mode (or conveniently via the toolbar) you can toggle between these two modes. The navigation mode is enabled by default and allows you to zoom in (while holding the left mouse button) and zoom out (while holding the right mouse button) continuously. By holding the middle mouse button and moving the mouse, you can move the visualized section of the world. In action mode, it is possible to apply forces to bodies in a running simulation or move them in a paused simulation. Please try this out. It can make a lot of fun! When the editor is on, advanced manipulation functions are unlocked. 
Visualization modes: There are basically two different visualization modes: A pixel/vector graphics view as well as a item-based view for precise editing and for showing all the details. By invoking View -> Pixel/vector view and View -> Item-based view (again it is recommended to use the toolbar) you can toggle between them. After reaching a certain zoom level, the view switches automatically to the appropriate view mode. 
Toggling info bar: Is enabled by default and can be disabled via View -> Info bar. 
Clicking Simulation -> Run again pauses the simulation. 
Further useful computation functions: 
Snapshots: Clicking Simulation -> Snapshot saves the current simulation state in memory. You can continue running the simulation and return to the saved state by invoking Simulation -> Restore. 
Single time step forward and backward: If a simulation is paused, a single time step can be calculated via Simulation -> Step forward. The special feature of this function is that the old state is saved and one can go back one time step with Simulation -> Step back. This also works several times in a row. 
Display link: By switching off View -> Display link you can disable the rendering and thus speed up the whole simulation. 
Accelerate active clusters: It is possible to allocate more computing power to active clusters (cell clusters that have tokens) by periodically freezing the remaining content for a certain number of time steps. This can be achieved by toggling Simulation -> Accelerate active clusters. 
One of the most important manipulation functions are the general settings and the simulation parameters: 
General settings: The world size of a simulation can be changed in Settings -> General settings. There are many more technical parameters that we can leave unchanged for now. They are used for the CUDA api and specify the number of threads and sizes of arrays. If scale content is enabled in the dialog, additional space created by a possible world enlargement will be periodically augmented with the existing cells and particles. 
Simulation parameters: The parameters can be set by opening the Settings -> Simulation parameters dialog. Let us give some selected examples to play around with: 
cell - max force: Sets the threshold at which damage occurs. If you lower the value, the bodies can break more easily. 
cell - fusion velocity: Sets the threshold at which fusion occurs. If you lower the value, the bodies can fuse more easily. 
radiation -> factor: Corresponds to the amount of energy that a cell emits from time to time. Higher values mean that the cells are destroyed more quickly. 
weapon - energy cost: Corresponds to the amount of energy a cell loses when it tries to attack its surroundings. Higher values make it more difficult for energy-consuming cell clusters to survive. 
cell - function - constructor -> mutation probability: Sets the different mutation rates. This is mainly relevant for evolution simulations. 
An overview of all parameters can be found in Help -> Documentation. 

Assembling own worlds 
After you have learned the most important global functions, it may be time to build your own world. There are many tutorials in Help -> Documentation that explain certain aspects. We will demonstrate a simple example: 
Create a new simulation by clicking on Simulation -> New. There you can choose the computation and simulation parameter settings. We may continue with the default values. Now we are able to add bodies. A primitive such as a disc can be added by Collection -> New disc. After doing this, we can simply multiply the structure by invoking Collection -> Random multiplier. In the dialog that appears one can set random velocities for the copies. For instance, a (angular) velocity range of -0.3 to 0.3 could be chosen. In the next step, more clusters could be added. To do this, we navigate to a free position and add a saved collection with Collection -> Load. In the dialog we may select examples/collections/spiral builder.aco and multiply it again as described above. If you want to move certain cell clusters by hand, you can do it by dragging and dropping with the mouse pointer in the action mode. 
When our construction is ready, we should take a snapshot (or save the simulation as a file) and run it to see the result. 

Creating worlds with own machines 
In order to assemble your own machines, you need a detailed knowledge of the cell functions and the information processing model described in Help -> Documentation. As a good starting point, one can load an example collection, e.g. examples/collections/spiral builder.aco, and study and manipulate it with the editor. The editor allows you to add, delete, rotate, and move cells as well as entire cell clusters using the mouse pointer and toolbar buttons. In addition, cells are equipped with functions. For example, a cell can be programmed with a computer function by a special machine language, or a weapon function, which attack its environment to steal energy from possible other cells. */