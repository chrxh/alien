#include "CreatureEditorWindow.h"

CreatureEditorWindow::CreatureEditorWindow()
    : AlienWindow("Creature editor", "windows.creature editor", false)
{
}

void CreatureEditorWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void CreatureEditorWindow::shutdownIntern()
{
}

void CreatureEditorWindow::processIntern()
{
}
