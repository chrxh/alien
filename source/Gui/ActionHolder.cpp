#include <QAction>

#include "ActionHolder.h"

ActionHolder::ActionHolder(QObject* parent) : QObject(parent)
{
	actionNewSimulation = new QAction("New", this);
	actionNewSimulation->setEnabled(true);
	actionLoadSimulation = new QAction("Load", this);
	actionLoadSimulation->setEnabled(true);
	actionSaveSimulation = new QAction("Save", this);
	actionSaveSimulation->setEnabled(true);
	actionRunSimulation = new QAction("Run", this);
	actionRunSimulation->setEnabled(true);
	actionRunSimulation->setCheckable(true);
	QIcon runIcon;
	runIcon.addFile(":/Icons/play.png", QSize(), QIcon::Normal, QIcon::Off);
	actionRunSimulation->setIcon(runIcon);
	actionRunSimulation->setIconVisibleInMenu(false);
	actionRunStepForward = new QAction("Step forward", this);
	actionRunStepForward->setEnabled(true);
	QIcon stepForwardIcon;
	stepForwardIcon.addFile(":/Icons/step.png", QSize(), QIcon::Normal, QIcon::Off);
	actionRunStepForward->setIcon(stepForwardIcon);
	actionRunStepForward->setIconVisibleInMenu(false);
	actionRunStepBackward = new QAction("Step backward", this);
	actionRunStepBackward->setEnabled(false);
	QIcon stepBackwardIcon;
	stepBackwardIcon.addFile(":/Icons/step_back.png", QSize(), QIcon::Normal, QIcon::Off);
	actionRunStepBackward->setIcon(stepBackwardIcon);
	actionRunStepBackward->setIconVisibleInMenu(false);
	actionSnapshot = new QAction("Snapshot", this);
	actionSnapshot->setEnabled(true);
	QIcon snapshotIcon;
	snapshotIcon.addFile(":/Icons/snapshot.png", QSize(), QIcon::Normal, QIcon::Off);
	actionSnapshot->setIcon(snapshotIcon);
	actionSnapshot->setIconVisibleInMenu(false);
	actionRestore = new QAction("Restore", this);
	actionRestore->setEnabled(false);
	QIcon restoreIcon;
	restoreIcon.addFile(":/Icons/restore_active.png", QSize(), QIcon::Normal, QIcon::Off);
	actionRestore->setIcon(restoreIcon);
	actionRestore->setIconVisibleInMenu(false);
	actionExit = new QAction("Exit", this);
	actionExit->setEnabled(true);

	actionEditSimParameters = new QAction("Edit", this);
	actionEditSimParameters->setEnabled(true);
	actionLoadSimParameters = new QAction("Load", this);
	actionLoadSimParameters->setEnabled(true);
	actionSaveSimParameters = new QAction("Save", this);
	actionSaveSimParameters->setEnabled(true);

	actionEditSymbols = new QAction("Edit", this);
	actionEditSymbols->setEnabled(true);
	actionLoadSymbols = new QAction("Load", this);
	actionLoadSymbols->setEnabled(true);
	actionSaveSymbols = new QAction("Save", this);
	actionSaveSymbols->setEnabled(true);
	actionMergeWithSymbols = new QAction("Merge with", this);
	actionMergeWithSymbols->setEnabled(true);

	actionEditor = new QAction("Editor", this);
	actionEditor->setEnabled(true);
	QIcon editorIcon;
	editorIcon.addFile(":/Icons/EditorView.png", QSize(), QIcon::Normal, QIcon::Off);
	actionEditor->setIcon(editorIcon);
	actionEditor->setIconVisibleInMenu(false);
	actionMonitor = new QAction("Monitor", this);
	actionMonitor->setEnabled(true);
	QIcon monitorIcon;
	monitorIcon.addFile(":/Icons/monitor.png", QSize(), QIcon::Normal, QIcon::Off);
	actionMonitor->setIcon(monitorIcon);
	actionMonitor->setIconVisibleInMenu(false);
	actionZoomIn = new QAction("Zoom in", this);
	actionZoomIn->setEnabled(true);
	QIcon zoomInIcon;
	zoomInIcon.addFile(":/Icons/zoom_in.png", QSize(), QIcon::Normal, QIcon::Off);
	actionZoomIn->setIcon(zoomInIcon);
	actionZoomIn->setIconVisibleInMenu(false);
	actionZoomOut = new QAction("Zoom out", this);
	actionZoomOut->setEnabled(true);
	QIcon zoomOutIcon;
	zoomOutIcon.addFile(":/Icons/zoom_out.png", QSize(), QIcon::Normal, QIcon::Off);
	actionZoomOut->setIcon(zoomOutIcon);
	actionZoomOut->setIconVisibleInMenu(false);
	actionFullscreen = new QAction("Fullscreen", this);
	actionFullscreen->setEnabled(true);
	actionFullscreen->setCheckable(true);
	actionFullscreen->setChecked(true);

	actionNewCell = new QAction("New cell", this);
	actionNewCell->setEnabled(true);
	actionNewParticle = new QAction("New particle", this);;
	actionNewParticle->setEnabled(true);
	actionCopyEntity = new QAction("Copy entity", this);;
	actionCopyEntity->setEnabled(false);
	actionPasteEntity = new QAction("Paste entity", this);
	actionPasteEntity->setEnabled(false);
	actionDeleteEntity = new QAction("Delete entity", this);
	actionDeleteEntity->setEnabled(false);
	actionNewToken = new QAction("New token", this);
	actionNewToken->setEnabled(false);
	actionCopyToken = new QAction("Copy token", this);
	actionCopyToken->setEnabled(false);
	actionPasteToken = new QAction("Paste token", this);
	actionPasteToken->setEnabled(false);
	actionDeleteToken = new QAction("Delete token", this);
	actionDeleteToken->setEnabled(false);

	actionNewRectangle = new QAction("Rectangle", this);
	actionNewRectangle->setEnabled(true);
	actionNewHexagon = new QAction("Hexagon", this);
	actionNewHexagon->setEnabled(true);
	actionNewParticles = new QAction("Particles", this);
	actionNewParticles->setEnabled(true);
	actionLoadCol = new QAction("Load", this);
	actionLoadCol->setEnabled(true);
	actionSaveCol = new QAction("Save", this);
	actionSaveCol->setEnabled(false);
	actionCopyCol = new QAction("Copy", this);
	actionCopyCol->setEnabled(false);
	actionPasteCol = new QAction("Paste", this);
	actionPasteCol->setEnabled(false);
	actionDeleteCol = new QAction("Delete", this);
	actionDeleteCol->setEnabled(false);
	actionMultiplyRandom = new QAction("Random", this);
	actionMultiplyRandom->setEnabled(false);
	actionMultiplyArrangement = new QAction("Arrangement", this);
	actionMultiplyArrangement->setEnabled(false);

	actionAbout = new QAction("About artificial life environment (alien)", this);
	actionAbout->setEnabled(true);
	actionDocumentation = new QAction("Documentation", this);
	actionDocumentation->setEnabled(true);
	actionDocumentation->setCheckable(true);
}

