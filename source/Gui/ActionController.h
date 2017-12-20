#pragma once
#include <QObject>

#include "Definitions.h"

class ActionController
	: public QObject
{
	Q_OBJECT

public:
	ActionController(QObject * parent = nullptr);
	virtual ~ActionController() = default;

	virtual void init(MainController* mainController, MainModel* mainModel, MainView* mainView, VisualEditController* visualEditor
		, Serializer* serializer, InfoController* infoController, DataEditController* dataEditor, ToolbarController* toolbar);

	virtual ActionHolder* getActionHolder();

private:
	Q_SLOT void onSetEditorMode();
	Q_SLOT void onRunClicked(bool run);
	Q_SLOT void onStepForward();
	Q_SLOT void onStepBackward();
	Q_SLOT void onMakeSnapshot();
	Q_SLOT void onRestoreSnapshot();
	Q_SLOT void onZoomInClicked();
	Q_SLOT void onZoomOutClicked();
	Q_SLOT void onNewSimulation();
	Q_SLOT void onSaveSimulation();
	Q_SLOT void onLoadSimulation();
	Q_SLOT void onEditSimulationParameters();
	Q_SLOT void onLoadSimulationParameters();
	Q_SLOT void onSaveSimulationParameters();
	Q_SLOT void onEditSymbolTable();
	Q_SLOT void onLoadSymbolTable();
	Q_SLOT void onSaveSymbolTable();

	void updateZoomFactor();

	ActionHolder* _actions = nullptr;
	MainController* _mainController = nullptr;
	MainModel* _mainModel = nullptr;
	MainView* _mainView = nullptr;
	VisualEditController* _visualEditor = nullptr;
	Serializer* _serializer = nullptr;
	InfoController* _infoController = nullptr;
	DataEditController* _dataEditor = nullptr;
	ToolbarController* _toolbar = nullptr;
};
