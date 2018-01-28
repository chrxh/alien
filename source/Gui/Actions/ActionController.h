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
		, Serializer* serializer, InfoController* infoController, DataEditController* dataEditor, ToolbarController* toolbar
		, DataRepository* repository, Notifier* notifier);

	virtual ActionHolder* getActionHolder();

private:
	Q_SLOT void onNewSimulation();
	Q_SLOT void onSaveSimulation();
	Q_SLOT void onLoadSimulation();
	Q_SLOT void onConfig();
	Q_SLOT void onRunClicked(bool run);
	Q_SLOT void onStepForward();
	Q_SLOT void onStepBackward();
	Q_SLOT void onMakeSnapshot();
	Q_SLOT void onRestoreSnapshot();

	Q_SLOT void onSetEditorMode(bool editMode);
	Q_SLOT void onZoomInClicked();
	Q_SLOT void onZoomOutClicked();
	Q_SLOT void onEditSimulationParameters();
	Q_SLOT void onLoadSimulationParameters();
	Q_SLOT void onSaveSimulationParameters();
	Q_SLOT void onEditSymbolTable();
	Q_SLOT void onLoadSymbolTable();
	Q_SLOT void onSaveSymbolTable();

	Q_SLOT void onNewCell();
	Q_SLOT void onNewParticle();
	Q_SLOT void onLoadCollection();
	Q_SLOT void onSaveCollection();
	Q_SLOT void onCopyCollection();
	Q_SLOT void onPasteCollection();
	Q_SLOT void onDeleteSelection();
	Q_SLOT void onDeleteCollection();
	Q_SLOT void onNewToken();
	Q_SLOT void onDeleteToken();
	Q_SLOT void onToggleCellInfo(bool showInfo);

	Q_SLOT void onNewRectangle();

	Q_SLOT void receivedNotifications(set<Receiver> const& targets);

	void updateZoomFactor();
	void updateActionsEnableState();

	ActionModel* _model = nullptr;
	MainController* _mainController = nullptr;
	MainModel* _mainModel = nullptr;
	MainView* _mainView = nullptr;
	DataRepository* _repository = nullptr;
	Notifier* _notifier = nullptr;
	Serializer* _serializer = nullptr;

	VisualEditController* _visualEditor = nullptr;
	DataEditController* _dataEditor = nullptr;
	InfoController* _infoController = nullptr;
	ToolbarController* _toolbar = nullptr;
};
