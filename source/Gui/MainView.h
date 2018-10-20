#pragma once
#include <QMainWindow>

#include "ModelBasic/Definitions.h"

#include "Definitions.h"

namespace Ui {
	class MainView;
}

class MainView
	: public QMainWindow
{
	Q_OBJECT

public:
	MainView(QWidget * parent = nullptr);
	virtual ~MainView();

	virtual void init(MainModel* model, MainController* controller, Serializer* serializer, DataRepository* repository
		, SimulationMonitor* simMonitor, Notifier* notifier);


	virtual void refresh();

	virtual void setupEditors(SimulationController* controller, SimulationAccess* access);
	virtual InfoController* getInfoController() const;

	virtual void showDocumentation(bool show);

protected:
	virtual void resizeEvent(QResizeEvent *event);
	virtual void closeEvent(QCloseEvent* event);

private:
	void setupMenu();
	void setupFontsAndColors();
	void setupWidgets();
	void setupFullScreen();

private:
	Q_SLOT void documentationWindowClosed();
	Q_SLOT void monitorClosed();

private:
	Ui::MainView* ui = nullptr;
	VisualEditController* _visualEditor = nullptr;
	DocumentationWindow* _documentationWindow = nullptr;

	MainModel* _model = nullptr;
	MainController* _controller = nullptr;
	ActionController* _actions = nullptr;
	DataRepository* _repository = nullptr;
	Notifier* _notifier = nullptr;

	DataEditController* _dataEditor = nullptr;
	ToolbarController* _toolbar = nullptr;
	InfoController* _infoController = nullptr;
	MonitorController* _monitor = nullptr;

	StartScreenController* _startScreen = nullptr;
	bool _initialied = false;
};
