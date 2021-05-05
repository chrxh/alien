#pragma once
#include <QMainWindow>

#include "EngineInterface/Definitions.h"

#include "Web/Definitions.h"

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

    virtual void init(
        MainModel* model,
        MainController* controller,
        Serializer* serializer,
        DataRepository* repository,
        Notifier* notifier, 
        WebSimulationController* webSimController,
        StartupController* versionController);

    virtual void initGettingStartedWindow();

    virtual void refresh();

	virtual void setupEditors(SimulationController* controller, SimulationAccess* access);
	virtual GeneralInfoController* getInfoController() const;
    virtual MonitorController* getMonitorController() const;

    virtual void toggleGettingStarted(bool show);
    virtual void toggleInfobar(bool show);
    virtual void showDocumentation();

protected:
	virtual void resizeEvent(QResizeEvent *event);
	virtual void closeEvent(QCloseEvent* event);

private:
	void setupMenuAndToolbar();
	void setupFontsAndColors();
	void setupWidgets();
	void setupFullScreen();
    void setupStartupWidget();

private:
    Q_SLOT void infobarChanged(bool show);
    Q_SLOT void gettingStartedWindowClosed();

private:
	Ui::MainView* ui = nullptr;
	SimulationViewController* _simulationViewController = nullptr;

	MainModel* _model = nullptr;
	MainController* _controller = nullptr;
	ActionController* _actions = nullptr;
	DataRepository* _repository = nullptr;
	Notifier* _notifier = nullptr;

	DataEditController* _dataEditor = nullptr;
	ToolbarController* _toolbar = nullptr;
	GeneralInfoController* _infoController = nullptr;
	MonitorController* _monitor = nullptr;
    LoggingController* _logging = nullptr;
    StartupController* _startupController = nullptr;

    GettingStartedWindow* _gettingStartedWindow = nullptr;

	bool _initialied = false;
    bool _closing = false;
};
