#pragma once
#include <QMainWindow>

#include "ModelBasic/Definitions.h"

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
        WebSimulationController* webSimController);

    virtual void initGettingStartedWindow();

    virtual void refresh();

	virtual void setupEditors(SimulationController* controller, SimulationAccess* access);
	virtual InfoController* getInfoController() const;

    virtual void toggleGettingStarted(bool show);
    virtual void showDocumentation();

protected:
	virtual void resizeEvent(QResizeEvent *event);
	virtual void closeEvent(QCloseEvent* event);

private:
	void setupMenuAndToolbar();
	void setupFontsAndColors();
	void setupWidgets();
	void setupFullScreen();

private:
	Q_SLOT void monitorClosed();
    Q_SLOT void gettingStartedWindowClosed();

private:
	Ui::MainView* ui = nullptr;
	VisualEditController* _visualEditor = nullptr;

	MainModel* _model = nullptr;
	MainController* _controller = nullptr;
	ActionController* _actions = nullptr;
	DataRepository* _repository = nullptr;
	Notifier* _notifier = nullptr;

	DataEditController* _dataEditor = nullptr;
	ToolbarController* _toolbar = nullptr;
	InfoController* _infoController = nullptr;
	MonitorController* _monitor = nullptr;

    GettingStartedWindow* _gettingStartedWindow = nullptr;

	bool _initialied = false;
};
