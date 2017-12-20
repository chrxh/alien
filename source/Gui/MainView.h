#pragma once
#include <QMainWindow>

#include "Model/Api/Definitions.h"

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

	virtual void init(MainModel* model, MainController* controller, Serializer* serializer);
	virtual void refresh();

	virtual void setupEditors(SimulationController* controller, DataRepository* manipulator, Notifier* notifier);
	virtual InfoController* getInfoController() const;

private:
	void setupMenu();
	void setupTheme();

	Ui::MainView* ui = nullptr;
	VisualEditController* _visualEditor = nullptr;
	MainModel* _model = nullptr;
	MainController* _controller = nullptr;
	Serializer* _serializer = nullptr;
	ActionController* _actions = nullptr;

	DataEditController* _dataEditor = nullptr;
	ToolbarController* _toolbar = nullptr;
	InfoController* _infoController = nullptr;
};
