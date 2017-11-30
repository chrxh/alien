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

	void init(MainModel* model, MainController* controller);

	void setupEditors(SimulationController* controller, DataManipulator* manipulator, Notifier* notifier);

private:
	void connectActions();
	void setupFont();
	void setupPalette();

	Ui::MainView* ui = nullptr;	//contains VisualEditor
	MainModel* _model = nullptr;
	MainController* _controller = nullptr;

	DataEditController* _dataEditor = nullptr;
	ToolbarController* _toolbar = nullptr;
};
