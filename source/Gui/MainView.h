#pragma once
#include <QMainWindow>

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

private:
	Ui::MainView *ui;
	MainModel* _model = nullptr;
	MainController* _controller = nullptr;
};
