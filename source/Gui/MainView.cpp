#include "MainView.h"

#include "ui_MainView.h"

MainView::MainView(QWidget * parent)
	: QMainWindow(parent)
	, ui(new Ui::MainView)
{
	ui->setupUi(this);
}

MainView::~MainView()
{
	delete ui;

}

void MainView::init(MainModel * model, MainController * controller)
{
	_model = model;
	_controller = controller;

	setWindowState(windowState() | Qt::WindowFullScreen);
	show();
}
